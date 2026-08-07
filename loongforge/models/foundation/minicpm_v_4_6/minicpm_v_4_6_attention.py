# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MiniCPM-V-4.6 gated softmax attention."""

import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Tuple, Union

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import build_module
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import (
    deprecate_inference_params,
    is_fa_min_version,
    nvtx_range_pop,
    nvtx_range_push,
)
from .peft import torch_linear_forward

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
    )
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None

try:
    from flashattn_hopper.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )

    HAVE_FA3 = True
except:
    HAVE_FA3 = False


class MiniCPMV46SelfAttention(SelfAttention):
    """
    Initialize Qwen3Next self-attention module

    Args:
        config: Transformer configuration object, containing model parameters
        submodules: Self-attention submodule configuration, used to build various components
        *args: Extra positional arguments passed to the parent class
        **kwargs: Extra keyword arguments passed to the parent class
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        *args,
        projection_split_mode: str = "merged",
        linear_backend: str = "transformer_engine",
        **kwargs,
    ):
        super().__init__(config, submodules, *args, **kwargs)
        self.projection_split_mode = projection_split_mode
        if linear_backend not in ("transformer_engine", "auto", "torch"):
            raise ValueError(
                "linear_backend must be one of: transformer_engine, auto, torch"
            )
        tp_size = self.pg_collection.tp.size()
        if linear_backend == "torch" and tp_size != 1:
            raise ValueError("The torch linear backend requires tensor parallel size 1")
        self.use_torch_linear = linear_backend == "torch" or (
            linear_backend == "auto" and tp_size == 1
        )
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            2 * self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
        )
        self.linear_qkv.canonical_split_sizes = {
            "linear_q": 2 * self.query_projection_size,
            "linear_k": self.kv_projection_size,
            "linear_v": self.kv_projection_size,
        }

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    @staticmethod
    def _expand_2d_padding_causal_mask(attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Build an explicit causal + key padding mask from a 2D padding mask."""
        if attention_mask is None or attention_mask.dim() != 2:
            return attention_mask
        if attention_mask.shape[-1] != seq_len:
            return attention_mask

        batch_size = attention_mask.shape[0]
        key_padding_mask = attention_mask[:, None, None, :]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len)
        return causal_mask | key_padding_mask.expand(batch_size, 1, seq_len, seq_len)

    def _prepare_full_attention_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Prepare 2D padding masks for the selected core-attention backend."""
        if attention_mask is None or attention_mask.dim() != 2:
            return attention_mask
        expected_seq_len = seq_len * self.pg_collection.cp.size()
        if attention_mask.shape[-1] not in (seq_len, expected_seq_len):
            return attention_mask
        if self.pg_collection.cp.size() > 1:
            padding = attention_mask.bool()
            if torch.any(padding[:, :-1] & ~padding[:, 1:]):
                raise ValueError(
                    "context parallel attention requires right-padded sequences"
                )
            # TE CP does not accept padding masks. With causal right padding,
            # valid queries cannot attend to padded keys and padded queries are
            # removed by the loss mask.
            return None
        if hasattr(self.core_attention, "te_forward_mask_type"):
            return attention_mask[:, None, None, :]
        if attention_mask.shape[-1] != seq_len:
            return attention_mask
        return self._expand_2d_padding_causal_mask(attention_mask, seq_len)

    def _hf_eager_core_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """HF Qwen3.5 eager attention equivalent for full-attention layers."""
        query = query.permute(1, 2, 0, 3).contiguous()
        key = key.permute(1, 2, 0, 3).contiguous()
        value = value.permute(1, 2, 0, 3).contiguous()

        num_key_value_groups = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        if num_key_value_groups > 1:
            key = key.repeat_interleave(num_key_value_groups, dim=1)
            value = value.repeat_interleave(num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * (
            self.hidden_size_per_attention_head ** -0.5
        )
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(
                    attention_mask, torch.finfo(attn_weights.dtype).min
                )
            else:
                attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_probs = F.dropout(
            attn_probs,
            p=self.config.attention_dropout,
            training=self.training,
        )
        context_layer = torch.matmul(attn_probs, value)
        context_layer = context_layer.transpose(1, 2).contiguous()
        return context_layer.permute(1, 0, 2, 3).contiguous().view(
            query.size(2),
            query.size(0),
            -1,
        )

    def _hf_sdpa_core_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """HF Qwen3.5 SDPA attention equivalent for full-attention layers."""
        query = query.permute(1, 2, 0, 3).contiguous()
        key = key.permute(1, 2, 0, 3).contiguous()
        value = value.permute(1, 2, 0, 3).contiguous()

        num_key_value_groups = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )

        attn_mask = None
        is_causal = True
        has_padding = (
            attention_mask is not None
            and attention_mask.dim() == 2
            and torch.any(attention_mask.bool())
        )
        if has_padding:
            seq_len = query.size(2)
            key_keep = (~attention_mask.bool())[:, None, None, :]
            causal_keep = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device)
            ).view(1, 1, seq_len, seq_len)
            attn_mask = key_keep & causal_keep
            is_causal = False
        enable_gqa = num_key_value_groups > 1 and attn_mask is None
        if num_key_value_groups > 1 and not enable_gqa:
            key = key.repeat_interleave(num_key_value_groups, dim=1)
            value = value.repeat_interleave(num_key_value_groups, dim=1)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            scale=self.hidden_size_per_attention_head ** -0.5,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
        context_layer = context_layer.transpose(1, 2).contiguous()
        return context_layer.view(query.size(0), query.size(2), -1).permute(1, 0, 2).contiguous()

    def _hf_qwen35_apply_rotary_pos_emb(
        self,
        tensor: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Qwen3.5 text RoPE with HF's BF16 arithmetic order."""
        if self.config.rotary_interleaved:
            return apply_rotary_pos_emb(
                tensor,
                freqs,
                config=self.config,
                cp_group=self.pg_collection.cp,
            )

        if freqs.dim() == tensor.dim() + 1 and freqs.size(-2) == 1:
            freqs = freqs.squeeze(-2)

        rotary_dim = freqs.shape[-1]
        tensor_rot = tensor[..., :rotary_dim]
        tensor_pass = tensor[..., rotary_dim:]
        tensor_rot_1, tensor_rot_2 = torch.chunk(tensor_rot, 2, dim=-1)
        tensor_rotated = torch.cat((-tensor_rot_2, tensor_rot_1), dim=-1)

        cos = torch.cos(freqs).to(tensor.dtype)
        sin = torch.sin(freqs).to(tensor.dtype)
        tensor_embed = (tensor_rot * cos) + (tensor_rotated * sin)
        return torch.cat((tensor_embed, tensor_pass), dim=-1)

    def _split_qwen35_qgkv_weights(self):
        linear_qkv = getattr(self.linear_qkv, "to_wrap", self.linear_qkv)
        weight = linear_qkv.weight
        num_querys_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        qg_dim = 2 * num_querys_per_group * self.hidden_size_per_attention_head
        kv_dim = self.hidden_size_per_attention_head
        grouped = weight.reshape(
            self.num_query_groups_per_partition,
            qg_dim + 2 * kv_dim,
            self.config.hidden_size,
        )
        qg = grouped[:, :qg_dim, :].reshape(-1, self.config.hidden_size)
        key = grouped[:, qg_dim : qg_dim + kv_dim, :].reshape(-1, self.config.hidden_size)
        value = grouped[:, qg_dim + kv_dim :, :].reshape(-1, self.config.hidden_size)
        return qg, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        if (
            self.use_torch_linear
            and packed_seq_params is not None
            and packed_seq_params.cu_seqlens_q.numel() == 2
        ):
            packed_seq_params = None

        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False)
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                '2.7.3'), 'flash attn verion v2.7.3 and above is required for dynamic batching.'

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, ) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix='qkv')
        query, key, value, gate = self.get_query_key_value_tensors(hidden_states, key_value_states)
        nvtx_range_pop(suffix='qkv')
        use_hf_eager_attention = False
        use_hf_sdpa_attention = (
            getattr(self.config, "qwen35_hf_sdpa_attention", False)
            and self.projection_split_mode == "qwen3_5"
            and packed_seq_params is None
            and inference_context is None
            and attention_bias is None
            and self.pg_collection.cp.size() == 1
        )
        use_hf_qwen35_rope = (
            getattr(self.config, "qwen35_hf_rope", False)
            and self.projection_split_mode == "qwen3_5"
            and packed_seq_params is None
            and inference_context is None
        )
        original_attention_mask = attention_mask
        attention_mask = (
            self._expand_2d_padding_causal_mask(attention_mask, query.size(0))
            if use_hf_eager_attention
            else attention_mask
            if use_hf_sdpa_attention
            else self._prepare_full_attention_mask(attention_mask, query.size(0))
        )

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (inference_context is not None and inference_context.is_decode_only() and not self.training)

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix='adjust_key_value')
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[self.layer_number]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if (in_decode_mode and self.config.enable_cuda_graph and inference_context.is_static_batching()):
            raise ValueError('CUDA graphs must use flash decode with static batching!')

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            ))

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix='adjust_key_value')

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix='rotary_pos_emb')
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if use_hf_qwen35_rope:
                    query = self._hf_qwen35_apply_rotary_pos_emb(query, q_pos_emb)
                elif inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        cp_group=self.pg_collection.cp,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(query, q_pos_emb, self.config, cu_seqlens_q,
                                                                     self.pg_collection.cp)
            if k_pos_emb is not None:
                if use_hf_qwen35_rope:
                    key = self._hf_qwen35_apply_rotary_pos_emb(key, k_pos_emb)
                else:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        cp_group=self.pg_collection.cp,
                    )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix='rotary_pos_emb')

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix='core_attention')
        if use_hf_eager_attention:
            core_attn_out = self._hf_eager_core_attention(
                query,
                key,
                value,
                attention_mask,
            )
        elif use_hf_sdpa_attention:
            core_attn_out = self._hf_sdpa_core_attention(
                query,
                key,
                value,
                original_attention_mask,
            )
        elif self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = (inference_context.cu_kv_lengths())

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix='core_attention')

        # =================
        # Output. [sq, b, h]
        # =================

        core_attn_out = core_attn_out * torch.sigmoid(gate.reshape_as(core_attn_out))
        nvtx_range_push(suffix='linear_proj')
        if self.use_torch_linear:
            output, bias = torch_linear_forward(self.linear_proj, core_attn_out)
        else:
            output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')

        return output, bias

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives query, key, value, and gate tensors from the input hidden states.
        """
        if self.projection_split_mode == "qwen3_5":
            projection_input = hidden_states
            if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
                projection_input = gather_from_sequence_parallel_region(hidden_states)
            qg_weight, key_weight, value_weight = self._split_qwen35_qgkv_weights()
            mixed_qg = F.linear(projection_input, qg_weight)
            if hasattr(self.linear_qkv, "adapter_output"):
                adapter_output = self.linear_qkv.adapter_output("q", hidden_states)
                if adapter_output is not None:
                    mixed_qg = mixed_qg + adapter_output.reshape(mixed_qg.shape)

            key = F.linear(projection_input, key_weight)
            if hasattr(self.linear_qkv, "adapter_output"):
                adapter_output = self.linear_qkv.adapter_output("k", hidden_states)
                if adapter_output is not None:
                    key = key + adapter_output.reshape(key.shape)

            value = F.linear(projection_input, value_weight)
            if hasattr(self.linear_qkv, "adapter_output"):
                adapter_output = self.linear_qkv.adapter_output("v", hidden_states)
                if adapter_output is not None:
                    value = value + adapter_output.reshape(value.shape)
            query, gate = torch.chunk(
                mixed_qg.view(
                    *mixed_qg.size()[:-1],
                    -1,
                    self.hidden_size_per_attention_head * 2,
                ),
                2,
                dim=-1,
            )
            query = query.reshape(
                query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
            )
            gate = gate.reshape(
                gate.size(0), gate.size(1), -1, self.hidden_size_per_attention_head
            )
            key = key.view(
                key.size(0),
                key.size(1),
                -1,
                self.hidden_size_per_attention_head,
            )
            value = value.view(
                value.size(0),
                value.size(1),
                -1,
                self.hidden_size_per_attention_head,
            )

            if self.q_layernorm is not None:
                query = self.q_layernorm(query)

            if self.k_layernorm is not None:
                key = self.k_layernorm(key)

            if self.config.test_mode:
                self.run_realtime_tests()

            return query, key, value, gate

        mixed_qgkv, _ = self.linear_qkv(hidden_states)

        new_tensor_shape = mixed_qgkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition * 2 + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qgkv = mixed_qgkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head * 2),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_gate, key, value) = SplitAlongDim(mixed_qgkv, 3, split_arg_list)
        else:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_gate, key, value) = torch.split(mixed_qgkv, split_arg_list, dim=3)

        query_gate = query_gate.reshape(query_gate.size(0), query_gate.size(1), -1, self.hidden_size_per_attention_head)
        query = query_gate[:, :, ::2]
        gate = query_gate[:, :, 1::2]

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate
