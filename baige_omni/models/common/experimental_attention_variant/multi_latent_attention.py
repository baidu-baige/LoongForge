# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Omni MLA wrapper carrying DSA absorb logic with CLI-gated fused routing."""

import copy
import math
from typing import Optional

import torch

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params
from megatron.core.fp8_utils import is_float8tensor

try:
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
except ImportError:
    TEGroupedLinear = None

from .dsa_fused_kernels import (
    fused_apply_mla_rope,
    fused_apply_mla_rope_for_absorb_kv,
    fused_rope_permute_cat,
)

class MLASelfAttentionFused(MLASelfAttention):
    """Omni-side MLA class that preserves the rolled-back DSA absorb path."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        if config.enable_chunkpipe:
            patched_config = config
        else:
            patched_config = copy.copy(config)

        patched_config.experimental_attention_variant = "dsa"
        super().__init__(
            config=patched_config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        # Note: chunkpipe cache is initialized in parent class MLASelfAttention.__init__
        # via init_chunk_key_value_cache_mla() when config.enable_chunkpipe is True

        if self.config.enable_chunkpipe:
            self.v_channels = self.config.v_head_dim
            self.padding_v_head_dim = False

        if TEGroupedLinear is None:
            raise ImportError(
                "--use-dsa-fused requires TEGroupedLinear from transformer_engine."
            )

        self.linear_kv_up_proj_absorb_q = build_module(
            TEGroupedLinear,
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim,
            self.config.kv_lora_rank,
            parallel_mode=None,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj_absorb_q",
        )
        self.linear_kv_up_proj_absorb_output = build_module(
            TEGroupedLinear,
            self.num_attention_heads_per_partition,
            self.config.kv_lora_rank,
            self.config.v_head_dim,
            parallel_mode=None,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj_absorb_output",
        )

        self.register_load_state_dict_post_hook(self.initialize_kv_absorb_weights)
        self.register_state_dict_pre_hook(self.update_linear_kv_up_proj)
        self.linear_kv_up_proj.weight.requires_grad = False

        # Eagerly materialize absorb weights from linear_kv_up_proj during
        # module construction so the optimizer sees the real derived values
        # instead of placeholder initialization. For SP-first, doing this in
        # the constructor also keeps the TP collective order deterministic.
        self.initialize_kv_absorb_weights(None, None)

    def concat_cached_chunk_key_value_dsa(self, curr_key):
        """Concatenate all cached key chunks for DSA path in chunkpipe.
        
        Unlike standard MLA, DSA stores key as [kv_compressed, k_pos_emb] directly
        without up-projection, and value is always None.
        
        Args:
            attention_mask (Tensor): Attention mask tensor for the current chunk
            curr_key (Tensor): Current chunk's key tensor with shape 
                [chunksize, batch_size, kv_lora_rank + qk_pos_emb_head_dim]
        
        Returns:
            tuple: (concatenated_key, None)
        """
        if not self.config.enable_chunkpipe:
            raise RuntimeError("Chunk concatenation requires chunkpipe to be enabled.")
        
        is_forward = self.config.chunkpipe_forward
        microbatch_idx = (self.config.chunkpipe_forward_microbatch if is_forward 
                         else self.config.chunkpipe_backward_microbatch)
        current_chunk_idx = microbatch_idx % self.num_chunks_per_seq
        start_microbatch_idx = microbatch_idx - current_chunk_idx

        # Calculate total sequence length after concatenation
        total_concatenated_tokens = (current_chunk_idx + 1) * self.config.chunksize
        # DSA key shape: [seq, batch, kv_lora_rank + qk_pos_emb_head_dim]
        key_hidden_size = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        concatenated_key_shape = (total_concatenated_tokens, self.config.micro_batch_size, key_hidden_size)
        
        concatenated_key = torch.zeros(concatenated_key_shape,
            device=self.kv_compressed_cache.device, dtype=self.kv_compressed_cache.dtype)

        def kv_compressed_hook_fn(chunk_index):
            """Hook function to accumulate gradient for cached compressed KV."""
            def hook_fn(grad):
                if chunk_index not in self.kv_compressed_cache_grad:
                    self.kv_compressed_cache_grad[chunk_index] = grad
                else:
                    self.kv_compressed_cache_grad[chunk_index] += grad
                return grad           
            return hook_fn

        def key_pos_emb_hook_fn(chunk_index):
            """Hook function to accumulate gradient for cached key position embeddings."""
            def hook_fn(grad):
                if chunk_index not in self.key_pos_emb_cache_grad:
                    self.key_pos_emb_cache_grad[chunk_index] = grad
                else:
                    self.key_pos_emb_cache_grad[chunk_index] += grad
                return grad           
            return hook_fn
        
        # Retrieve all previous chunks from cache
        current_pos = 0
        for prev_chunk_idx in range(current_chunk_idx):
            cache_chunk_idx = self.micro_batch_to_cache_chunk_map[start_microbatch_idx + prev_chunk_idx]
            
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            kv_indices = torch.arange(self.config.chunksize // tp_size) + \
                (cache_chunk_idx * self.config.chunksize // tp_size)
            pos_indices = torch.arange(self.config.chunksize) + (cache_chunk_idx * self.config.chunksize)

            cached_kv_compressed = self.kv_compressed_cache[kv_indices, :, :]
            cached_key_pos_emb = self.key_pos_emb_cache[pos_indices, :, 0:1, :]

            # Set up gradient hooks for backward pass
            if self.is_enable_grad_chunkpipe():
                cached_kv_compressed.requires_grad = True
                cached_key_pos_emb.requires_grad = True
                cached_kv_compressed.register_hook(kv_compressed_hook_fn(prev_chunk_idx))
                cached_key_pos_emb.register_hook(key_pos_emb_hook_fn(prev_chunk_idx))
            
            # For DSA: need to gather kv_compressed if sequence parallel, then cat
            if self.config.sequence_parallel:
                cached_kv_compressed_all = gather_from_sequence_parallel_region(cached_kv_compressed)
            else:
                cached_kv_compressed_all = cached_kv_compressed
            
            # Reconstruct key as [kv_compressed, k_pos_emb]
            cached_key = torch.cat([cached_kv_compressed_all, cached_key_pos_emb.squeeze(1)], dim=-1)
            
            concatenated_key[current_pos : current_pos + self.config.chunksize, :, :] = cached_key
            current_pos += self.config.chunksize

        # Add current chunk's key
        concatenated_key[current_pos : current_pos + self.config.chunksize, :, :] = curr_key
            
        return concatenated_key, None

    def initialize_kv_absorb_weights(self, module, incompatible_keys):
        """Initialize absorb weights from linear_kv_up_proj for checkpoint compatibility."""

        if incompatible_keys is not None:
            # Absorb weights are derived from linear_kv_up_proj at load time.
            absorb_markers = (
                ".linear_kv_up_proj_absorb_q.",
                ".linear_kv_up_proj_absorb_output.",
            )
            incompatible_keys.missing_keys[:] = [
                key
                for key in incompatible_keys.missing_keys
                if not any(marker in key for marker in absorb_markers)
            ]

        assert self.linear_kv_up_proj.parallel_mode == "column", (
            "DSA currently only supports linear_kv_up_proj with column parallel mode. "
            "Row parallel mode support is under development and will be available soon."
        )
        assert self.linear_kv_up_proj.use_bias is False, (
            "DSA currently only supports linear_kv_up_proj with no bias. "
            "Bias support is under development and will be avaliable soon."
        )

        with torch.no_grad():
            kv_up_weight = self.linear_kv_up_proj.weight.clone().detach()
            # In FP8 mode each parameter carries a high-precision init value so the
            # optimizer can build accurate main params without going through the lossy
            # FP8 representation.  Propagate that value into the absorb weights so the
            # optimizer sees accurate BF16/FP32 state for these derived parameters too.
            kv_up_weight_hp = None
            if is_float8tensor(kv_up_weight):
                if hasattr(self.linear_kv_up_proj.weight, 'get_high_precision_init_val'):
                    kv_up_weight_hp = (
                        self.linear_kv_up_proj.weight.get_high_precision_init_val().view(
                            self.num_attention_heads_per_partition, -1, self.config.kv_lora_rank
                        )
                    )
                kv_up_weight = kv_up_weight.dequantize()
            kv_up_weight = kv_up_weight.view(
                self.num_attention_heads_per_partition, -1, self.config.kv_lora_rank
            )

            k_up_proj, v_up_proj = torch.split(
                kv_up_weight, [self.config.qk_head_dim, self.config.v_head_dim], dim=-2
            )
            k_up_proj = k_up_proj.transpose(1, 2).contiguous()
            v_up_proj = v_up_proj.contiguous()

            k_up_proj_hp = v_up_proj_hp = None
            if kv_up_weight_hp is not None:
                k_up_proj_hp, v_up_proj_hp = torch.split(
                    kv_up_weight_hp, [self.config.qk_head_dim, self.config.v_head_dim], dim=-2
                )
                k_up_proj_hp = k_up_proj_hp.transpose(1, 2).contiguous()
                v_up_proj_hp = v_up_proj_hp.contiguous()

            for head_idx in range(self.num_attention_heads_per_partition):
                q_absorb_weight = getattr(self.linear_kv_up_proj_absorb_q, f"weight{head_idx}")
                if is_float8tensor(q_absorb_weight):
                    q_absorb_weight.quantize_(k_up_proj[head_idx])
                    if k_up_proj_hp is not None and hasattr(
                        q_absorb_weight, 'set_high_precision_init_val'
                    ):
                        q_absorb_weight.set_high_precision_init_val(k_up_proj_hp[head_idx])
                else:
                    q_absorb_weight.copy_(k_up_proj[head_idx])

                output_absorb_weight = getattr(
                    self.linear_kv_up_proj_absorb_output, f"weight{head_idx}"
                )
                if is_float8tensor(output_absorb_weight):
                    output_absorb_weight.quantize_(v_up_proj[head_idx])
                    if v_up_proj_hp is not None and hasattr(
                        output_absorb_weight, 'set_high_precision_init_val'
                    ):
                        output_absorb_weight.set_high_precision_init_val(v_up_proj_hp[head_idx])
                else:
                    output_absorb_weight.copy_(v_up_proj[head_idx])

    def update_linear_kv_up_proj(self, module, prefix, keep_vars):
        """Reconstruct linear_kv_up_proj from absorb weights before saving checkpoint."""
        from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensor

        assert self.linear_kv_up_proj.parallel_mode == "column", (
            "DSA currently only supports linear_kv_up_proj with column parallel mode. "
            "Row parallel mode support is under development and will be available soon."
        )
        assert self.linear_kv_up_proj.use_bias is False, (
            "DSA currently only supports linear_kv_up_proj with no bias. "
            "Bias support is under development and will be avaliable soon."
        )

        with torch.no_grad():
            q_absorb_list = []
            output_absorb_list = []
            for head_idx in range(self.num_attention_heads_per_partition):
                head_q_absorb = getattr(
                    self.linear_kv_up_proj_absorb_q, f"weight{head_idx}"
                ).clone().detach()
                if isinstance(head_q_absorb, QuantizedTensor):
                    q_absorb_list.append(head_q_absorb.dequantize())
                else:
                    q_absorb_list.append(head_q_absorb)

                head_output_absorb = getattr(
                    self.linear_kv_up_proj_absorb_output, f"weight{head_idx}"
                ).clone().detach()
                if isinstance(head_output_absorb, QuantizedTensor):
                    output_absorb_list.append(head_output_absorb.dequantize())
                else:
                    output_absorb_list.append(head_output_absorb)

            q_absorb = torch.stack(q_absorb_list, dim=0)
            output_absorb = torch.stack(output_absorb_list, dim=0)
            q_absorb = q_absorb.transpose(1, 2).contiguous()

            kv_up_weight = torch.cat([q_absorb, output_absorb], dim=-2)
            kv_up_weight = kv_up_weight.contiguous().view(-1, self.config.kv_lora_rank)

            if isinstance(self.linear_kv_up_proj.weight, QuantizedTensor):
                self.linear_kv_up_proj.weight.quantize_(kv_up_weight)
            else:
                self.linear_kv_up_proj.weight.copy_(kv_up_weight)

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass with Omni absorb-output projection before linear_proj."""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context and not inference_context.is_static_batching():
            assert (
                self.config.cache_mla_latents
            ), "currently to use dynamic backend for MLA cache mla latents must be true"

        if self.config.cache_mla_latents:
            self.prepare_for_absorption()

        query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        # Chunkpipe: concatenate cached KV chunks with current chunk
        if self.config.enable_chunkpipe:
            key, value = self.concat_cached_chunk_key_value_dsa(key)

        query = query.contiguous()
        key = key.contiguous()
        if value is not None:
            value = value.contiguous()

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if self.offload_core_attention and self.training:
                query = fine_grained_offloading_group_start(query, name="core_attn")

            if inference_context is None or inference_context.is_static_batching():
                extra_kwargs = {
                    "x": hidden_states,
                    "qr": q_compressed,
                }
                with get_fine_grained_offloading_context(self.offload_core_attention):
                    core_attn_out = self.core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        packed_seq_params=packed_seq_params,
                        attn_mask_type=attn_mask_type,
                        **extra_kwargs,
                    )

                m_splits_v = [math.prod(core_attn_out.size()[:-2])] * self.num_attention_heads_per_partition
                core_attn_out_permute = core_attn_out.movedim(-2, 0).contiguous()
                core_attn_out, _ = self.linear_kv_up_proj_absorb_output(
                    core_attn_out_permute, m_splits_v
                )
                core_attn_out = core_attn_out.transpose(0, core_attn_out.ndim - 2)
                core_attn_out = core_attn_out.flatten(-2, -1).contiguous()

            elif self.cache_mla_latents:
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                )
                if not inference_context.is_decode_only():
                    core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")
            if self.offload_core_attention and self.training:
                (core_attn_out,) = fine_grained_offloading_group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )

        if self.cache_mla_latents and inference_context.is_decode_only():
            core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
            core_attn_out = core_attn_out.contiguous()
            core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

        if self.padding_v_head_dim:
            _prefix = core_attn_out.shape[:-1]
            core_attn_out = core_attn_out.reshape(*_prefix, -1, self.v_channels)
            core_attn_out = core_attn_out[..., : self.config.v_head_dim].reshape(*_prefix, -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        if self.offload_attn_proj:
            core_attn_out = fine_grained_offloading_group_start(core_attn_out, name="attn_proj")
        with get_fine_grained_offloading_context(self.offload_attn_proj):
            output, bias = self.linear_proj(core_attn_out)
        if self.offload_attn_proj:
            output, bias = fine_grained_offloading_group_commit(
                output, bias, name="attn_proj", forced_released_tensors=[core_attn_out]
            )

        return output, bias

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """Derive query/key/value tensors using Omni fused DSA absorb-q path."""
        if self.cache_mla_latents:
            return super().get_query_key_value_tensors(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                inference_context=inference_context,
                inference_params=inference_params,
            )

        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        inference_context = deprecate_inference_params(inference_context, inference_params)
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # Calculate position embedding offset for chunkpipe
        pos_emb_offset = 0
        if self.config.enable_chunkpipe:
            ck_fwd_mic = self.config.chunkpipe_forward_microbatch % self.num_chunks_per_seq
            if not self.config.chunkpipe_forward:
                ck_fwd_mic = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
            pos_emb_offset = ck_fwd_mic * self.config.chunksize

        mscale = 1.0
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(
                    rotary_seq_len, offset=pos_emb_offset, packed_seq=packed_seq)

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

        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if (
                parallel_state.get_tensor_model_parallel_world_size() > 1
                and self.config.sequence_parallel
            ):
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        if packed_seq_params is not None:
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        if self.config.q_lora_rank is not None:
            q_compressed = self.q_layernorm(q_compressed)
        kv_compressed = self.kv_layernorm(kv_compressed)

        def qkv_up_proj_and_rope_apply_for_dsa(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        ):
            assert self.linear_kv_up_proj_absorb_q is not None, (
                "get_query_kv_tensor() can only be called when linear_kv_up_proj_absorb_q is not "
                "None. This method is used for Omni fused DSA where kv_up weights are absorbed "
                "into query and output projections."
            )

            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                q, _ = self.linear_q_proj(q_compressed)
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            k_pos_emb_local = torch.unsqueeze(k_pos_emb, -2)

            if self.config.sequence_parallel and not self.config.enable_chunkpipe:
                kv_compressed = gather_from_sequence_parallel_region(kv_compressed)

            q_len = q.size()[0]

            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()

                q_no_pe, q_pos_emb_raw = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # Absorb key up projection weight into query
                q_no_pe_4d = q_no_pe.unsqueeze(1) if q_no_pe.ndim == 3 else q_no_pe
                q_content, _ = self.linear_kv_up_proj_absorb_q(
                    q_no_pe_4d.permute(2, 0, 1, 3).contiguous(),
                    [q_len * q_no_pe_4d.size(1)] * self.num_attention_heads_per_partition,
                )

                query = fused_rope_permute_cat(
                    q_content,
                    q_pos_emb_raw,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )

                kv_cached = fused_apply_mla_rope_for_absorb_kv(
                    kv_compressed.unsqueeze(1),
                    k_pos_emb_local,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_pos_emb_head_dim,
                    self.config.kv_lora_rank,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
                kv_cached = kv_cached.squeeze(1)
            else:
                if inference_context is not None:
                    sequence_start = inference_context.sequence_len_offset
                    sequence_end = sequence_start + q_len
                    rotary_pos_emb_local = rotary_pos_emb[sequence_start:sequence_end]
                elif packed_seq_params is None or self.config.context_parallel_size == 1:
                    rotary_pos_emb_local = rotary_pos_emb[0:q_len]
                else:
                    rotary_pos_emb_local = rotary_pos_emb

                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb_local,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )
                k_pos_emb_local = apply_rotary_pos_emb(
                    k_pos_emb_local,
                    rotary_pos_emb_local,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )

                # Chunkpipe: register gradient hooks and cache KV
                if self.config.enable_chunkpipe:
                    def kv_compressed_hook_fn(grad):
                        """Hook function to combine compressed KV gradients from subsequent chunk."""
                        chunks_in_current_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
                        if chunks_in_current_sequence == self.num_chunks_per_seq - 1:
                            return grad
                        else:
                            grad_from_prev_chunk = self.kv_compressed_cache_grad.pop(chunks_in_current_sequence)
                            return grad + grad_from_prev_chunk
                    
                    def key_pos_emb_hook_fn(grad):
                        """Hook function to combine key position embedding gradients from subsequent chunk."""
                        chunks_in_current_sequence = self.config.chunkpipe_backward_microbatch % self.num_chunks_per_seq
                        if chunks_in_current_sequence == self.num_chunks_per_seq - 1:
                            return grad
                        else:
                            grad_from_prev_chunk = self.key_pos_emb_cache_grad.pop(chunks_in_current_sequence)
                            return grad + grad_from_prev_chunk

                    if self.is_enable_grad_chunkpipe():
                        kv_compressed.register_hook(kv_compressed_hook_fn)
                        k_pos_emb_local.register_hook(key_pos_emb_hook_fn)
                    self.append_chunk_key_value_cache_mla(kv_compressed, k_pos_emb_local)

                    if self.config.sequence_parallel:
                        kv_compressed = gather_from_sequence_parallel_region(kv_compressed)

                kv_cached = torch.cat([kv_compressed, k_pos_emb_local.squeeze(1)], dim=-1)

                # Absorb key up projection weight into query
                q_no_pe_4d = q_no_pe.unsqueeze(1) if q_no_pe.ndim == 3 else q_no_pe
                q_content, _ = self.linear_kv_up_proj_absorb_q(
                    q_no_pe_4d.permute(2, 0, 1, 3).contiguous(),
                    [q_len * q_no_pe_4d.size(1)] * self.num_attention_heads_per_partition,
                )
                q_content = q_content.permute(1, 2, 0, 3).contiguous()
                q_content = q_content.squeeze(1) if packed_seq_params is not None else q_content

                query = torch.cat([q_content, q_pos_emb], dim=-1)

            key = kv_cached
            value = None
            return query.contiguous(), key.contiguous(), value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=self.config.fp8)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply_for_dsa,
                q_compressed,
                kv_compressed,
                k_pos_emb,
                rotary_pos_emb,
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply_for_dsa(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        return query, key, value, q_compressed, kv_compressed
