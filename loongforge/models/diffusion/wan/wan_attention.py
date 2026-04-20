# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""attention module"""

import torch

from copy import deepcopy
from megatron.core.transformer.attention import (
    CrossAttention,
    SelfAttention,
    Attention,
    SelfAttentionSubmodules,
    CrossAttentionSubmodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from .communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from megatron.core.parallel_state import get_context_parallel_group
from .wan_ulysses import DistributedAttention
import torch.nn as nn
import math

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


class WanSelfAttention(SelfAttention):
    """Self-attention layer class"""

    def __init__(
        self, config, submodules, ulysses_gather_idx=0, ulysses_scatter_idx=2, **kwargs
    ):
        _submodules = deepcopy(submodules)
        _submodules.core_attention = lambda: None
        super().__init__(config, _submodules, **kwargs)

        _config = deepcopy(config)
        _config.context_parallel_size = 1
        _config.num_attention_heads //= config.context_parallel_size
        _config.num_query_groups //= config.context_parallel_size

        _core_attention = build_module(
            submodules.core_attention,
            config=_config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=kwargs.get("cp_comm_type"),
            softmax_scale=self.config.softmax_scale,
        )

        self.core_attention = DistributedAttention(
            _core_attention,
            get_context_parallel_group(check_initialized=False),
            (
                self.config.recompute_num_layers
                if self.config.recompute_num_layers is not None
                else 0
            ),
            gather_idx=ulysses_gather_idx,
            scatter_idx=ulysses_scatter_idx,
        )

        ## Use custom RMSNorm to override original q_layernorm and k_layernorm due to precision issues
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None
        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        # q k layernorm, original shape qkv, qkv, qkv..., every qkv_last_size take  self.hidden_size_per_attention_head
        full_mixed_qkv = gather_forward_split_backward(
            mixed_qkv, get_context_parallel_group(), dim=0, grad_scale="up"
        )

        num_segments = self.config.num_attention_heads
        segment_len = full_mixed_qkv.shape[-1] // num_segments
        q_shape = list(full_mixed_qkv.shape[:-1]) + [segment_len // 3]
        all_q_shape = list(full_mixed_qkv.shape[:-1]) + [
            num_segments * (segment_len // 3)
        ]
        tensor = full_mixed_qkv.view(-1)
        chunk_len = q_shape[-1]
        segments = tensor.unfold(
            0, segment_len, segment_len
        )  # [num_segments, segment_len]
        q_all = segments[:, :chunk_len]
        k_all = segments[:, chunk_len : 2 * chunk_len]
        v_all = segments[:, 2 * chunk_len :]
        q_all_norm_in = q_all.reshape(*all_q_shape)
        k_all_norm_in = k_all.reshape(*all_q_shape)
        q_all = self.q_layernorm(q_all_norm_in).reshape(v_all.shape)
        k_all = self.k_layernorm(k_all_norm_in).reshape(v_all.shape)

        # Reorganize back to original shape
        merged = (
            torch.cat([q_all, k_all, v_all], dim=-1)
            .flatten()
            .view(full_mixed_qkv.shape)
        )
        mixed_qkv = split_forward_gather_backward(
            merged, get_context_parallel_group(), dim=0, grad_scale="down"
        )
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(
            query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
        )

        # if self.q_layernorm is not None:
        #     query = self.q_layernorm(query)

        # if self.k_layernorm is not None:
        #     key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()
        return query, key, value

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        """
        Perform a forward pass through the attention module.
        """
        # hidden_states: [sq, b, h]
        if self.config.flash_decode:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention
        # self or cross attn.

        query, key, value = self.get_query_key_value_tensors(
            hidden_states, key_value_states
        )
        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        query, key, value, rotary_pos_emb, attn_mask_type, _ = (
            self._adjust_key_value_for_inference(
                inference_params,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
        )

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
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

            assert self.apply_rotary_fn is not None, "apply_rotary_fn must be defined"
            query = self.apply_rotary_fn(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
            )
            key = self.apply_rotary_fn(
                key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
            )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = self.apply_rotary_fn(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
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
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        # =================
        # Output. [sq, b, h]
        # =================
        output = self.linear_proj(core_attn_out)
        return output


class WanCrossAttention(CrossAttention):
    """
    CrossAttention for wan
    """

    def __init__(
        self,
        config,
        submodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        **kwargs
    ):

        _submodules = deepcopy(submodules)
        _submodules.core_attention = lambda: None
        super().__init__(
            config,
            _submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            **kwargs
        )

        _config = deepcopy(config)
        _config.context_parallel_size = 1
        _config.num_attention_heads //= config.context_parallel_size
        _config.num_query_groups //= config.context_parallel_size

        _core_attention = build_module(
            submodules.core_attention,
            config=_config,
            layer_number=layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=kwargs.get("cp_comm_type"),
            softmax_scale=self.config.softmax_scale,
        )

        self.core_attention = DistributedAttention(
            _core_attention,
            get_context_parallel_group(check_initialized=False),
            (
                self.config.recompute_num_layers
                if self.config.recompute_num_layers is not None
                else 0
            ),
        )
        if self.config.has_image_input:
            _core_attention_img = build_module(
                submodules.core_attention,
                config=_config,
                layer_number=self.layer_number,
                attn_mask_type=self.attn_mask_type,
                attention_type=self.attention_type,
                cp_comm_type=kwargs.get("cp_comm_type"),
                softmax_scale=self.config.softmax_scale,
            )

            self.core_attention_img = DistributedAttention(
                _core_attention_img,
                get_context_parallel_group(check_initialized=False),
                (
                    self.config.recompute_num_layers
                    if self.config.recompute_num_layers is not None
                    else 0
                ),
                pad_kv=True,
                effective_length=self.config.max_image_length,
            )

            self.linear_kv_img = build_module(
                submodules.linear_kv,
                self.config.hidden_size,
                2 * self.kv_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
            )

        ## Override original q_layernorm and k_layernorm
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        if submodules.k_layernorm is not None and self.config.has_image_input:
            self.k_img_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_img_layernorm = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        attention_return_type=0,
        flash_attn_checkpoint=False,
        layer_index=-1,
        is_backward=False,
    ):
        """
        Perform a forward pass through the attention module.
        """
        # # For self attention we just duplicate the rotary_pos_emb if it isn't already
        # if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        #     rotary_pos_emb = (rotary_pos_emb,) * 2
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        if self.config.has_image_input:
            query, key, value, key_img, value_img = self.get_query_key_value_tensors(
                hidden_states, key_value_states
            )
        else:
            query, key, value = self.get_query_key_value_tensors(
                hidden_states, key_value_states
            )

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        query, key, value, rotary_pos_emb, attn_mask_type, _ = (
            self._adjust_key_value_for_inference(
                inference_params,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
        )

        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        if self.config.has_image_input:
            core_attention_img_out = self.core_attention_img(
                query,
                key_img,
                value_img,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            if attention_return_type == 1:
                return core_attn_out

            core_attn_out = core_attn_out + core_attention_img_out

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.linear_proj(core_attn_out)

        return output

    # Get query, key(context + image), value(context + image)
    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # img = y[:, :257],ctx = y[:, 257:]
        # ### S B C, S dimension split into 2 parts, one part is context, one part is image
        cated = gather_forward_split_backward(
            key_value_states, get_context_parallel_group(), dim=0, grad_scale="up"
        )

        # concat image, text
        cp = self.config.context_parallel_size
        text_len = self.config.max_text_length // cp

        if self.config.has_image_input:
            image_len = math.ceil(self.config.max_image_length / cp)
            chunk_size = image_len + text_len

            chunks_img, chunks_text = [], []
            for i in range(self.config.context_parallel_size):
                chunk = cated[i * chunk_size : (i + 1) * chunk_size]
                chunks_img.append(chunk[:image_len])
                chunks_text.append(chunk[image_len:])
            key_value_states_img = torch.cat(chunks_img, dim=0)
            key_value_states_ctx = torch.cat(chunks_text, dim=0)
        else:
            key_value_states_ctx = cated

        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states_ctx)
        # RMSNorm K in mixed_kv

        # q k layernorm, original shape qkv, qkv, qkv..., every qkv_last_size take  self.hidden_size_per_attention_head
        def norm_k(mixed_kv, norm_func, clip_padding=False):
            full_kv = mixed_kv
            num_segments = self.config.num_attention_heads
            segment_len = full_kv.shape[-1] // num_segments
            k_shape = list(full_kv.shape[:-1]) + [segment_len // 2]
            all_k_shape = list(full_kv.shape[:-1]) + [num_segments * (segment_len // 2)]
            tensor = full_kv.view(-1)
            chunk_len = k_shape[-1]
            segments = tensor.unfold(
                0, segment_len, segment_len
            )  # [num_segments, segment_len]
            k_all = segments[:, :chunk_len]
            v_all = segments[:, chunk_len : 2 * chunk_len]
            k_all_norm_in = k_all.reshape(*all_k_shape)
            if clip_padding:
                img_len = self.config.max_image_length
                cliped = k_all_norm_in[:img_len]
                k_all_norm_in = torch.cat(
                    (norm_func(cliped), k_all_norm_in[img_len:]), dim=0
                )
            else:
                k_all_norm_in = norm_func(k_all_norm_in)
            k_all = k_all_norm_in.reshape(v_all.shape)
            # Reorganize back to original shape
            merged = torch.cat([k_all, v_all], dim=-1).flatten().view(full_kv.shape)
            return merged

        mixed_kv = norm_k(mixed_kv, self.k_layernorm)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)
        query = gather_forward_split_backward(
            query, get_context_parallel_group(), dim=0, grad_scale="up"
        )

        query = self.q_layernorm(query)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        query = split_forward_gather_backward(
            query, get_context_parallel_group(), dim=0, grad_scale="down"
        )
        key = split_forward_gather_backward(
            key, get_context_parallel_group(), dim=0, grad_scale="down"
        )
        value = split_forward_gather_backward(
            value, get_context_parallel_group(), dim=0, grad_scale="down"
        )
        if self.config.has_image_input:
            mixed_kv_img, _ = self.linear_kv_img(key_value_states_img)

            mixed_kv_img = norm_k(mixed_kv_img, self.k_img_layernorm, clip_padding=True)

            # ###RMSNorm K in mixed_kv_img
            new_tensor_shape_img = mixed_kv_img.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )

            mixed_kv_img = mixed_kv_img.view(*new_tensor_shape_img)
            (key_img, value_img) = tensor_parallel.split_tensor_along_last_dim(
                mixed_kv_img, 2
            )
            key_img = split_forward_gather_backward(
                key_img, get_context_parallel_group(), dim=0, grad_scale="down"
            )
            value_img = split_forward_gather_backward(
                value_img, get_context_parallel_group(), dim=0, grad_scale="down"
            )
            return query, key, value, key_img, value_img

        return query, key, value
