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
import torch.nn as nn

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


class WanSelfAttention(SelfAttention):
    """Self-attention layer class

    Uses Megatron's native context parallelism via TEDotProductAttention.
    cp_comm_type is passed through to TEDotProductAttention which handles
    all CP communication internally (ring attention / all-to-all / hybrid).
    """

    def __init__(self, config, submodules, **kwargs):
        super().__init__(config, submodules, **kwargs)

        q_hidden_size = self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
        k_hidden_size = self.num_query_groups_per_partition * self.hidden_size_per_attention_head

        # Override q_layernorm and k_layernorm with custom ones if specified
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=q_hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None
        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=k_hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives query, key and value tensors from hidden_states.
        """
        # Attention heads [sq/cp, b, h] --> [sq/cp, b, ng * (np/ng + 2) * hn]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        sq, b, _ = mixed_qkv.shape
        ng = self.num_query_groups_per_partition
        np_pp = self.num_attention_heads_per_partition
        hn = self.hidden_size_per_attention_head
        q_per_g = np_pp // ng  # num Q heads per group

        # [sq, b, ng*(q_per_g+2)*hn] -> [sq, b, ng, q_per_g+2, hn]
        mixed_qkv = mixed_qkv.view(sq, b, ng, q_per_g + 2, hn)

        # Extract Q, K, V contiguously
        query = mixed_qkv[:, :, :, :q_per_g, :].contiguous()  # [sq, b, ng, q_per_g, hn]
        key   = mixed_qkv[:, :, :, q_per_g, :].contiguous()   # [sq, b, ng, hn]
        value = mixed_qkv[:, :, :, q_per_g + 1, :].contiguous()  # [sq, b, ng, hn]

        # Q/K RMSNorm
        query = query.view(sq, b, np_pp * hn)
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        key = key.view(sq, b, ng * hn)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        # Reshape to per-head layout: [sq, b, np, hn] and [sq, b, ng, hn]
        query = query.view(sq, b, np_pp, hn)
        key   = key.view(sq, b, ng, hn)

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
            rotary_pos_cos = None
            rotary_pos_sin = None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
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
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
            )
            key = self.apply_rotary_fn(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
            )

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
    CrossAttention for wan.

    Uses Megatron's native context parallelism via TEDotProductAttention.
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
        super().__init__(
            config,
            submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            **kwargs
        )

        # Override q_layernorm and k_layernorm
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
        Perform a forward pass through the cross-attention module.
        """
        query, key, value = self.get_query_key_value_tensors(
            hidden_states, key_value_states
        )

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

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.linear_proj(core_attn_out)
        return output

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives query tensor from hidden_states, and key/value tensors
        from key_value_states.
        """

        def norm_k(mixed_kv, norm_func):
            full_kv = mixed_kv
            num_segments = self.config.num_attention_heads
            segment_len = full_kv.shape[-1] // num_segments
            head_dim = segment_len // 2

            kv_view = full_kv.view(*full_kv.shape[:-1], num_segments, 2, head_dim)
            k_all = kv_view[..., 0, :]
            v_all = kv_view[..., 1, :]
            k_all_norm_in = k_all.reshape(*full_kv.shape[:-1], num_segments * head_dim)
            k_all = norm_func(k_all_norm_in).reshape_as(k_all)
            return torch.stack((k_all, v_all), dim=-2).reshape_as(full_kv)

        mixed_kv, _ = self.linear_kv(key_value_states)
        mixed_kv = norm_k(mixed_kv, self.k_layernorm)
        new_tensor_shape = mixed_kv.size()[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)
        key, value = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        query, _ = self.linear_q(hidden_states)
        query = self.q_layernorm(query)
        new_tensor_shape = query.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
