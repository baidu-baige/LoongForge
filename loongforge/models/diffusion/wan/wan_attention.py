# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""attention module"""

import torch
import torch.distributed as dist

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
from megatron.core.packed_seq_params import PackedSeqParams
import torch.nn as nn

# Sentinel to distinguish "attribute absent" from "attribute is None".
_MISSING = object()


try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


# [Packing][CP][Ulysses] All-to-all for THD format: scatter one tensor
# dimension and gather another while preserving autograd through the inverse op.
def _thd_all_to_all(input_: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _THDSeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim, group):
        """Run THD sequence all-to-all in the forward pass."""
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _thd_all_to_all(input_, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Run inverse THD sequence all-to-all for gradient propagation."""
        return _thd_all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim, ctx.group), None, None, None


def _thd_compact(x, cu_actual, cu_padded):
    """Remove inter-sample padding from a THD tensor, returning compact tensor and indices."""
    num_sequences = cu_actual.shape[0] - 1
    indices = []
    for sequence_index in range(num_sequences):
        start_padded = cu_padded[sequence_index].item()
        actual_len = cu_actual[sequence_index + 1].item() - cu_actual[sequence_index].item()
        for token_offset in range(actual_len):
            indices.append(start_padded + token_offset)
    indices_t = torch.tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(0, indices_t), indices_t


def _thd_expand(x_compact, indices, total_padded_len, extra_dims):
    """Scatter compact attention output back to padded positions."""
    out = torch.zeros(
        total_padded_len, *extra_dims, dtype=x_compact.dtype, device=x_compact.device
    )
    out.index_copy_(0, indices, x_compact)
    return out


class WanSelfAttention(SelfAttention):
    """Self-attention layer class

    Uses Megatron's native context parallelism via TEDotProductAttention.
    cp_comm_type is passed through to TEDotProductAttention which handles
    all CP communication internally (ring attention / all-to-all / hybrid).
    """

    def __init__(self, config, submodules, **kwargs):
        super().__init__(config, submodules, **kwargs)

        # Override q_layernorm and k_layernorm with custom ones if specified
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
        # Squeeze batch dim for THD packed format (TE requires 3D: [S, H, D])
        thd_mode = (packed_seq_params is not None and
                    getattr(packed_seq_params, 'qkv_format', None) == 'thd')
        if thd_mode and query.dim() == 4:
            query = query.squeeze(1)   # [sq, b, np, hn] -> [sq, np, hn]
            key = key.squeeze(1)
            value = value.squeeze(1)

        saved_cp_group = None
        saved_cp_global_ranks = None
        saved_cp_comm_type = None
        saved_num_gqa_groups = None
        saved_num_attn_heads = None
        ulysses_group = None
        compact_indices = None
        total_padded_len = None
        if thd_mode and hasattr(self.core_attention, 'cp_group') and self.core_attention.cp_group is not None:
            cp_group = self.core_attention.cp_group
            is_hierarchical = isinstance(cp_group, list)
            is_ulysses = getattr(self.core_attention, 'cp_comm_type', 'p2p') in ('a2a', 'all_to_all')
            if is_hierarchical:
                # [Packing][CP][Ulysses] Do Ulysses a2a outside TE, keep TE ring on the ring subgroup.
                saved_cp_group = cp_group
                ulysses_group = cp_group[0]
                ring_group = cp_group[1]
                self.core_attention.cp_group = ring_group
                saved_cp_global_ranks = getattr(self.core_attention, 'cp_global_ranks', None)
                if saved_cp_global_ranks is not None:
                    self.core_attention.cp_global_ranks = dist.get_process_group_ranks(ring_group)
                saved_cp_comm_type = getattr(self.core_attention, 'cp_comm_type', None)
                self.core_attention.cp_comm_type = "p2p"
            elif is_ulysses:
                # [Packing][CP][Ulysses] Pure Ulysses uses external a2a and no TE ring.
                saved_cp_group = cp_group
                ulysses_group = cp_group
                self.core_attention.cp_group = None

        attn_packed_seq_params = packed_seq_params
        if thd_mode:
            cu_q = packed_seq_params.cu_seqlens_q
            cu_q_padded = packed_seq_params.cu_seqlens_q_padded
            has_padding = (cu_q_padded is not None
                           and not torch.equal(cu_q_padded[:-1], cu_q[:-1]))
            if has_padding:
                total_padded_len = query.shape[0]
                query, compact_indices = _thd_compact(query, cu_q, cu_q_padded)
                key, _ = _thd_compact(key, cu_q, cu_q_padded)
                value, _ = _thd_compact(value, cu_q, cu_q_padded)
                compact_cu = cu_q - cu_q[0]
                max_seqlen = (cu_q[1:] - cu_q[:-1]).max().item()
                attn_packed_seq_params = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=compact_cu,
                    cu_seqlens_kv=compact_cu,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                )

        if ulysses_group is not None:
            # [Packing][CP][Ulysses] THD layout is [S, H, D]: scatter heads, gather sequence.
            query = _THDSeqAllToAll.apply(query, 1, 0, ulysses_group)
            key = _THDSeqAllToAll.apply(key, 1, 0, ulysses_group)
            value = _THDSeqAllToAll.apply(value, 1, 0, ulysses_group)
            ulysses_degree = dist.get_world_size(ulysses_group)
            saved_num_gqa_groups = getattr(self.core_attention, 'num_gqa_groups_per_partition', None)
            saved_num_attn_heads = getattr(self.core_attention, 'num_attention_heads', None)
            if saved_num_gqa_groups is not None:
                self.core_attention.num_gqa_groups_per_partition = saved_num_gqa_groups // ulysses_degree
            if saved_num_attn_heads is not None:
                self.core_attention.num_attention_heads = saved_num_attn_heads // ulysses_degree

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=attn_packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=attn_packed_seq_params,
            )

        if ulysses_group is not None:
            # [Packing][CP][Ulysses] Invert pre-attention a2a: scatter sequence, gather heads.
            core_attn_out = _THDSeqAllToAll.apply(core_attn_out, 0, 1, ulysses_group)

        if compact_indices is not None:
            core_attn_out = _thd_expand(
                core_attn_out, compact_indices, total_padded_len, core_attn_out.shape[1:]
            )

        if saved_cp_group is not None:
            self.core_attention.cp_group = saved_cp_group
        if saved_cp_global_ranks is not None:
            self.core_attention.cp_global_ranks = saved_cp_global_ranks
        if saved_cp_comm_type is not None:
            self.core_attention.cp_comm_type = saved_cp_comm_type
        if saved_num_gqa_groups is not None:
            self.core_attention.num_gqa_groups_per_partition = saved_num_gqa_groups
        if saved_num_attn_heads is not None:
            self.core_attention.num_attention_heads = saved_num_attn_heads

        # Unsqueeze batch dim back for THD mode
        if thd_mode and core_attn_out.dim() == 2:
            core_attn_out = core_attn_out.unsqueeze(1)  # [sq, h] -> [sq, b, h]

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

        self.has_image_input = getattr(self.config, "has_image_input", False)
        # Read CLIP image-token count from config so it can vary per model.
        self.clip_num_image_tokens = getattr(self.config, "clip_num_image_tokens", 257)
        if self.has_image_input:
            self.linear_k_img = build_module(
                submodules.linear_k_img,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="k_img",
                tp_group=self.pg_collection.tp,
            )
            self.linear_v_img = build_module(
                submodules.linear_v_img,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="v_img",
                tp_group=self.pg_collection.tp,
            )
            if submodules.k_img_layernorm is not None:
                self.k_img_layernorm = build_module(
                    submodules.k_img_layernorm,
                    hidden_size=self.config.hidden_size,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.k_img_layernorm = None
        else:
            self.linear_k_img = None
            self.linear_v_img = None
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
        Perform a forward pass through the cross-attention module.
        """
        img_states = None
        if self.has_image_input:
            n_img = self.clip_num_image_tokens
            if key_value_states is None or key_value_states.shape[0] < n_img:
                raise ValueError(
                    f"Wan2.1 I2V cross-attention requires {n_img} "
                    f"CLIP image tokens before text tokens."
                )
            img_states = key_value_states[:n_img]
            key_value_states = key_value_states[n_img:]

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

        # Squeeze batch dim for THD packed format (TE requires 3D: [S, H, D])
        thd_mode = (packed_seq_params is not None and
                    getattr(packed_seq_params, 'qkv_format', None) == 'thd')
        if thd_mode and query.dim() == 4:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # Snapshot variables. Initialized BEFORE the try block so finally can
        # always reference them, even if an exception is raised between try
        # entry and the first mutation site.
        saved_cp_group = None
        saved_cp_global_ranks = None
        saved_cp_comm_type = None
        saved_num_gqa_groups = None
        saved_num_attn_heads = None
        ulysses_group = None
        compact_indices_q = None
        total_padded_len_q = None
        attn_packed_seq_params = packed_seq_params
        # _MISSING means "the attribute did not exist"; do not write it back.
        saved_attrs = {}

        # All mutation / potentially-throwing setup is wrapped in try so finally
        # always restores core_attention state. This covers:
        #   - cp_group / cp_comm_type / cp_global_ranks rewrites
        #   - num_gqa_groups_per_partition / num_attention_heads rewrites
        #   - disable_core_cp branch attribute rewrites
        #   - _thd_compact, _THDSeqAllToAll, NCCL ops which may raise
        #   - the explicit NotImplementedError guard for THD+Ulysses+img_states
        try:
            if thd_mode and hasattr(self.core_attention, 'cp_group') and self.core_attention.cp_group is not None:
                cp_group = self.core_attention.cp_group
                is_hierarchical = isinstance(cp_group, list)
                is_ulysses = getattr(self.core_attention, 'cp_comm_type', 'p2p') in ('a2a', 'all_to_all')
                if is_hierarchical:
                    # [Packing][CP][Ulysses] Do Ulysses a2a outside TE, keep TE ring on the ring subgroup.
                    saved_cp_group = cp_group
                    ulysses_group = cp_group[0]
                    ring_group = cp_group[1]
                    self.core_attention.cp_group = ring_group
                    saved_cp_global_ranks = getattr(self.core_attention, 'cp_global_ranks', None)
                    if saved_cp_global_ranks is not None:
                        self.core_attention.cp_global_ranks = dist.get_process_group_ranks(ring_group)
                    saved_cp_comm_type = getattr(self.core_attention, 'cp_comm_type', None)
                    self.core_attention.cp_comm_type = "p2p"
                elif is_ulysses:
                    # [Packing][CP][Ulysses] Pure Ulysses uses external a2a and no TE ring.
                    saved_cp_group = cp_group
                    ulysses_group = cp_group
                    self.core_attention.cp_group = None

            if thd_mode:
                cu_q = packed_seq_params.cu_seqlens_q
                cu_q_padded = packed_seq_params.cu_seqlens_q_padded
                cu_kv = packed_seq_params.cu_seqlens_kv
                cu_kv_padded = packed_seq_params.cu_seqlens_kv_padded
                has_q_padding = (cu_q_padded is not None
                                 and not torch.equal(cu_q_padded[:-1], cu_q[:-1]))
                has_kv_padding = (cu_kv_padded is not None
                                  and not torch.equal(cu_kv_padded[:-1], cu_kv[:-1]))
                if has_q_padding or has_kv_padding:
                    total_padded_len_q = query.shape[0]
                    if has_q_padding:
                        query, compact_indices_q = _thd_compact(query, cu_q, cu_q_padded)
                    if has_kv_padding:
                        key, _ = _thd_compact(key, cu_kv, cu_kv_padded)
                        value, _ = _thd_compact(value, cu_kv, cu_kv_padded)
                    compact_cu_q = cu_q - cu_q[0] if has_q_padding else cu_q
                    compact_cu_kv = cu_kv - cu_kv[0] if has_kv_padding else cu_kv
                    max_seqlen_q = (cu_q[1:] - cu_q[:-1]).max().item()
                    max_seqlen_kv = (cu_kv[1:] - cu_kv[:-1]).max().item()
                    attn_packed_seq_params = PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=compact_cu_q,
                        cu_seqlens_kv=compact_cu_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                    )

            if ulysses_group is not None:
                # [Packing][CP][Ulysses] THD layout is [S, H, D]: scatter heads, gather sequence.
                query = _THDSeqAllToAll.apply(query, 1, 0, ulysses_group)
                key = _THDSeqAllToAll.apply(key, 1, 0, ulysses_group)
                value = _THDSeqAllToAll.apply(value, 1, 0, ulysses_group)
                ulysses_degree = dist.get_world_size(ulysses_group)
                saved_num_gqa_groups = getattr(self.core_attention, 'num_gqa_groups_per_partition', None)
                saved_num_attn_heads = getattr(self.core_attention, 'num_attention_heads', None)
                if saved_num_gqa_groups is not None:
                    self.core_attention.num_gqa_groups_per_partition = saved_num_gqa_groups // ulysses_degree
                if saved_num_attn_heads is not None:
                    self.core_attention.num_attention_heads = saved_num_attn_heads // ulysses_degree

            disable_core_cp = (
                self.has_image_input
                and not thd_mode
                and hasattr(self.core_attention, 'cp_group')
                and self.core_attention.cp_group is not None
            )
            if disable_core_cp:
                saved_attrs['cp_group'] = self.core_attention.cp_group
                saved_attrs['cp_global_ranks'] = getattr(self.core_attention, 'cp_global_ranks', _MISSING)
                saved_attrs['cp_comm_type'] = getattr(self.core_attention, 'cp_comm_type', _MISSING)
                self.core_attention.cp_group = None
                if hasattr(self.core_attention, 'cp_global_ranks'):
                    self.core_attention.cp_global_ranks = None
                if hasattr(self.core_attention, 'cp_comm_type'):
                    self.core_attention.cp_comm_type = None

            # THD packing + Ulysses CP scatters query along the head dim before this
            # point, but image-attention K/V are not a2a-scattered. Mixing them would
            # give wrong results silently (or a shape-mismatch crash). Forbid the
            # combination until explicitly supported. Inside try so finally still
            # restores any state mutated above.
            if thd_mode and ulysses_group is not None and img_states is not None:
                raise NotImplementedError(
                    "Wan I2V image cross-attention is not supported under THD packing + Ulysses CP yet."
                )

            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=attn_packed_seq_params,
            )
            if img_states is not None:
                img_key, img_value = self.get_image_key_value_tensors(img_states)
                if thd_mode and img_key.dim() == 4:
                    img_key = img_key.squeeze(1)
                    img_value = img_value.squeeze(1)
                image_attn_out = self.core_attention(
                    query,
                    img_key,
                    img_value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=None,
                )
                core_attn_out = core_attn_out + image_attn_out
        finally:
            # Always restore mutated state, even if core_attention raised.
            if saved_cp_group is not None:
                self.core_attention.cp_group = saved_cp_group
            if saved_cp_global_ranks is not None:
                self.core_attention.cp_global_ranks = saved_cp_global_ranks
            if saved_cp_comm_type is not None:
                self.core_attention.cp_comm_type = saved_cp_comm_type
            if saved_num_gqa_groups is not None:
                self.core_attention.num_gqa_groups_per_partition = saved_num_gqa_groups
            if saved_num_attn_heads is not None:
                self.core_attention.num_attention_heads = saved_num_attn_heads
            for k, v in saved_attrs.items():
                if v is not _MISSING:
                    setattr(self.core_attention, k, v)

        if ulysses_group is not None:
            # [Packing][CP][Ulysses] Invert pre-attention a2a: scatter sequence, gather heads.
            core_attn_out = _THDSeqAllToAll.apply(core_attn_out, 0, 1, ulysses_group)

        if compact_indices_q is not None:
            core_attn_out = _thd_expand(
                core_attn_out, compact_indices_q, total_padded_len_q, core_attn_out.shape[1:]
            )

        # Unsqueeze batch dim back for THD mode
        if thd_mode and core_attn_out.dim() == 2:
            core_attn_out = core_attn_out.unsqueeze(1)

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

        target_dtype = next(self.linear_kv.parameters()).dtype
        key_value_states = key_value_states.to(dtype=target_dtype)
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

    def get_image_key_value_tensors(self, image_states):
        """Derives image key/value tensors from CLIP image states."""
        target_dtype = next(self.linear_k_img.parameters()).dtype
        image_states = image_states.to(dtype=target_dtype)
        key, _ = self.linear_k_img(image_states)
        if self.k_img_layernorm is not None:
            key = self.k_img_layernorm(key)
        value, _ = self.linear_v_img(image_states)

        new_tensor_shape = key.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        key = key.view(*new_tensor_shape)
        value = value.view(*new_tensor_shape)
        return key, value
