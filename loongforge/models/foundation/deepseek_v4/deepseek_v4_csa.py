# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compressed sparse attention modules for DeepSeek-v4 hybrid attention."""
import os
import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
except Exception:
    fused_mla_rope_inplace = None
from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import (
    all_to_all,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    rotate_activation,
)

from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive_thd
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    batch_of_row,
    build_flat_topk_idxs,
    dsa_sparse_attn,
    fused_indexer_sparse_attn,
    indexer_topk,
)

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide, get_pg_size, nvtx_range_pop, nvtx_range_push

import deep_gemm
import flashinfer
import lightning_indexer_bwd
import transformer_engine_torch as tex


from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

def _all_to_all_hp2sp(input_: torch.Tensor, tp_group) -> torch.Tensor:
    """All-to-All: head-parallel to seq-parallel.

    [S, B, H/TP, D] -> [S_padded/TP, B, H, D]

    Scatters heads and gathers sequence across TP ranks.
    """
    world_size = tp_group.size()
    if world_size == 1:
        return input_

    s, *bhd = input_.shape

    # Pad sequence to be divisible by world_size
    remainder = s % world_size
    padding_size = (world_size - remainder) % world_size
    if padding_size > 0:
        out = input_.new_zeros(s + padding_size, *input_.shape[1:])
        out[:s] = input_
        input_ = out

    s_padded = input_.size(0)
    input_ = input_.reshape(world_size, s_padded // world_size, *bhd)

    input_exchanged = all_to_all(tp_group, input_)  # [TP, s/TP, *bhd]
    output = input_exchanged.movedim(0, -3)  # [s/TP, (b,) TP, h/TP, d]
    output = output.flatten(-3, -2).contiguous()  # [s/TP, (b,) H, d]
    return output

def _all_to_all_sp2hp(input_: torch.Tensor, tp_group, ori_s: int = None) -> torch.Tensor:
    """All-to-All: seq-parallel to head-parallel.

    [S/TP, ..., H, D] -> [S, ..., H/TP, D]

    Gathers sequence and scatters heads across TP ranks. Supports both SBHD
    ``[S/TP, B, H, D]`` and THD ``[S/TP, H, D]`` layouts.
    """
    world_size = tp_group.size()
    if world_size == 1:
        return input_

    s_local, *middle, h, d = input_.shape
    h_local = h // world_size

    input_ = input_.reshape(s_local, *middle, world_size, h_local, d)
    input_ = input_.movedim(input_.ndim - 3, 0).contiguous()

    input_exchanged = all_to_all(tp_group, input_)
    output = input_exchanged.reshape(world_size * s_local, *middle, h_local, d)

    # Remove padding if original sequence length is provided
    if ori_s is not None:
        output = output[:ori_s].contiguous()

    return output

class _AllToAllHp2Sp(torch.autograd.Function):
    """Autograd-aware All-to-All: head-parallel -> seq-parallel."""

    @staticmethod
    def forward(ctx, input_, tp_group):
        """Forward function."""
        ctx.tp_group = tp_group
        ctx.orig_s = input_.size(0)
        return _all_to_all_hp2sp(input_, tp_group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _all_to_all_sp2hp(grad_output, ctx.tp_group, ctx.orig_s), None


class _AllToAllSp2Hp(torch.autograd.Function):
    """Autograd-aware All-to-All: seq-parallel -> head-parallel."""

    @staticmethod
    def forward(ctx, input_, tp_group):
        """Forward function."""
        ctx.tp_group = tp_group
        ctx.orig_s = None
        if input_.ndim == 4:
            input_ = input_.transpose(0, 1).contiguous()
            output = _all_to_all_sp2hp(input_, tp_group)
            output = output.transpose(0, 1).contiguous()
            return output
        return _all_to_all_sp2hp(input_, tp_group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if grad_output.ndim == 4:
            grad_output = grad_output.transpose(0, 1).contiguous()
            output = _all_to_all_hp2sp(grad_output, ctx.tp_group)
            output = output.transpose(0, 1).contiguous()
            return output, None
        return _all_to_all_hp2sp(grad_output, ctx.tp_group), None

# ---------------------------------------------------------------------------
# Helper functions for index computation
# ---------------------------------------------------------------------------


# TODO: the lru_cache may not work well with packed sequence
@lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(window_size: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Compute sliding-window indices for a single sequence (cached).

    Returns:
        indices: [seqlen, window_size] int tensor, -1 for invalid positions.
    """
    base = torch.arange(seqlen, device=device_str).unsqueeze(1)
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices [batch, seqlen, window_size]."""
    matrix = _get_window_topk_idxs_cached(window_size, seqlen, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# TODO: the lru_cache may not work well with packed sequence
@lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> torch.Tensor:
    """Compute all-compressed-positions indices for a single sequence (cached).

    Returns:
        indices: [seqlen, seqlen // ratio] int tensor, -1 for future positions.
    """
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device_str).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix


def get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All-compressed-position indices [batch, seqlen, seqlen // ratio]."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------


def _apply_fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens: Optional[torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Apply the fused MLA RoPE kernel with automatic 3-D / 4-D handling."""
    packed_seq = cu_seqlens is not None

    # Strip the dummy batch axis for packed sequences: (total, 1, h, d) -> (total, h, d)
    squeezed_b = packed_seq and x.dim() == 4 and x.size(1) == 1
    if squeezed_b:
        x = x.squeeze(1)

    # Add a dummy head axis for non-packed sequences: (b, s, d) -> (b, s, 1, d)
    squeeze_head = not packed_seq and x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)

    out = fused_mla_rope_inplace(
        x,
        cos,
        sin,
        nope_dim,
        pos_dim,
        cu_seqlens,
        cp_group.rank(),
        cp_group.size(),
        remove_interleaving=True,
    )

    if squeezed_b:
        out = out.unsqueeze(1)
    if squeeze_head:
        out = out.squeeze(-2)
    return out


def _apply_unfused_rope(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    config: TransformerConfig,
    cu_seqlens: Optional[torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Apply unfused RoPE (split, rotate, concat) with 3-D / 4-D handling.

    DSv4 forces ``mscale=1.0`` -- the model relies on Q/KV RMS-norm +
    unit-magnitude rotation, not Yarn's concentration factor.
    """
    packed_seq = cu_seqlens is not None

    # Drop dummy ``b=1`` from packed 4-D ``(total, 1, h, d)`` callers.
    squeezed_b = packed_seq and x.dim() == 4 and x.size(1) == 1
    # Packed 3-D ``(total, 1, d)``: collapse batch and add a temporary head dim.
    squeezed_b_3d = packed_seq and x.dim() == 3 and x.size(1) == 1
    if squeezed_b:
        x = x.squeeze(1)
    elif squeezed_b_3d:
        x = x.squeeze(1).unsqueeze(-2)

    # Non-packed 3-D ``(b, s, d)``: add a temporary head dim.
    squeeze_head = not packed_seq and x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)

    x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
    x_pe = apply_rotary_pos_emb(
        x_pe,
        rotary_pos_emb,
        config=config,
        cu_seqlens=cu_seqlens,
        mscale=1.0,
        cp_group=cp_group,
        mla_rotary_interleaved=True,
        mla_output_remove_interleaving=True,
        max_seqlen=max_seqlen,
    )
    out = torch.cat([x_nope, x_pe], dim=-1)

    if squeezed_b:
        out = out.unsqueeze(1)
    elif squeezed_b_3d:
        out = out.squeeze(-2).unsqueeze(1)
    elif squeeze_head:
        out = out.squeeze(-2)
    return out


def _apply_rope(
    x: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module: RotaryEmbedding,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    cp_group: torch.distributed.ProcessGroup = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen_rope: Optional[int] = None,
) -> torch.Tensor:
    """Apply RoPE to the last ``pos_dim`` dims, leaving the rest unchanged.

    Two layouts:

    * **SBHD** (``cu_seqlens=None``): builds a single rotary table of length
      ``rotary_seq_len * ratio`` and slices with stride ``ratio``.
    * **THD packed** (``cu_seqlens`` supplied): globally strided tables
      (``table[:max_total:ratio]``), matching the SBHD approach.

    Args:
        max_seqlen_rope: pre-computed ``max(seg_lens) * ratio`` for the
            THD + ``ratio > 1`` path (avoids a GPU->CPU sync when the
            caller already knows the max original sequence length).
    """
    packed_seq = cu_seqlens is not None

    if packed_seq:
        if max_seqlen_rope is None:
            raise ValueError(
                "_apply_rope: max_seqlen_rope is required for THD packed sequences "
                "to avoid a GPU->CPU sync that breaks CUDA graph capture."
            )
        max_total = max_seqlen_rope
    else:
        max_total = None

    use_fused = config.apply_rope_fusion

    if use_fused:
        if packed_seq:
            cos, sin = rotary_pos_emb_module.get_cached_cos_sin(
                max_total, dtype=x.dtype, packed_seq=True
            )
            if ratio > 1:
                cos = cos[:max_total:ratio]
                sin = sin[:max_total:ratio]
        else:
            total = rotary_seq_len * ratio if ratio > 1 else rotary_seq_len
            cos, sin = rotary_pos_emb_module.get_cached_cos_sin(
                total, dtype=x.dtype, packed_seq=False
            )
            if ratio > 1:
                cos = cos[:total:ratio][:rotary_seq_len]
                sin = sin[:total:ratio][:rotary_seq_len]
        return _apply_fused_rope(x, cos, sin, nope_dim, pos_dim, cu_seqlens, cp_group)

    # ---- Unfused path: build rotary_pos_emb tensor ----------------------
    if packed_seq:
        rope_result = rotary_pos_emb_module(max_total, packed_seq=True)
        rotary_pos_emb = rope_result[0] if isinstance(rope_result, tuple) else rope_result
        if ratio > 1:
            rotary_pos_emb = rotary_pos_emb[:max_total:ratio]
    else:
        total = rotary_seq_len * ratio if ratio > 1 else rotary_seq_len
        rope_result = rotary_pos_emb_module(total, packed_seq=False)
        rotary_pos_emb = rope_result[0] if isinstance(rope_result, tuple) else rope_result
        if ratio > 1:
            rotary_pos_emb = rotary_pos_emb[:total:ratio][:rotary_seq_len]

    max_seqlen = rotary_pos_emb.shape[0] if packed_seq else None
    return _apply_unfused_rope(
        x, rotary_pos_emb, nope_dim, pos_dim, config, cu_seqlens, cp_group, max_seqlen=max_seqlen
    )



# ---------------------------------------------------------------------------
# Sparse attention kernel (unfused, differentiable)
# ---------------------------------------------------------------------------


def unfused_compressed_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable sparse attention with MQA + learnable attention sink.
    Note: the unfused function is mainly for reference, and the performance
    and the memory footprint of it is not good for the real scenario.

    Layout is detected from ``query.ndim``:

    * **SBHD** (4-D query):
        query        ``(sq, b, np, hn)``    multi-head Q.
        kv_full      ``(n_kv, b, hn)``      single-head MQA KV (original
                                            + compressed concatenated).
        topk_indices ``(b, sq, topk)``      int32 **LOCAL per-batch** ids
                                            (``-1`` invalid).
        Returns ``(sq, b, np * hn)``.

    * **THD** (3-D query — callers should pre-``squeeze(1)`` the dummy b=1 dim):
        query        ``(total_q, np, hn)``  packed multi-head Q.
        kv_full      ``(total_kv, hn)``     packed single-head MQA KV.
        topk_indices ``(total_q, topk)``    int32 **flat-global** ids into
                                            ``kv_full`` (``-1`` invalid).
        Returns ``(total_q, np * hn)``.

    The math (gather -> MQA scores -> softmax with sink -> weighted sum) is
    identical for both layouts; SBHD adds permute / globalize-indices /
    unpermute around the call.

    Args:
        attn_sink: ``(np,)`` per-head learnable bias for the sink term.
        softmax_scale: scalar applied to ``Q . K^T`` before softmax.
    """
    is_thd = query.ndim == 3

    # ----------- Layout-specific input prep -------------------------------
    if is_thd:
        q_flat = query  # (rows, np, hn)
        kv_flat = kv_full  # (n_kv, hn)
        global_indices = topk_indices  # (rows, topk)
    else:
        sq, b, np_, hn = query.size()
        n_kv = kv_full.size(0)
        # b-major flatten of query and kv_full.
        q_flat = query.permute(1, 0, 2, 3).reshape(b * sq, np_, hn)
        kv_flat = kv_full.permute(1, 0, 2).reshape(b * n_kv, hn)
        # Globalize topk_indices: ``global = batch_idx * n_kv + local``.
        valid = topk_indices >= 0
        batch_ids = torch.arange(b, device=query.device).view(b, 1, 1)
        global_indices = torch.where(valid, topk_indices + batch_ids * n_kv, topk_indices).reshape(
            b * sq, -1
        )

    # ----------- Shared core: gather, MQA softmax with sink, sum ---------
    rows, np_, hn = q_flat.shape

    safe_indices = global_indices.clamp(min=0).long()
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(-1, -1, hn)
    kv_gathered = torch.gather(
        kv_flat.unsqueeze(0).expand(rows, -1, -1), dim=1, index=safe_indices_exp
    )  # (rows, topk, hn)

    q_f = q_flat.float()
    kv_g = kv_gathered.float()
    scores = torch.einsum("inh,ikh->ink", q_f, kv_g) * softmax_scale  # (rows, np, topk)

    invalid_mask = (global_indices < 0).unsqueeze(1)  # (rows, 1, topk)
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    sink = attn_sink.view(1, np_, 1).float()
    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_max = torch.max(scores_max, sink)

    exp_scores = torch.exp(scores - scores_max)
    exp_sink = torch.exp(sink - scores_max)
    attn_weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sink)

    output = torch.einsum("ink,ikh->inh", attn_weights, kv_g)
    output = output.to(query.dtype)

    # ----------- Layout-specific output reshape ---------------------------
    if is_thd:
        return output.reshape(rows, np_ * hn)
    return output.reshape(b, sq, np_ * hn).permute(1, 0, 2).contiguous()


# ============================================================
# THD (packed) helper functions from 0629 Megatron-LM PR#5011
# ============================================================

def _get_csa_compressed_capacity(
    packed_seq_params: Optional[PackedSeqParams], ratio: int, total_tokens: int
) -> Optional[int]:
    """Return a host-known compressed capacity for THD CUDA graph capture.

    The exact ``sum(seq_len // ratio)`` lives in ``cu_seqlens`` on device.
    Reading it on the host would break CUDA graph capture, and
    ``PackedSeqParams`` must stay a generic MCore contract rather than
    carrying CSA-specific metadata.  Use a static upper bound instead;
    device-side ``cu_seqlens_compressed`` keeps the true valid rows and
    downstream kernels leave the extra rows as tail padding.
    """
    if packed_seq_params is None or ratio <= 1:
        return None
    max_seqlen = packed_seq_params.max_seqlen_q
    cu_seqlens = (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )
    if max_seqlen is None or cu_seqlens is None:
        return None
    num_sequences = max(int(cu_seqlens.shape[0]) - 1, 0)
    return min(int(total_tokens) // ratio, num_sequences * (int(max_seqlen) // ratio))


# ---------------------------------------------------------------------------
# THD (packed) variants of the index helpers above.
#
# Both produce per-row local-to-segment indices in the SAME index space that
# ``dsa_kernels.local_to_global_flat(..., cu_seqlens_q=..., cu_seqlens_kv=...)``
# expects: each row is one query token in the packed layout, each value is
# either ``-1`` (invalid / future position) or a non-negative local KV id in
# ``[0, seqlen_kv_full[batch_of_row])`` where ``seqlen_kv_full[b] =
# seqlen_kv[b] + seqlen_compressed[b]``. Window indices live in
# ``[0, seqlen_kv[b])``; compressed indices live in
# ``[seqlen_kv[b], seqlen_kv[b] + seqlen_compressed[b])``.
#
# These mirror the SBHD helpers above but cannot be lru-cached because
# their output shape depends on the per-batch ``cu_seqlens`` tensors.
# ---------------------------------------------------------------------------


def get_window_topk_idxs_thd(
    window_size: int, cu_seqlens_q: torch.Tensor, total_q: Optional[int] = None
) -> torch.Tensor:
    """Sliding-window indices for a packed THD layout.

    For each query token ``i`` in segment ``b`` (with ``pos_in_seq =
    i - cu_seqlens_q[b]``), the window covers the last ``window_size``
    KV positions within the same segment's original KV region:
    indices ``[max(0, pos-window_size+1), ..., pos]``; positions
    extending before the start of the segment are emitted as ``-1``.

    Args:
        window_size: number of positions per window.
        cu_seqlens_q: ``(B+1,)`` int32 cumulative Q lengths
            (self-attention: same as KV lengths).
        total_q: total number of query tokens (avoids a GPU→CPU sync
            when the caller already knows it, e.g. from ``x.shape[0]``).

    Returns:
        ``(total_q, window_size)`` int32 — LOCAL (per-segment) KV indices.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    device = cu_seqlens_q.device
    batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
    token_idx = torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
    valid = token_idx < cu_seqlens_q[-1]
    pos_in_seq = token_idx - cu_seqlens_q[batch_of_token]
    pos_in_seq = torch.where(valid, pos_in_seq, torch.zeros_like(pos_in_seq))

    offsets = torch.arange(window_size, device=device, dtype=cu_seqlens_q.dtype)
    matrix = (pos_in_seq - window_size + 1).clamp(min=0).unsqueeze(1) + offsets.unsqueeze(0)
    matrix = torch.where(matrix > pos_in_seq.unsqueeze(1), torch.full_like(matrix, -1), matrix)
    matrix = torch.where(valid.unsqueeze(1), matrix, torch.full_like(matrix, -1))
    return matrix.int()


def get_compress_topk_idxs_thd(
    ratio: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    total_q: Optional[int] = None,
    max_n_compressed: Optional[int] = None,
) -> torch.Tensor:
    """All compressed-position indices for a packed THD layout.

    For each query token ``i`` in segment ``b`` (``pos_in_seq = i -
    cu_seqlens_q[b]``), the valid compressed positions within that
    segment are ``[0, 1, ..., (pos+1) // ratio - 1]`` (clamped to
    ``seqlen_compressed[b]``). The returned indices are already shifted
    by the per-segment offset ``seqlen_kv[b]`` so that they live in the
    *full* per-segment KV index space ``[seqlen_kv[b], seqlen_kv[b] +
    seqlen_compressed[b])`` — exactly mirroring the SBHD helper's
    ``offset=sq`` shift.

    Args:
        ratio: indexer compression ratio.
        cu_seqlens_q: ``(B+1,)`` int32 cumulative Q lengths.
        cu_seqlens_kv: ``(B+1,)`` int32 cumulative original-KV lengths
            (used to derive the per-segment compressed-offset).
        cu_seqlens_compressed: ``(B+1,)`` int32 cumulative compressed-KV
            lengths (== Compressor's second return value).
        total_q: total number of query tokens (avoids a GPU→CPU sync
            when the caller already knows it, e.g. from ``x.shape[0]``).
        max_n_compressed: max compressed sequence length across segments
            (avoids a GPU→CPU sync when the caller can derive it, e.g.
            ``max_seqlen_q // ratio``).

    Returns:
        ``(total_q, max_compressed_per_seq)`` int32 — LOCAL (per-segment)
        full-KV indices, ``-1`` for future positions.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    device = cu_seqlens_q.device
    seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    seq_lens_compressed = cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]
    if max_n_compressed is None:
        if seq_lens_compressed.numel() == 0:
            return torch.empty((total_q, 0), dtype=torch.int32, device=device)
        max_n_compressed = int(seq_lens_compressed.max().item())
    if max_n_compressed == 0:
        return torch.empty((total_q, 0), dtype=torch.int32, device=device)

    batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
    token_idx = torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
    row_valid = token_idx < cu_seqlens_q[-1]
    pos_in_seq = token_idx - cu_seqlens_q[batch_of_token]
    pos_in_seq = torch.where(row_valid, pos_in_seq, torch.zeros_like(pos_in_seq))

    n_valid_per_row = ((pos_in_seq + 1) // ratio).clamp(max=seq_lens_compressed[batch_of_token])
    n_valid_per_row = torch.where(row_valid, n_valid_per_row, torch.zeros_like(n_valid_per_row))
    offset_per_row = seq_lens_kv[batch_of_token]

    col_idx = (
        torch.arange(max_n_compressed, device=device, dtype=cu_seqlens_q.dtype)
        .unsqueeze(0)
        .expand(total_q, -1)
    )
    valid = col_idx < n_valid_per_row.unsqueeze(1)
    matrix = torch.where(valid, col_idx + offset_per_row.unsqueeze(1), torch.full_like(col_idx, -1))
    return matrix.int()


def build_cu_seqlens_kv_full(
    cu_seqlens_kv: torch.Tensor, cu_seqlens_compressed: torch.Tensor
) -> torch.Tensor:
    """Cumulative sequence lengths for the per-segment-concatenated
    ``kv_full_thd = cat_per_seg([kv_thd, compressed_kv_thd])``.

    ``kv_full_thd[cu_seqlens_kv_full[b] + i]`` for ``i in [0, seqlen_kv[b])``
    is ``kv_thd[cu_seqlens_kv[b] + i]``; for ``i in [seqlen_kv[b],
    seqlen_kv[b] + seqlen_compressed[b])`` it's
    ``compressed_kv_thd[cu_seqlens_compressed[b] + (i - seqlen_kv[b])]``.
    """
    full_lens = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]) + (
        cu_seqlens_compressed[1:] - cu_seqlens_compressed[:-1]
    )
    return torch.cat(
        [
            torch.zeros(1, dtype=cu_seqlens_kv.dtype, device=cu_seqlens_kv.device),
            full_lens.cumsum(0).to(cu_seqlens_kv.dtype),
        ]
    )


def cat_per_segment(
    kv_thd: torch.Tensor,
    compressed_kv_thd: Optional[torch.Tensor],
    cu_seqlens_kv: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    cu_seqlens_kv_full: torch.Tensor,
) -> torch.Tensor:
    """Build ``kv_full_thd`` by per-segment concatenation of ``kv_thd`` and
    ``compressed_kv_thd`` (the THD equivalent of ``torch.cat([kv,
    compressed_kv], dim=0)`` in the SBHD path).

    Fully vectorized: computes destination indices for all tokens via
    ``batch_of_row`` + offset arithmetic and writes with two indexed
    assignments — no Python loop, no GPU→CPU sync.

    Args:
        kv_thd: ``(total_kv, *trailing)``.
        compressed_kv_thd: ``(total_comp, *trailing)`` or ``None`` if every
            segment had ``seqlen < ratio`` (returns ``kv_thd`` unchanged).
        cu_seqlens_kv:        ``(B+1,)`` int32.
        cu_seqlens_compressed:``(B+1,)`` int32.
        cu_seqlens_kv_full:   ``(B+1,)`` int32 (computed by
            :func:`build_cu_seqlens_kv_full`).

    Returns:
        ``(total_kv_full, *trailing)`` packed concat.
    """
    if compressed_kv_thd is None:
        return kv_thd

    total_kv = kv_thd.shape[0]
    # NOTE: we deliberately use compressed_kv_thd.shape[0] (capacity, possibly
    # padded for CUDA graph capture) rather than cu_seqlens_compressed[-1] (true
    # count).  The fallback routing on invalid compressed rows (below) writes to
    # indices in [total_kv, total_kv_full), so the tail-padding slots *must*
    # exist in ``out``.  Do not shrink this allocation to true-count without
    # also updating the invalid-row routing logic.
    total_kv_full = total_kv + compressed_kv_thd.shape[0]
    device = kv_thd.device
    out_shape = (total_kv_full,) + tuple(kv_thd.shape[1:])
    out = torch.empty(out_shape, dtype=kv_thd.dtype, device=device)

    # KV tokens: dst[i] = cu_full[b] + (i - cu_kv[b])
    batch_of_kv = batch_of_row(cu_seqlens_kv, total_q=total_kv)
    src_kv = torch.arange(total_kv, device=device, dtype=cu_seqlens_kv.dtype)
    valid_kv = src_kv < cu_seqlens_kv[-1]
    dst_kv = cu_seqlens_kv_full[batch_of_kv] + (src_kv - cu_seqlens_kv[batch_of_kv])
    # Invalid (padding) KV rows must be routed to tail-pad slots in
    # ``out`` — using ``src_kv`` here is unsafe when ``total_kv >
    # cu_seqlens_kv[-1]`` because the padding rows' src indices fall
    # inside the valid-kv_full range and race with real-segment writes.
    # ``out`` is sized ``total_kv + total_comp_capacity`` so any slot in
    # ``[total_kv_full - n_invalid_kv, total_kv_full)`` is reserved
    # tail-padding (compressed-invalid uses the same region; duplicate
    # writes there are harmless since no valid final index reads them).
    dst_kv = torch.where(valid_kv, dst_kv, torch.full_like(dst_kv, total_kv_full - 1))
    out[dst_kv] = kv_thd

    # Compressed tokens: dst[j] = cu_full[b] + kv_len[b] + (j - cu_comp[b]).
    # ``compressed_kv_thd`` may be capacity-padded for CUDA graph capture; rows
    # beyond ``cu_seqlens_compressed[-1]`` are written to tail padding slots that
    # no valid final idx can reference.
    total_comp_capacity = compressed_kv_thd.shape[0]
    if total_comp_capacity > 0:
        kv_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
        src_comp = torch.arange(
            total_comp_capacity, device=device, dtype=cu_seqlens_compressed.dtype
        )
        batch_of_comp = batch_of_row(cu_seqlens_compressed, total_q=total_comp_capacity)
        valid_comp = src_comp < cu_seqlens_compressed[-1]
        dst_comp = (
            cu_seqlens_kv_full[batch_of_comp]
            + kv_lens[batch_of_comp]
            + (src_comp - cu_seqlens_compressed[batch_of_comp])
        )
        dst_comp = torch.where(valid_comp, dst_comp, total_kv + src_comp)
        out[dst_comp] = compressed_kv_thd

    return out


# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------



@dataclass
class CompressorSubmodules:
    """Submodule specs for CSA and HCA Compressor."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


class Compressor(MegatronModule):
    """Gated pooling compressor for CSA and HCA sparse attention.

    Compresses a sequence of tokens into a shorter sequence by pooling groups of
    ``compress_ratio`` tokens using learned gated weights.

    For ``compress_ratio == 4``, overlapping compression is used (``coff = 2``).
    For ``compress_ratio == 128``, non-overlapping compression is used (``coff = 1``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the compressor submodules and compression parameters."""
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        self.linear_wkv = build_module(
            submodules.linear_wkv,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        for param in self.linear_wkv.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'average_gradients_across_tp_domain', True)

        self.linear_wgate = build_module(
            submodules.linear_wgate,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        for param in self.linear_wgate.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'average_gradients_across_tp_domain', True)

        # keep to high precision
        _ape = torch.empty(
            compress_ratio, proj_out_dim, device=torch.cuda.current_device(), dtype=torch.float32
        )
        config.init_method(_ape)
        self.ape = nn.Parameter(_ape)
        if config.tensor_model_parallel_size > 1:
            setattr(self.ape, "average_gradients_across_tp_domain", True)

        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.norm = build_module(
            submodules.norm, config=norm_config, hidden_size=head_dim, eps=config.layernorm_epsilon
        )
        # compressor.norm sees full-seq input (compressor always runs on gathered
        # sequence even under SP), so all ranks compute identical grads -> AVG
        # across TP (overrides standard SP=SUM via if/elif)
        for param in self.norm.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'average_gradients_across_tp_domain', True)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        """Apply overlapping window transform for 4x compression.

        Input shape:  [n_groups, ratio, b, coff * head_dim]
        Output shape: [n_groups, 2 * ratio, b, head_dim]
        """
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compress hidden states into shorter KV sequence.

        Args:
            x: [sq, b, hidden_size]

        Returns:
            compressed_kv [sq // ratio, b, head_dim] or None if too short.
        """
        nvtx_range_push("compressor")

        sq, b, _ = x.size()
        ratio = self.compress_ratio

        if sq < ratio:
            nvtx_range_pop("compressor")
            return None

        # Pad sequence length to multiple of 128 for FP8 blockwise GEMM compatibility
        pad_len = (128 - sq % 128) % 128 if self.config.fp8 else 0
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
        else:
            x_padded = x
        kv, _ = self.linear_wkv(x_padded)  # [sq+pad, b, coff * head_dim]
        score, _ = self.linear_wgate(x_padded)  # [sq+pad, b, coff * head_dim]
        if pad_len > 0:
            kv = kv[:sq]
            score = score[:sq]

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]

        n_compressed = cutoff // ratio

        # Reshape: [n_compressed, ratio, b, coff * head_dim]
        kv = kv.view(n_compressed, ratio, b, -1)
        score = score.view(n_compressed, ratio, b, -1)

        # APE: [ratio, coff * head_dim] -> [1, ratio, 1, coff * head_dim]
        score = score + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        kv = (kv * torch.softmax(score, dim=1)).sum(dim=1)  # [n_compressed, b, head_dim]

        kv = self.norm(kv.to(x.dtype))

        kv = _apply_rope(
            kv,
            self.head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            n_compressed,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        if self.rotate:
            kv = rotate_activation(kv)

        nvtx_range_pop("compressor")
        return kv  # [n_compressed, b, head_dim]


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


    def _overlap_transform_thd(
        self, tensor: torch.Tensor, is_first_in_seg: torch.Tensor, fill_value: float = 0
    ) -> torch.Tensor:
        """Batched overlapping window transform for THD packed layout.

        Input shape:  [total_comp, ratio, b, coff * head_dim]
        Output shape: [total_comp, 2 * ratio, b, head_dim]
        """
        n, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
        prev_data[is_first_in_seg] = fill_value
        new_tensor[:, :ratio] = prev_data
        return new_tensor

    def _forward_thd(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_q: Optional[int] = None,
        fixed_total_comp: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """THD per-segment compression - fully vectorized.

        Args:
            x: (total, 1, hidden_size) packed bf16.
            cu_seqlens: (B+1,) int32 cumulative seq lengths.
            max_seqlen_q: max original sequence length.
            fixed_total_comp: static capacity for CUDA graph capture.

        Returns:
            (compressed_thd, cu_seqlens_compressed) where
            compressed_thd is (total_compressed, 1, head_dim) or None,
            cu_seqlens_compressed is (B+1,) int32.
        """
        ratio = self.compress_ratio
        device = x.device
        dtype = x.dtype

        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        seg_compressed_lens = seq_lens // ratio
        cu_seqlens_compressed = torch.cat([
            torch.zeros(1, dtype=cu_seqlens.dtype, device=device),
            seg_compressed_lens.cumsum(0).to(cu_seqlens.dtype),
        ])
        total_comp = (
            int(fixed_total_comp)
            if fixed_total_comp is not None
            else int(cu_seqlens_compressed[-1].item())
        )

        if total_comp == 0:
            return None, cu_seqlens_compressed

        # Token-wise projections on flat input (with FP8 pad if needed)
        total_tokens = x.shape[0]
        pad_len = (128 - total_tokens % 128) % 128 if self.config.fp8 else 0
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
        else:
            x_padded = x
        kv, _ = self.linear_wkv(x_padded)
        score, _ = self.linear_wgate(x_padded)
        if pad_len > 0:
            kv = kv[:total_tokens]
            score = score[:total_tokens]

        # Build gather index: (total_comp, ratio)
        row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_compressed.dtype)
        batch_ids = batch_of_row(cu_seqlens_compressed, total_q=total_comp)
        valid_comp = row_idx < cu_seqlens_compressed[-1]
        local_pos = row_idx - cu_seqlens_compressed[batch_ids]
        local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
        base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
        base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
        offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
        gather_idx = base + offsets  # (total_comp, ratio)

        kv_grouped = kv[gather_idx]  # (total_comp, ratio, 1, coff * d)
        score_grouped = score[gather_idx]

        score_grouped = score_grouped + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            is_first = local_pos == 0
            kv_grouped = self._overlap_transform_thd(kv_grouped, is_first, fill_value=0)
            score_grouped = self._overlap_transform_thd(
                score_grouped, is_first, fill_value=float('-inf')
            )

        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32).to(kv_grouped.dtype)
        compressed_thd = (kv_grouped * weights).sum(dim=1)

        compressed_thd = self.norm(compressed_thd.to(dtype))

        max_seqlen_rope = (max_seqlen_q // ratio) * ratio if max_seqlen_q is not None else None
        compressed_thd = _apply_rope(
            compressed_thd,
            self.head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            rotary_seq_len=0,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
            cu_seqlens=cu_seqlens_compressed,
            max_seqlen_rope=max_seqlen_rope,
        )

        if self.rotate:
            compressed_thd = rotate_activation(compressed_thd)
        return compressed_thd, cu_seqlens_compressed


class CSAIndexerKernelFunction(torch.autograd.Function):
    """Fused FP8 indexer kernel for CSA, avoiding full [sq, sk] materialization.

    Adapted from DSAIndexerKernel in AIAK-Training-Omni. Uses deep_gemm.fp8_mqa_logits
    to compute scores in a memory-efficient manner with FP8 quantization.
    """

    quantizer = None  # Lazily initialized

    @staticmethod
    def _get_quantizer():
        if CSAIndexerKernelFunction.quantizer is None:
            CSAIndexerKernelFunction.quantizer = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                amax_epsilon=1e-12,
                force_pow_2_scales=True,
                block_scaling_dim=1,
            )
        return CSAIndexerKernelFunction.quantizer

    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,  # [sq_local, b, h, d]
        index_k: torch.Tensor,  # [sk, b, d]  (full sk, not SP-sliced)
        weights: torch.Tensor,  # [sq_local, b, h]
        index_topk: int,
        compress_ratio: int,
        sq_offset: int = 0,  # global sq position offset for this rank's slice
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """FP8 fused indexer forward for CSA compressed keys.

        When SP is enabled, ``index_q`` / ``weights`` are SP-sliced along sq
        and ``sq_offset`` gives the global starting position of the slice so
        the causal-mask range over compressed keys stays correct.

        When ``packed_seq_params`` is provided (THD packed layout), the kernel
        uses ``cu_seqlens_q`` / ``cu_seqlens_kv`` to build per-query causal
        masks respecting segment boundaries.
        """
        sq, bsz, head, dim = index_q.size()
        sk = index_k.size(0)
        assert bsz == 1, "CSAIndexerKernel only supports batch_size=1"
        assert dim == 128, "Only support dim=128"
        device = index_q.device

        softmax_scale = dim ** -0.5

        # Squeeze batch dim for deep_gemm: [sq, h, d] and [sk, d]
        q = index_q.squeeze(1).contiguous()  # [sq, h, d]
        k = index_k.squeeze(1).contiguous()  # [sk, d]
        w = weights.squeeze(1).contiguous()  # [sq, h]

        # Pad k rows to multiple of 4 so FP8 quantizer scale aligns with data
        sk_orig = k.shape[0]
        sk_aligned = (sk_orig + 3) // 4 * 4
        if sk_aligned != sk_orig:
            k = torch.nn.functional.pad(k, (0, 0, 0, sk_aligned - sk_orig))

        quantizer = CSAIndexerKernelFunction._get_quantizer()
        quantized_q = quantizer.quantize(q)
        quantized_k = quantizer.quantize(k)
        q_fp8 = quantized_q.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        k_fp8 = quantized_k.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        q_scale = quantized_q._rowwise_scale_inv.reshape(q.shape[:-1])  # [sq, h]
        k_scale = quantized_k._rowwise_scale_inv.reshape(k.shape[0])  # [sk_aligned]

        weight_scaled = w * q_scale * softmax_scale

        if packed_seq_params is not None:
            # --- Packed (THD) path ---
            # cu_seqlens_q: boundaries of query segments in the packed sequence
            # cu_seqlens_kv: boundaries of compressed key segments
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            cu_seqlens_kv = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            )

            # For each query token, find which segment it belongs to
            seg_lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).int()
            q_seg_ids = torch.repeat_interleave(
                torch.arange(seg_lens_q.size(0), device=device, dtype=torch.int), seg_lens_q
            )  # [total_q]

            # k_start: each query's compressed keys start at cu_seqlens_kv[seg_id]
            k_start = cu_seqlens_kv[q_seg_ids].int()

            # Position within segment for each query
            pos_in_seg = torch.arange(sq, device=device, dtype=torch.int) - cu_seqlens_q[q_seg_ids].int()

            # k_end: k_start + (pos_in_seg + 1) // compress_ratio
            seg_lens_kv = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).int()
            k_end_local = (pos_in_seg + 1) // compress_ratio
            # Clamp to segment's actual compressed key length
            k_end_local = torch.minimum(k_end_local, seg_lens_kv[q_seg_ids])
            k_end = k_start + k_end_local
            # Ensure k_end > k_start (at least 1 key visible)
            #k_end = torch.maximum(k_end, k_start + 1)

            max_seqlen_k = int(packed_seq_params.max_seqlen_kv)
            index_score = deep_gemm.fp8_mqa_logits(
                q_fp8, (k_fp8, k_scale), weight_scaled,
                k_start, k_end,
                clean_logits=False,
                max_seqlen_k=max_seqlen_k,
            )
            # Post-process: mask out positions beyond valid range
            mask = torch.arange(max_seqlen_k, device=device)[None, :] < (k_end - k_start)[:, None]
            index_score = index_score.masked_fill(~mask, float('-inf'))

            # Top-k
            effective_topk = min(index_topk, max_seqlen_k)
            if effective_topk <= index_score.size(-1):
                index_score_topk, topk_indices = flashinfer.top_k(
                    index_score.contiguous(), effective_topk, sorted=False
                )
            else:
                index_score_topk, topk_indices = index_score.topk(effective_topk, dim=-1)

            # Contract with THD callers and with the unfused fused_qk_topk_naive_thd:
            # topk_indices are SEGMENT-LOCAL ([0, n_comp_of_this_seg)) with -1
            # sentinel for invalid slots. The caller applies causal filtering and
            # adds the kv_full segment offset itself, so we must NOT add k_start here.
            topk_valid = topk_indices < (k_end - k_start).unsqueeze(1)
            topk_indices = torch.where( 
                topk_valid,
                topk_indices,
                torch.full_like(topk_indices, -1),
            )

        else:
            # --- Original single-sequence (SBHD) path ---
            # Causal mask for compressed keys: k_end[i_global] = (i_global + 1) // compress_ratio.
            k_start = torch.zeros(sq, dtype=torch.int, device=device)
            k_end = (
                torch.arange(sq, dtype=torch.int, device=device) + sq_offset + 1
            ) // compress_ratio
            # Clamp k_end to at least 1 to avoid empty ranges for early positions
            k_end = k_end.clamp(min=1)

            index_score = deep_gemm.fp8_mqa_logits(
                q_fp8, (k_fp8, k_scale), weight_scaled, k_start, k_end
            )

            # Top-k
            effective_topk = min(index_topk, sk)
            if effective_topk <= index_score.size(-1):
                index_score_topk, topk_indices = flashinfer.top_k(
                    index_score.contiguous(), effective_topk, sorted=False
                )
            else:
                index_score_topk, topk_indices = index_score.topk(effective_topk, dim=-1)

        ctx.softmax_scale = softmax_scale
        ctx.index_topk = effective_topk
        ctx.sk_orig = sk_orig
        ctx.save_for_backward(q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, k_start, k_end)

        return index_score_topk, topk_indices

    @staticmethod
    def backward(ctx, grad_score, grad_topk):
        """FP8 fused indexer backward using lightning_indexer_bwd."""
        q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, ks, ke = ctx.saved_tensors

        d_q, d_k, d_weights = lightning_indexer_bwd.fp8_mqa_logits_bwd(
            grad_score.contiguous(),
            q_fp8,
            (k_fp8, k_scale),
            weight_scaled,
            ks,
            ke,
            topk_indices=topk_indices.int(),
            topk=ctx.index_topk,
        )

        d_weights = d_weights * q_scale * ctx.softmax_scale
        d_q = d_q / q_scale.unsqueeze(-1)
        d_k = d_k[:ctx.sk_orig] / k_scale[:ctx.sk_orig].unsqueeze(-1)

        # Unsqueeze batch dim back: [sq, 1, h, d], [sk, 1, d], [sq, 1, h]
        # Last 4 None: index_topk, compress_ratio, sq_offset, packed_seq_params (no grad)
        return d_q.unsqueeze(1), d_k.unsqueeze(1), d_weights.unsqueeze(1), None, None, None, None

class CSAIndexerKernel(torch.nn.Module):
    """Wrapper for fused FP8 indexer kernel for CSA."""

    def __init__(self, compress_ratio: int):
        super().__init__()
        self.compress_ratio = compress_ratio

    def forward(self, index_q, index_k, weights, index_topk, sq_offset=0, packed_seq_params=None):
        """Forward function."""
        return CSAIndexerKernelFunction.apply(
            index_q, index_k, weights, index_topk, self.compress_ratio, sq_offset, packed_seq_params
        )


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CSAIndexerSubmodules:
    """Submodule specs for CSAIndexer."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


class CSAIndexer(MegatronModule):
    """Learned top-k retrieval over compressed positions for CSA sparse attention.

    Computes index scores to select the most relevant compressed KV positions for each
    query.  Reuses the scoring logic from ``DSAIndexer`` (einsum -> relu -> weight -> sum
    -> topk) and ``rotate_activation`` (Hadamard transform) from ``dsa.py``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSAIndexerSubmodules,
        compress_ratio: int,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize CSA indexer projections and its rotated compressor."""
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        )

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        # Q projection
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # FIX: TE Linear.reset_parameters unconditionally calls
        # set_tensor_model_parallel_attributes(is_parallel=True, dim=0) regardless of
        # parallel_mode="duplicated", which makes distrib_optimizer treat this duplicated
        # weight as a TP-shard and allocate only 1/tp_size of the main_grad buffer.
        # Backward writes full-size grads into a 1/8 buffer => address-wrap accumulation,
        # producing the indexer.linear_wq_b grad explosion. Force replicated attrs.
        for param in self.linear_wq_b.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'tensor_model_parallel', False)
                setattr(param, 'partition_dim', -1)
                setattr(param, 'partition_stride', 1)
                setattr(param, 'average_gradients_across_tp_domain', True)

        # Weights projection
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        # FIX: same as linear_wq_b above.
        for param in self.linear_weights_proj.parameters():
            if config.tensor_model_parallel_size > 1:
                setattr(param, 'tensor_model_parallel', False)
                setattr(param, 'partition_dim', -1)
                setattr(param, 'partition_stride', 1)
                setattr(param, 'average_gradients_across_tp_domain', True)

        # Own compressor (smaller head_dim, with Hadamard rotation)
        self.compressor = build_module(
            submodules.compressor,
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
            pg_collection=pg_collection,
        )

        self.use_fused_indexer = getattr(config, 'use_fused_indexer', True)
        if self.use_fused_indexer:
            self.indexer_kernel = CSAIndexerKernel(compress_ratio)

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ):
        """Compute Q, compressed K, and weights before top-k selection.

        Two layouts:

        * **SBHD** (``packed_seq_params=None``): Returns ``(q, k, weights)``.
        * **THD packed** (``packed_seq_params.qkv_format == 'thd'``):
          Returns ``(q, k, weights, cu_seqlens_compressed)``.
        """
        nvtx_range_push("indexer_before_topk")

        is_thd = packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', None) == 'thd'

        sq, bsz, _ = x.size()  # in THD: sq = total_q, bsz = 1.

        cu_seqlens_q = None
        max_seqlen_rope = None
        if is_thd:
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            max_seqlen_rope = (
                int(packed_seq_params.max_seqlen_q)
                if packed_seq_params.max_seqlen_q is not None
                else None
            )

        # Pad seq dim to multiple of 128 for FP8 Linear compatibility
        pad_len = (128 - sq % 128) % 128 if self.config.fp8 and not is_thd else 0
        if pad_len > 0:
            qr = torch.nn.functional.pad(qr, (0, 0, 0, 0, 0, pad_len))
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))

        # Q path
        q, _ = self.linear_wq_b(qr)
        if pad_len > 0:
            q = q[:sq]
        q = q.reshape(sq, bsz, self.index_n_heads, self.index_head_dim)
        q = _apply_rope(
            q,
            self.index_head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            rotary_seq_len=sq,
            ratio=1,
            cp_group=self.pg_collection.cp,
            cu_seqlens=cu_seqlens_q,
            max_seqlen_rope=max_seqlen_rope,
        )
        q = rotate_activation(q)

        # K path: own compressor
        if is_thd:
            if pad_len > 0:
                x_for_comp = x[:sq]
            else:
                x_for_comp = x
            k, cu_seqlens_compressed = self.compressor._forward_thd(
                x_for_comp, cu_seqlens_q, max_seqlen_q=max_seqlen_rope
            )
        else:
            if pad_len > 0:
                x_for_comp = x[:sq]
            else:
                x_for_comp = x
            k = self.compressor(x_for_comp)  # [sq//ratio, b, index_head_dim]
            # Align k length to multiple of 128 for FP8
            if k is not None:
                target_n = sq // self.compress_ratio
                target_n_aligned = ((target_n + 127) // 128) * 128
                if k.size(0) >= target_n_aligned:
                    k = k[:target_n_aligned]
                else:
                    _k_pad = target_n_aligned - k.size(0)
                    k = torch.cat([k, k.new_zeros((_k_pad, *k.shape[1:]))], dim=0)

        weights, _ = self.linear_weights_proj(x)
        if pad_len > 0:
            weights = weights[:sq]
        weights = weights * (self.index_n_heads**-0.5)

        nvtx_range_pop("indexer_before_topk")
        if is_thd:
            return q, k, weights, cu_seqlens_compressed
        return q, k, weights

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (index_scores, topk_indices).

        Supports both SBHD and THD packed layouts.
        """
        nvtx_range_push("indexer")
        is_thd = packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', None) == 'thd'

        if is_thd:
            q, k, weights, cu_seqlens_compressed_idx = self.forward_before_topk(
                x, qr, packed_seq_params
            )
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            nvtx_range_push("indexer_qk_topk")
            if k is None:
                # Every segment shorter than ratio -> no compressed keys
                total_q = q.shape[0]
                index_scores = None
                topk_indices = torch.full(
                    (total_q, self.index_topk), -1, dtype=torch.int64, device=q.device
                )
            elif self.use_fused_indexer:
                # Use fused FP8 CSAIndexerKernel with packed_seq_params
                effective_topk = min(self.index_topk, k.shape[0])
                max_seqlen_kv = int(cu_seqlens_compressed_idx.diff().max().item())
                csa_packed = PackedSeqParams(
                    qkv_format='thd',
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_compressed_idx,
                    max_seqlen_q=packed_seq_params.max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )
                # SP slicing for fused indexer to keep TE blockwise FP8 quantize
                # within CUDA grid limits at long seqlen. Match the non-packed
                # CSA path by splitting the local query length into two halves.
                sq_full = q.size(0)
                half = sq_full // 2
                cu_q_full = csa_packed.cu_seqlens_q
                straddles_half = bool(
                    ((cu_q_full[:-1] < half) & (cu_q_full[1:] > half)).any().item()
                )
                if (
                    half > 0
                    and half % 128 == 0
                    and (sq_full - half) % 128 == 0
                    and not straddles_half
                ):
                    score_chunks = []
                    topk_chunks = []
                    cu_q = csa_packed.cu_seqlens_q
                    cu_kv = csa_packed.cu_seqlens_kv
                    for chunk_start, chunk_end in ((0, half), (half, sq_full)):
                        chunk_q = q[chunk_start:chunk_end]
                        chunk_weights = weights[chunk_start:chunk_end]

                        # Build a packed view for this query slice. Since the slice can
                        # cut through a packed segment, clamp the original segment
                        # boundaries to [chunk_start, chunk_end] and rebase to zero.
                        q_starts = torch.clamp(cu_q[:-1], min=chunk_start, max=chunk_end)
                        q_ends = torch.clamp(cu_q[1:], min=chunk_start, max=chunk_end)
                        keep = q_ends > q_starts
                        q_starts = q_starts[keep]
                        q_ends = q_ends[keep]
                        if q_starts.numel() == 0:
                            continue
                        chunk_cu_q = torch.empty(
                            q_starts.numel() + 1, device=cu_q.device, dtype=cu_q.dtype
                        )
                        chunk_cu_q[0] = 0
                        chunk_cu_q[1:] = torch.cumsum((q_ends - q_starts).to(cu_q.dtype), dim=0)

                        # Reuse the corresponding compressed-KV segments. They are
                        # small relative to q and keep the original indexing semantics.
                        seg_ids = torch.nonzero(keep, as_tuple=False).flatten()
                        first_seg = int(seg_ids[0].item())
                        last_seg = int(seg_ids[-1].item())
                        kv_base = cu_kv[first_seg]
                        chunk_cu_kv = cu_kv[first_seg : last_seg + 2] - kv_base
                        chunk_k = k[kv_base : cu_kv[last_seg + 1]]

                        chunk_packed = PackedSeqParams(
                            qkv_format='thd',
                            cu_seqlens_q=chunk_cu_q,
                            cu_seqlens_kv=chunk_cu_kv,
                            max_seqlen_q=int((chunk_cu_q[1:] - chunk_cu_q[:-1]).max().item()),
                            max_seqlen_kv=int((chunk_cu_kv[1:] - chunk_cu_kv[:-1]).max().item()),
                        )
                        scores_i, topk_i = self.indexer_kernel(
                            chunk_q, chunk_k, chunk_weights, effective_topk, 0, chunk_packed
                        )
                        kv_base_device = kv_base.to(topk_i.device)
                        topk_i = torch.where(topk_i >= 0, topk_i + kv_base_device, topk_i)
                        score_chunks.append(scores_i)
                        topk_chunks.append(topk_i)

                    max_topk = max(chunk.size(1) for chunk in topk_chunks)
                    for i, (scores_i, topk_i) in enumerate(zip(score_chunks, topk_chunks)):
                        pad = max_topk - topk_i.size(1)
                        if pad > 0:
                            score_pad = torch.full(
                                (scores_i.size(0), pad),
                                torch.finfo(scores_i.dtype).min,
                                device=scores_i.device,
                                dtype=scores_i.dtype,
                            )
                            topk_pad = torch.full(
                                (topk_i.size(0), pad), -1, device=topk_i.device, dtype=topk_i.dtype
                            )
                            score_chunks[i] = torch.cat((scores_i, score_pad), dim=1)
                            topk_chunks[i] = torch.cat((topk_i, topk_pad), dim=1)
                    index_scores = torch.cat(score_chunks, dim=0)
                    topk_indices = torch.cat(topk_chunks, dim=0)
                else:
                    index_scores, topk_indices = self.indexer_kernel(
                        q, k, weights, effective_topk, 0, csa_packed
                    )
            else:
                # Squeeze the dummy b=1 dim
                q_thd = q.squeeze(1)
                k_thd = k.squeeze(1)
                w_thd = weights.squeeze(1)
                effective_topk = min(self.index_topk, k_thd.shape[0])
                index_scores, topk_indices = fused_qk_topk_naive_thd(
                    q_thd,
                    k_thd,
                    w_thd,
                    index_topk=effective_topk,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_compressed_idx,
                    ratio=self.compress_ratio,
                )
            nvtx_range_pop("indexer_qk_topk")
            nvtx_range_pop("indexer")
            return index_scores, topk_indices

        # SBHD path
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)
        nvtx_range_push("indexer_qk_topk")
        effective_topk = min(self.index_topk, k.size(0))
        if self.use_fused_indexer:
            # SP slicing for fused indexer
            tp_size = get_pg_size(self.pg_collection.tp)
            use_sp = self.config.sequence_parallel and tp_size > 1
            if use_sp:
                sq_full = q.size(0)
                half = sq_full // 2
                if half % 128 != 0 or (sq_full - half) % 128 != 0:
                    index_scores, topk_indices = self.indexer_kernel(q, k, weights, effective_topk, 0, None)
                else:
                    s1, t1 = self.indexer_kernel(q[:half], k, weights[:half], effective_topk, 0, None)
                    s2, t2 = self.indexer_kernel(q[half:], k, weights[half:], effective_topk, half, None)
                    index_scores = torch.cat([s1, s2], dim=0)
                    topk_indices = torch.cat([t1, t2], dim=0)
            else:
                index_scores, topk_indices = self.indexer_kernel(q, k, weights, effective_topk, 0, None)
        else:
            index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, effective_topk, mask)
        nvtx_range_pop("indexer_qk_topk")
        nvtx_range_pop("indexer")
        return index_scores, topk_indices



# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for CompressedSparseAttention."""

    compressor: Union[ModuleSpec, type] = None
    indexer: Union[ModuleSpec, type] = None


class CompressedSparseAttention(MegatronModule):
    """Sparse core attention for CompressedSparseAttention.

    Combines sliding window attention with compressed KV attention.  The spec always
    provides compressor and indexer submodule specs; this ``__init__`` inspects
    ``config.csa_compress_ratios[layer_idx]`` and conditionally builds them:

    * ``ratio == 0``:  window-only (compressor and indexer NOT built)
    * ``ratio == 4``:  window + 4x compressed + learned Indexer (both built)
    * ``ratio == 128``: window + 128x compressed, attend to all (compressor built only)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module = None,
        compress_ratio: int = 0,
        is_mtp_layer: bool = False,
    ):
        """Initialize compressed sparse attention kernels and optional indexer."""
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        self.layer_number = layer_number
        
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim

        self.n_local_heads = divide(config.num_attention_heads, get_pg_size(self.pg_collection.tp))

        if softmax_scale is None:
            softmax_scale = config.v_head_dim**-0.5
        self.softmax_scale = softmax_scale

        self.force_unfused_dsa = getattr(config, "force_unfused_dsa", False)
        self.apply_dsa_kernel_fusion = getattr(config, "apply_dsa_kernel_fusion", False)

        # Choose between loongforge fused kernel (default) and mcore fused kernel.
        # Set env var USE_AIAK_MEGATRON_FUSED=1 or config.use_mcore_fused=True
        # to use AIAK-Megatron's dsa_sparse_attn (requires flash_mla with sparse_fwd).
        self.use_mcore_fused = (
            os.environ.get("USE_MCORE_FUSED", "0") == "1"
            or getattr(config, "use_mcore_fused", False)
        )

        # Learnable attention sink per head
        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))
        if config.tensor_model_parallel_size > 1:
            setattr(self.attn_sink, "tensor_model_parallel", True)

        # Conditionally build Compressor (ratio > 1)
        if self.compress_ratio > 1 and submodules.compressor is not None:
            self.compressor = build_module(
                submodules.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer (ratio == 4)
        if (
            self.compress_ratio == 4
            and not config.csa_dense_mode
            and submodules.indexer is not None
        ):
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
            )
            # DIAG: tag layer_number on indexer for forward diag prints
            self.indexer.layer_number = self.layer_number
        else:
            self.indexer = None


        if not self.force_unfused_dsa:
            try:
                from loongforge.models.common.experimental_attention_variant.dsa_fused_kernels import (
                    DSADotProductAttention as _DSAKernel,
                )
            except ImportError:
                raise ImportError(
                    "Fused sparse attention requires DSADotProductAttention kernel. "
                    "Set force_unfused_dsa=True in config or install the required kernel."
                )
            self.sparse_attention = _DSAKernel(config=config, softmax_scale=self.softmax_scale)


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor = None,
        qr: torch.Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ) -> torch.Tensor:
        """Forward pass for CompressedSparseAttention.

        Args:
            query:  [sq, b, np, v_head_dim]
            key:    [sq, b, 1, v_head_dim]  (single-head MQA; head dim squeezed internally)
            value:  unused (key == value in MQA)
            attention_mask: attention mask (may be None for causal).
            x:      [sq, b, hidden_size]  original hidden states.
            qr:     [sq, b, q_lora_rank]  compressed query representation.

        Returns:
            output: [sq, b, np * v_head_dim]
        """
        nvtx_range_push("compressed_sparse_attn")
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            output = self._forward_thd(query, key, x, qr, packed_seq_params)
            nvtx_range_pop("compressed_sparse_attn")
            return output

        sq, b, np, hn = query.size()
        # --- Step 1: Prepare single-head KV (squeeze singleton head dim) ---
        kv = key.squeeze(-2)  # [sq, b, 1, v_head_dim] -> [sq, b, v_head_dim]

        # --- Step 2: Compression ---
        # Skip compression when sq < window_size: window already covers all positions
        # (full attention contained in window), compressed path would only dilute softmax
        # by splitting probability mass across full-res window and low-res compressed views.
        if (self.compressor is not None and self.compress_ratio > 1
                and sq >= self.window_size):
            compressed_kv = self.compressor(x)  # [n_compressed, b, v_head_dim]
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed indices start after original positions

        # --- Step 3: Window indices ---
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)

        # --- Step 4: Compressed indices ---
        indexer_loss = None

        if self.force_unfused_dsa:
            if self.compress_ratio > 1 and n_compressed > 0:
                nvtx_range_push("compressed_indices")
                if self.indexer is not None:
                    x_det = x.detach()
                    qr_det = qr.detach()

                    causal_mask = (
                        torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                    )
                    positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                    causal_mask = (
                        torch.where(
                            causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0
                        )
                        .unsqueeze(0)
                        .expand(b, -1, -1)
                    )  # [b, sq, n_compressed]

                    indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', None)
                    if self.training and torch.is_grad_enabled() and indexer_loss_coeff is not None:
                        q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                            x_det, qr_det, packed_seq_params
                        )
                        # compressed_kv is [n, b, hn]; expand to [n, b, np, hn] for loss
                        key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, np, -1)
                        # ``FusedDSAIndexerLoss`` does not accept a separate
                        # indexer_softmax_scale; apply it here via the
                        # weights-scaling trick so the effective weights match
                        # the pre-scale-split behaviour.
                        weights_for_unfused = weights_indexer * self.indexer.softmax_scale
                        topk_indices_compressed, indexer_loss = FusedDSAIndexerLoss.apply(
                            q_indexer,
                            weights_for_unfused,
                            k_indexer,
                            query.detach(),
                            key_for_loss.detach(),
                            self.softmax_scale,
                            min(self.indexer.index_topk, n_compressed),
                            indexer_loss_coeff,
                            causal_mask,
                            getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                            self.indexer.pg_collection,
                            self.config.calculate_per_token_loss,
                        )
                        if indexer_loss_coeff > 0:
                            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                                loss=indexer_loss,
                                layer_number=self.layer_number,
                                num_layers=self.config.num_layers
                                + (self.config.mtp_num_layers or 0),
                            )
                    else:
                        _, topk_indices_compressed = self.indexer(
                            x_det, qr_det, mask=causal_mask, packed_seq_params=packed_seq_params
                        )

                    # Normalize topk_indices_compressed to 3D [b, sq, topk]:
                    # FusedDSAIndexerLoss / fused-indexer return 2D [sq, topk];
                    # naive indexer returns 3D [b, sq, topk]. Below cat() with
                    # 3D window_idxs requires 3D, so unsqueeze when needed.
                    if topk_indices_compressed.dim() == 2:
                        topk_indices_compressed = topk_indices_compressed.unsqueeze(0)

                    n_valid_per_pos = positions // self.compress_ratio  # [sq, 1]
                    valid = topk_indices_compressed < n_valid_per_pos
                    compress_topk_idxs = torch.where(
                        valid, topk_indices_compressed + offset, torch.tensor(-1, device=x.device)
                    )
                else:
                    compress_topk_idxs = get_compress_topk_idxs(
                        self.compress_ratio, b, sq, offset, query.device
                    )

                topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
                nvtx_range_pop("compressed_indices")
            else:
                topk_idxs = window_idxs

            topk_idxs = topk_idxs.int()

            # --- Step 5: Sparse attention ---
            nvtx_range_push("sparse_attn_kernel")
            output = unfused_compressed_sparse_attn(
                query, kv_full, self.attn_sink.float(), topk_idxs, self.softmax_scale
            )
            nvtx_range_pop("sparse_attn_kernel")

        else:
            tp_group = self.pg_collection.tp
            tp_size = get_pg_size(tp_group)
            tp_rank = tp_group.rank() if tp_size > 1 else 0

            chunk_query = _AllToAllHp2Sp.apply(query, tp_group)

            chunk_sq = chunk_query.size(0)
            chunk_offset = kv_full.size(0)

            chunk_start = tp_rank * chunk_sq
            chunk_end = chunk_start + chunk_sq

            if self.compress_ratio > 1 and n_compressed > 0:
                if self.indexer is not None:
                    x_det = x.detach()
                    qr_det = qr.detach()

                    causal_mask = (
                        torch.arange(n_compressed, device=x.device).unsqueeze(0).expand(sq, -1)
                    )
                    positions = torch.arange(1, sq + 1, device=x.device).unsqueeze(1)
                    causal_mask = (
                        torch.where(
                            causal_mask >= positions // self.compress_ratio, float("-inf"), 0.0
                        )
                        .unsqueeze(0)
                        .expand(b, -1, -1)
                    )  # [b, sq, n_compressed]

                    index_scores, topk_indices_compressed = self.indexer(
                        x_det, qr_det, mask=causal_mask, packed_seq_params=packed_seq_params
                    )

                    # Fused kernel returns [sq, topk]; naive returns [b, sq, topk]
                    # Normalize to [b, sq, topk] for downstream code
                    if topk_indices_compressed.dim() == 2:
                        topk_indices_compressed = topk_indices_compressed.unsqueeze(0)
                        index_scores = index_scores.unsqueeze(0)

                    n_valid_per_pos = positions // self.compress_ratio  # [sq, 1]
                    valid = topk_indices_compressed < n_valid_per_pos
                    compress_topk_idxs = torch.where(
                        valid, topk_indices_compressed + offset, torch.tensor(-1, device=x.device)
                    )
                else:
                    compress_topk_idxs = get_compress_topk_idxs(
                        self.compress_ratio, b, sq, offset, query.device
                    )

                topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1).int()
                chunk_topk_idxs = topk_idxs[:, chunk_start:chunk_end, :]
                _csa_pad = chunk_sq - chunk_topk_idxs.shape[1]
                if _csa_pad > 0:  # SP padding fix: pad truncated tail with -1 sentinel
                    chunk_topk_idxs = F.pad(chunk_topk_idxs, (0, 0, 0, _csa_pad), value=-1)
                chunk_topk_idxs = chunk_topk_idxs.contiguous()


                output, p_out = self.sparse_attention(
                    chunk_query,
                    kv_full,
                    chunk_topk_idxs,
                    chunk_offset,
                    attn_mask_type=AttnMaskType.causal,
                    return_p_out=True,
                    window_size=128,
                    attn_sink=self.attn_sink.float(),
                )

                indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', None)
                if (self.indexer is not None and self.training and torch.is_grad_enabled()
                        and indexer_loss_coeff is not None and indexer_loss_coeff > 0):
                    from loongforge.models.common.experimental_attention_variant.dsa_fused_kernels import (
                        triton_attn_dist,
                    )
                    import triton

                    row_valid = (positions.squeeze(-1) >= self.compress_ratio)  # [sq]
                    chunk_row_valid = row_valid[chunk_start:chunk_end]  # [chunk_sq] (may be short under SP-pad)
                    _csa_indexer_pad = chunk_sq - chunk_row_valid.size(0)
                    if _csa_indexer_pad > 0:
                        chunk_row_valid = F.pad(chunk_row_valid, (0, _csa_indexer_pad), value=False)

                    with torch.no_grad():
                        # p_out: [chunk_sq, h_q, p_out_topk] -> main_attn_probs: [chunk_sq, p_out_topk]
                        p_out_topk = p_out.size(-1)
                        p_out_for_dist = p_out
                        # triton_attn_dist requires topk to be power of 2; pad if needed
                        next_pow2 = triton.next_power_of_2(p_out_topk)
                        if next_pow2 != p_out_topk:
                            p_out_for_dist = F.pad(p_out, (0, next_pow2 - p_out_topk), value=float('-inf'))
                        main_attn_probs = triton_attn_dist(p_out_for_dist, self.softmax_scale)
                        if next_pow2 != p_out_topk:
                            main_attn_probs = main_attn_probs[:, :p_out_topk]
                        main_attn_probs = main_attn_probs.masked_fill(
                            ~chunk_row_valid.unsqueeze(-1), 0.0
                        )

                    index_scores_topk = (
                        index_scores
                        if self.indexer.use_fused_indexer
                        else index_scores.gather(-1, topk_indices_compressed)
                    )

                    if not self.indexer.use_fused_indexer:
                        index_scores_topk = index_scores_topk * self.indexer.softmax_scale
                    chunk_index_scores = index_scores_topk[:, chunk_start:chunk_end, :]
                    if _csa_indexer_pad > 0:  # SP padding fix: align sq dim with chunk_row_valid
                        chunk_index_scores = F.pad(chunk_index_scores, (0, 0, 0, _csa_indexer_pad), value=0.0)
                    chunk_index_scores = chunk_index_scores.squeeze(0)
                    chunk_index_scores = chunk_index_scores.masked_fill(
                        ~chunk_row_valid.unsqueeze(-1), 0.0
                    )
                    index_attn_probs = torch.softmax(chunk_index_scores, dim=-1)
                    index_attn_probs = index_attn_probs.masked_fill(
                        ~chunk_row_valid.unsqueeze(-1), 0.0
                    )

                    # flash_mla_sparse_fwd p_out: [s_q, h_q, topk_aligned64 - window_size].
                    # The last (topk_aligned64 - chunk_topk_idxs.size(-1)) entries are
                    # 64-alignment padding (-1), and the trailing portion of the
                    # compress-keys are dropped beyond effective_topk. So the first
                    # effective_topk entries of main_attn_probs correspond 1:1 with
                    # compress_topk_idxs == index_attn_probs's keyset.
                    if main_attn_probs.size(-1) > index_attn_probs.size(-1):
                        main_attn_probs = main_attn_probs[:, :index_attn_probs.size(-1)]

                    indexer_loss = F.kl_div(
                        (index_attn_probs + 1e-10).log(),
                        main_attn_probs + 1e-10,
                        reduction="sum",
                    )
                    indexer_loss = indexer_loss_coeff * indexer_loss / sq
                    indexer_loss = reduce_from_tensor_model_parallel_region(
                        indexer_loss, group=tp_group
                    )

                    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                        loss=indexer_loss,
                        layer_number=self.layer_number,
                        num_layers=self.config.num_layers
                        + (self.config.mtp_num_layers or 0),
                    )

                nvtx_range_pop("compressed_indices")

            else:
                topk_idxs = window_idxs.int()
                chunk_topk_idxs = topk_idxs[:, chunk_start:chunk_end, :]
                _csa_pad = chunk_sq - chunk_topk_idxs.shape[1]
                if _csa_pad > 0:  # SP padding fix: pad truncated tail with -1 sentinel
                    chunk_topk_idxs = F.pad(chunk_topk_idxs, (0, 0, 0, _csa_pad), value=-1)
                chunk_topk_idxs = chunk_topk_idxs.contiguous()

                output = self.sparse_attention(
                    chunk_query,
                    kv_full,
                    chunk_topk_idxs,
                    chunk_offset,
                    attn_mask_type=AttnMaskType.causal,
                    return_p_out=False,
                    window_size=0,
                    attn_sink=self.attn_sink.float(),
                )
                output = output.unsqueeze(0)

            output = _AllToAllSp2Hp.apply(output, tp_group)
            output = output[:, :sq, :].contiguous()  # SP padding fix: trim padded seq dim before reshape
            output = output.reshape(sq, b, -1)

        # --- Step 6: Attach indexer loss ---
        if indexer_loss is not None and self.training and torch.is_grad_enabled():
            # When inside a packed-sub-seq recursion, defer the attach: append
            # to the parent-collected list and let the outer packed dispatcher
            # do a single aggregated attach. See packed branch above for why.
            capture_list = getattr(self, "_capture_indexer_loss_list", None)
            if capture_list is not None:
                capture_list.append(indexer_loss)
            else:
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        nvtx_range_pop("compressed_sparse_attn")
        return output


    # ==================================================================
    # THD (packed sequence) native methods - ported from 0629 PR#5011
    # ==================================================================

    def _forward_thd(
        self,
        query: torch.Tensor,  # (total_q, np, hn)        TE THD convention
        key: torch.Tensor,  # (total_kv, 1, 1, hn)     packed, MQA
        x: torch.Tensor,  # (total_q, 1, hidden_size)
        qr: torch.Tensor,  # (total_q, 1, q_lora_rank)
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        """THD-packed branch of :meth:`forward`. See class docstring for layout.

        Performs common setup (shape validation, per-segment compression,
        full-KV layout construction, window indices) then dispatches to
        one of three per-path helpers:

        * :meth:`_forward_fused_no_indexer_thd` — window-only / window + all-compressed.
        * :meth:`_forward_fused_indexer_training_thd` — training + indexer + loss (returns
          directly with attached indexer loss).
        * :meth:`_forward_fused_indexer_inference_thd` — inference + indexer (no loss).

        Paths A and C return ``compress_topk_idxs`` which are globalized
        and fed to the fused/unfused sparse attention in Step 5 below.
        """
        # ---- Inputs / shape contract ----------------------------------------
        # query    : (total_q, np, hn)        multi-head Q (TE THD convention)
        # key      : (total_kv, 1, 1, hn)     packed single-head MQA KV (the
        #            DSv4 hybrid adds a dummy batch dim to keep the MQA-head
        #            unsqueeze symmetric with SBHD)
        # x, qr    : (total_q, 1, *)
        total_q, _np, _ = query.shape

        cu_seqlens_q = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        cu_seqlens_kv = (
            packed_seq_params.cu_seqlens_kv_padded
            if packed_seq_params.cu_seqlens_kv_padded is not None
            else packed_seq_params.cu_seqlens_kv
        )
        max_seqlen_q = int(packed_seq_params.max_seqlen_q)
        max_seqlen_kv = int(packed_seq_params.max_seqlen_kv)

        # Squeeze the dummy b=1 and MQA head-dim to get the KV-flat layout.
        # (key arrives as (total_kv, 1, 1, hn) for MQA.)
        kv_thd = key.squeeze(-2).squeeze(1)  # (total_kv, hn)

        # ---- Step 2: per-segment compression (vectorized THD) ----------------
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv, cu_seqlens_compressed = self.compressor._forward_thd(
                x, cu_seqlens_q, max_seqlen_q=max_seqlen_q
            )
            if compressed_kv is not None:
                compressed_kv = compressed_kv.squeeze(1)  # (total_comp, hn)
                n_compressed_total = compressed_kv.shape[0]
            else:
                n_compressed_total = 0
        else:
            compressed_kv = None
            cu_seqlens_compressed = torch.zeros_like(cu_seqlens_kv)
            n_compressed_total = 0

        # ---- Build full per-segment-concatenated KV layout ------------------
        cu_seqlens_kv_full = build_cu_seqlens_kv_full(cu_seqlens_kv, cu_seqlens_compressed)
        kv_full_thd = cat_per_segment(
            kv_thd, compressed_kv, cu_seqlens_kv, cu_seqlens_compressed, cu_seqlens_kv_full
        )

        # ---- Step 3: window indices (per-segment local) ---------------------
        window_idxs = get_window_topk_idxs_thd(
            self.window_size, cu_seqlens_q, total_q=total_q
        )  # (total_q, win_topk) local-to-segment

        # Upper bound on the max compressed-KV length per segment.  Not exact
        # when segment lengths aren't divisible by compress_ratio, but
        # cuDNN/flash kernels tolerate over-estimates (used only for tile sizing).
        max_seqlen_compressed_idx = (
            max_seqlen_q // self.compress_ratio if self.compress_ratio > 1 else 0
        )

        # ---- Step 4: path dispatch --------------------------------------------
        is_training = self.training and torch.is_grad_enabled()
        has_indexer = (
            self.compress_ratio > 1 and n_compressed_total > 0 and self.indexer is not None
        )

        indexer_loss = None

        if not self.apply_dsa_kernel_fusion:
            output, indexer_loss = self._forward_unfused_csa_thd(
                query,
                x,
                qr,
                kv_full_thd,
                compressed_kv,
                n_compressed_total,
                _np,
                total_q,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                cu_seqlens_compressed,
                window_idxs,
                max_seqlen_q,
                max_seqlen_compressed_idx,
                packed_seq_params,
            )
        elif has_indexer and is_training:
            output, indexer_loss = self._forward_fused_indexer_training_thd(
                query,
                x,
                qr,
                packed_seq_params,
                total_q,
                _np,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                max_seqlen_q,
                max_seqlen_compressed_idx,
                compressed_kv,
                kv_full_thd,
                window_idxs,
            )
        elif has_indexer:
            output = self._forward_fused_indexer_inference_thd(
                query,
                x,
                qr,
                kv_full_thd,
                packed_seq_params,
                total_q,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                window_idxs,
                max_seqlen_q,
                max_seqlen_compressed_idx,
                max_seqlen_kv,
            )
        else:
            output = self._forward_fused_no_indexer_thd(
                query,
                kv_full_thd,
                total_q,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_kv_full,
                cu_seqlens_compressed,
                n_compressed_total,
                window_idxs,
                max_seqlen_compressed_idx=max_seqlen_compressed_idx,
                packed_seq_params=packed_seq_params,
            )

        if indexer_loss is not None:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return output


    def _forward_unfused_csa_thd(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_full_thd: torch.Tensor,
        compressed_kv: Optional[torch.Tensor],
        n_compressed_total: int,
        np_: int,
        total_q: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        cu_seqlens_compressed: torch.Tensor,
        window_idxs: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_compressed_idx: int,
        packed_seq_params: PackedSeqParams,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch fallback path for THD (no fused kernels).

        Mirrors :meth:`_forward_unfused_csa` for the SBHD layout.
        Returns ``(output, indexer_loss)`` where *output* is
        ``(total_q, 1, np * hn)``.
        """
        device = query.device
        indexer_loss = None

        if self.compress_ratio > 1 and n_compressed_total > 0:
            if self.indexer is not None:
                x_det = x.detach()
                qr_det = qr.detach()

                if self.training and torch.is_grad_enabled():
                    q_indexer, k_indexer, weights_indexer, cu_seqlens_compressed_idx = (
                        self.indexer.forward_before_topk(x_det, qr_det, packed_seq_params)
                    )
                    if k_indexer is None:
                        raise RuntimeError(
                            "CompressedSparseAttention THD unfused Path B requires "
                            "at least one segment with compressed indexer K."
                        )
                    q_thd = q_indexer.squeeze(1)
                    w_thd = weights_indexer.squeeze(1)
                    k_thd = k_indexer.squeeze(1)

                    key_for_loss_thd = compressed_kv.unsqueeze(1).expand(-1, np_, -1)
                    weights_for_unfused = w_thd * self.indexer.softmax_scale
                    indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

                    # ``_forward_thd`` (caller) absorbs trailing padded
                    # tokens into the last ``cu_seqlens_q[-1]`` bucket
                    # so ``batch_of_row`` doesn't OOB; the Compressor
                    # does the same to ``cu_seqlens_compressed_idx[-1]``.
                    # Both are correct for the sparse-attention path,
                    # but the per-segment indexer-loss loop in
                    # ``fwd/bwd_fused_indexer_loss_naive_thd`` would
                    # then iterate a "fake" absorbed-padding segment
                    # with ``seqlen_k_b < topk`` — triggering a write
                    # shape mismatch and a downstream ``scatter_`` OOB
                    # on its ``-1`` entries.
                    #
                    # Restore the original (pre-absorption) cu_seqlens
                    # for the loss path so the segment loop's
                    # ``if seqlen_k_b == 0: continue`` guard skips the
                    # padding-only iteration. When no padding exists,
                    # ``packed_seq_params`` already equals the absorbed
                    # version and this is a no-op.
                    # Rebuild compressed cu_seqlens from *unpadded* Q lengths.
                    # This may disagree with the Indexer's cu_seqlens_compressed_idx
                    # (which uses padded lengths) for the last segment — that's
                    # intentional: the extra compressed tokens from padding sit at
                    # the tail of k_thd and are simply never visited by the loss
                    # loop, which is correct since they don't represent real data.
                    # Non-last segments are unaffected (padding absorption only
                    # extends the final segment).
                    cu_seqlens_q_for_loss = packed_seq_params.cu_seqlens_q
                    seg_lens_q = cu_seqlens_q_for_loss[1:] - cu_seqlens_q_for_loss[:-1]
                    cu_seqlens_compressed_idx_for_loss = torch.cat(
                        [
                            torch.zeros(
                                1,
                                dtype=cu_seqlens_q_for_loss.dtype,
                                device=cu_seqlens_q_for_loss.device,
                            ),
                            (seg_lens_q // self.compress_ratio)
                            .cumsum(0)
                            .to(cu_seqlens_q_for_loss.dtype),
                        ]
                    )
                    topk_indices_cmp, indexer_loss = FusedDSAIndexerLoss.apply(
                        q_thd,
                        weights_for_unfused,
                        k_thd,
                        query.detach(),
                        key_for_loss_thd.detach(),
                        self.softmax_scale,
                        min(self.indexer.index_topk, max_seqlen_compressed_idx),
                        indexer_loss_coeff,
                        None,
                        getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                        self.indexer.pg_collection,
                        self.config.calculate_per_token_loss,
                        cu_seqlens_q_for_loss,
                        cu_seqlens_compressed_idx_for_loss,
                        self.compress_ratio,
                    )

                    if indexer_loss_coeff > 0:
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                        )
                else:
                    _, topk_indices_cmp = self.indexer(
                        x_det, qr_det, mask=None, packed_seq_params=packed_seq_params
                    )

                # Shift into per-segment full-KV index space.
                if topk_indices_cmp.shape[-1] > 0:
                    seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                    batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
                    offset_per_row = seq_lens_kv[batch_of_token].unsqueeze(1)
                    # Per-segment causal post-filter — mirrors the SBHD
                    # ``_forward_unfused_csa`` post-filter. The training
                    # indexer (``fwd_fused_indexer_loss_naive_thd``)
                    # returns RAW per-segment top-K ids without sentinel
                    # for non-causal picks, so a query at intra-segment
                    # position ``i`` (0-indexed) may select compressed
                    # indices ``>= (i+1)//ratio`` whose pre-mask scores
                    # were ``-inf``; treat those as ``-1`` so the sparse
                    # attention skips them.
                    pos_in_seg = (
                        torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
                        - cu_seqlens_q[batch_of_token]
                    )
                    n_valid_per_row = ((pos_in_seg + 1) // self.compress_ratio).unsqueeze(1)
                    causal_valid = topk_indices_cmp < n_valid_per_row
                    is_valid = (topk_indices_cmp >= 0) & causal_valid
                    compress_topk_idxs = torch.where(
                        is_valid,
                        topk_indices_cmp + offset_per_row,
                        torch.full_like(topk_indices_cmp, -1),
                    )
                else:
                    compress_topk_idxs = topk_indices_cmp
            else:
                compress_topk_idxs = get_compress_topk_idxs_thd(
                    self.compress_ratio,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    cu_seqlens_compressed,
                    total_q=total_q,
                    max_n_compressed=max_seqlen_compressed_idx,
                )

            topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.int()

        flat_idxs, _ = build_flat_topk_idxs(
            topk_idxs, batch_size=-1, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv_full
        )

        output = unfused_compressed_sparse_attn(
            query, kv_full_thd, self.attn_sink.float(), flat_idxs, self.softmax_scale
        )
        return output.unsqueeze(1), indexer_loss

    def _forward_fused_attention_thd(
        self,
        query: torch.Tensor,
        kv_full_thd: torch.Tensor,
        flat_idxs: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        return_p_out: bool = False,
        attn_sink: Optional[torch.Tensor] = None,
        window_size: int = 0,
    ):
        """Run LoongForge fused THD sparse attention with TP/SP head exchange.

        THD packed input is already sequence-parallel when sequence_parallel is
        enabled, while query heads are tensor-parallel sharded. The sparse MLA
        kernel only supports global head counts (64/128), so mirror the SBHD
        path: all-to-all HP->SP before the kernel, slice the per-rank query rows
        and indices, then all-to-all SP->HP on the output.
        """
        tp_group = self.pg_collection.tp
        tp_size = get_pg_size(tp_group)
        if self.config.sequence_parallel and tp_size > 1:
            tp_rank = tp_group.rank()
            sq = query.size(0)
            chunk_query = _AllToAllHp2Sp.apply(query, tp_group)
            chunk_sq = chunk_query.size(0)
            chunk_start = tp_rank * chunk_sq
            chunk_end = chunk_start + chunk_sq
            chunk_idxs = flat_idxs[chunk_start:chunk_end]
            if chunk_idxs.size(0) < chunk_sq:
                chunk_idxs = F.pad(chunk_idxs, (0, 0, 0, chunk_sq - chunk_idxs.size(0)), value=-1)
            chunk_idxs = chunk_idxs.contiguous()
            chunk_sink = attn_sink
            if chunk_sink is not None and chunk_sink.numel() != chunk_query.size(1):
                chunk_sink = gather_from_tensor_model_parallel_region(chunk_sink, group=tp_group)

            output = self.sparse_attention(
                chunk_query,
                kv_full_thd,
                chunk_idxs.int(),
                kv_full_thd.size(0),
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=packed_seq_params,
                return_p_out=return_p_out,
                window_size=window_size,
                attn_sink=chunk_sink,
            )
            if return_p_out:
                output, p_out = output

            output = _AllToAllSp2Hp.apply(output, tp_group)
            output = output[:sq].contiguous()
            if return_p_out:
                p_out = p_out[:sq].contiguous()
            if output.ndim == 3:
                output = output.reshape(output.shape[0], -1)
            if return_p_out:
                return output, p_out, chunk_start, chunk_end, chunk_sq
            return output

        output = self.sparse_attention(
            query,
            kv_full_thd,
            flat_idxs.int(),
            kv_full_thd.size(0),  # disable FlashMLA causal mask (already encoded in indices)
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=packed_seq_params,
            return_p_out=return_p_out,
            window_size=window_size,
            attn_sink=attn_sink,
        )
        if return_p_out:
            output, p_out = output
        if output.ndim == 3:
            output = output.reshape(output.shape[0], -1)
        if return_p_out:
            return output, p_out, 0, output.shape[0], output.shape[0]
        return output

    def _forward_fused_no_indexer_thd(
        self,
        query: torch.Tensor,
        kv_full_thd: torch.Tensor,
        total_q: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        cu_seqlens_compressed: torch.Tensor,
        n_compressed_total: int,
        window_idxs: torch.Tensor,
        max_seqlen_compressed_idx: int = 0,
        packed_seq_params: "PackedSeqParams" = None,
    ) -> torch.Tensor:
        """Path A (THD): fused sparse attn with window or deterministic
        compressed indices.

        Returns ``(total_q, 1, np * hn)`` -- the attention output.
        """
        if self.compress_ratio > 1 and n_compressed_total > 0:
            compress_topk_idxs = get_compress_topk_idxs_thd(
                self.compress_ratio,
                cu_seqlens_q,
                cu_seqlens_kv,
                cu_seqlens_compressed,
                total_q=total_q,
                max_n_compressed=max_seqlen_compressed_idx,
            )
            flat_idxs, _ = build_flat_topk_idxs(
                window_idxs,
                compress_topk_idxs,
                batch_size=-1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )
        else:
            flat_idxs, _ = build_flat_topk_idxs(
                window_idxs,
                batch_size=-1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )

        if self.use_mcore_fused:
            # AIAK-Megatron fused path: requires flash_mla with flash_mla_sparse_fwd
            output = dsa_sparse_attn(
                query, kv_full_thd, self.attn_sink.float(), flat_idxs, self.softmax_scale, is_thd=True
            )
        else:
            # LoongForge fused path: HP->SP exchange makes the kernel see global heads.
            output, p_out, chunk_start, chunk_end, chunk_sq = self._forward_fused_attention_thd(
                query,
                kv_full_thd,
                flat_idxs,
                packed_seq_params,
                return_p_out=True,
                attn_sink=self.attn_sink.float(),
                window_size=window_idxs.shape[-1],
            )
        return output.unsqueeze(1)

    def _forward_fused_indexer_inference_thd(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_full_thd: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        total_q: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        window_idxs: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_compressed_idx: int,
        max_seqlen_kv: int,
    ) -> torch.Tensor:
        """Path C (THD): separate indexer forward (no loss) + fused sparse attn (compact).

        Returns ``(total_q, 1, np * hn)`` -- the attention output.
        """
        x_det = x.detach()
        qr_det = qr.detach()

        # Use loongforge CSAIndexer.forward() which uses CSAIndexerKernel internally
        # (avoids AIAK-Megatron indexer_topk that requires cudnn DSA)
        _, topk_indices_cmp = self.indexer(
            x_det, qr_det, mask=None, packed_seq_params=packed_seq_params
        )
        # topk_indices_cmp: (total_q, index_topk), int64 -> int32
        topk_indices_cmp = topk_indices_cmp.int()

        # Shift into per-segment full-KV index space with causal post-filter.
        # Mirrors the unfused path exactly: apply causal filter then add
        # the original-KV offset for kv_full indexing.
        if topk_indices_cmp.shape[-1] > 0:
            device = topk_indices_cmp.device
            seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
            offset_per_row = seq_lens_kv[batch_of_token].unsqueeze(1)
            # Per-segment causal post-filter — same as unfused path.
            # For early queries where (pos_in_seg+1)//ratio == 0,
            # no compressed key is causally valid; mark as -1.
            pos_in_seg = (
                torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
                - cu_seqlens_q[batch_of_token]
            )
            n_valid_per_row = ((pos_in_seg + 1) // self.compress_ratio).unsqueeze(1)
            causal_valid = topk_indices_cmp < n_valid_per_row
            is_valid = (topk_indices_cmp >= 0) & causal_valid
            compress_topk_idxs = torch.where(
                is_valid,
                topk_indices_cmp + offset_per_row,
                torch.full_like(topk_indices_cmp, -1),
            )
        else:
            compress_topk_idxs = topk_indices_cmp

        if self.use_mcore_fused:
            # AIAK-Megatron fused path: uses compact=True (requires cudnn DSA)
            flat_idxs, flat_tlen = build_flat_topk_idxs(
                window_idxs,
                compress_topk_idxs,
                batch_size=-1,
                compact=True,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )
            output = dsa_sparse_attn(
                query,
                kv_full_thd,
                self.attn_sink.float(),
                flat_idxs,
                self.softmax_scale,
                topk_length=flat_tlen,
                is_thd=True,
            )
        else:
            # LoongForge fused path: no compact (avoids cudnn DSA dependency).
            # HP->SP exchange keeps fused sparse attention on the global-head layout.
            flat_idxs, _ = build_flat_topk_idxs(
                window_idxs,
                compress_topk_idxs,
                batch_size=-1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )
            output = self._forward_fused_attention_thd(
                query,
                kv_full_thd,
                flat_idxs,
                packed_seq_params,
                return_p_out=False,
                attn_sink=self.attn_sink.float(),
                window_size=window_idxs.shape[-1],
            )
        return output.unsqueeze(1)

    def _forward_fused_indexer_training_thd(
        self,
        query: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        total_q: int,
        np_: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_seqlens_kv_full: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_compressed_idx: int,
        compressed_kv: torch.Tensor,
        kv_full_thd: torch.Tensor,
        window_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Path B (THD): fused indexer (with loss) + fused sparse attn.

        Returns ``(output, indexer_loss)`` where *output* is
        ``(total_q, 1, np * hn)``.
        """
        sparse_loss = getattr(self.config, "dsa_indexer_use_sparse_loss", True)
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

        x_det = x.detach()
        qr_det = qr.detach()
        q_indexer, k_indexer, weights_indexer, cu_seqlens_compressed_idx = (
            self.indexer.forward_before_topk(x_det, qr_det, packed_seq_params)
        )
        if k_indexer is None:
            raise RuntimeError(
                "CompressedSparseAttention THD Path B requires at least "
                "one segment with compressed indexer K; got none. (Should "
                "be unreachable when ``n_compressed_total > 0``.)"
            )

        q_thd = q_indexer.squeeze(1)
        w_thd = weights_indexer.squeeze(1)
        k_thd = k_indexer.squeeze(1)

        # Supply unpadded cu_seqlens so padding rows are excluded from
        # the indexer KL loss.
        cu_seqlens_q_unpadded = None
        if (
            packed_seq_params.cu_seqlens_q is not None
            and packed_seq_params.cu_seqlens_q_padded is not None
            and packed_seq_params.cu_seqlens_q.data_ptr()
            != packed_seq_params.cu_seqlens_q_padded.data_ptr()
        ):
            cu_seqlens_q_unpadded = packed_seq_params.cu_seqlens_q

        if self.use_mcore_fused:
            # AIAK-Megatron fused path: fused indexer + sparse attn
            output, indexer_loss = fused_indexer_sparse_attn(
                query,
                kv_full_thd,
                self.attn_sink.float(),
                window_idxs,
                q_thd,
                k_thd,
                w_thd,
                self.indexer.index_topk,
                self.compress_ratio,
                self.softmax_scale,
                self.indexer.softmax_scale,
                indexer_loss_coeff,
                sparse_loss=sparse_loss,
                kv_offset=0,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_kv_full=cu_seqlens_kv_full,
                cu_seqlens_compressed_idx=cu_seqlens_compressed_idx,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_compressed_idx=max_seqlen_compressed_idx,
                compressed_kv=compressed_kv,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
                cu_seqlens_q_unpadded=cu_seqlens_q_unpadded,
            )
        else:
            # Loongforge fused path: use CSAIndexer (CSAIndexerKernel) for topk
            index_scores_cmp, topk_indices_cmp = self.indexer(
                x_det, qr_det, mask=None, packed_seq_params=packed_seq_params
            )
            topk_indices_cmp = topk_indices_cmp.int()
            # Shift into per-segment full-KV index space with causal post-filter.
            if topk_indices_cmp.shape[-1] > 0:
                device = topk_indices_cmp.device
                seq_lens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
                batch_of_token = batch_of_row(cu_seqlens_q, total_q=total_q)
                offset_per_row = seq_lens_kv[batch_of_token].unsqueeze(1)
                # Per-segment causal post-filter — same as unfused path.
                pos_in_seg = (
                    torch.arange(total_q, device=device, dtype=cu_seqlens_q.dtype)
                    - cu_seqlens_q[batch_of_token]
                )
                n_valid_per_row = ((pos_in_seg + 1) // self.compress_ratio).unsqueeze(1)
                causal_valid = topk_indices_cmp < n_valid_per_row
                is_valid = (topk_indices_cmp >= 0) & causal_valid
                compress_topk_idxs = torch.where(
                    is_valid,
                    topk_indices_cmp + offset_per_row,
                    torch.full_like(topk_indices_cmp, -1),
                )
            else:
                compress_topk_idxs = topk_indices_cmp

            flat_idxs, _ = build_flat_topk_idxs(
                window_idxs,
                compress_topk_idxs,
                batch_size=-1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )
            output, p_out, chunk_start, chunk_end, chunk_sq = self._forward_fused_attention_thd(
                query,
                kv_full_thd,
                flat_idxs,
                packed_seq_params,
                return_p_out=True,
                attn_sink=self.attn_sink.float(),
                window_size=window_idxs.shape[-1],
            )

            # Compute indexer loss from LoongForge sparse attention probabilities.
            # This keeps the training fused path on sparse top-k tensors instead of
            # calling FusedDSAIndexerLoss, whose THD fallback forms dense QK scores.
            indexer_loss = None
            if indexer_loss_coeff > 0:
                from loongforge.models.common.experimental_attention_variant.dsa_fused_kernels import (
                    triton_attn_dist,
                )
                import triton

                with torch.no_grad():
                    p_out_topk = p_out.size(-1)
                    p_out_for_dist = p_out
                    next_pow2 = triton.next_power_of_2(p_out_topk)
                    if next_pow2 != p_out_topk:
                        p_out_for_dist = F.pad(p_out, (0, next_pow2 - p_out_topk), value=float('-inf'))
                    main_attn_probs = triton_attn_dist(p_out_for_dist, self.softmax_scale)
                    if next_pow2 != p_out_topk:
                        main_attn_probs = main_attn_probs[..., :p_out_topk]
                    if main_attn_probs.ndim == 3:
                        main_attn_probs = main_attn_probs.mean(dim=1)
                    chunk_row_valid = (
                        torch.arange(chunk_start, chunk_end, device=query.device)
                        - cu_seqlens_q[batch_of_token[chunk_start:chunk_end]]
                    ) >= self.compress_ratio
                    if chunk_sq > chunk_row_valid.size(0):
                        chunk_row_valid = F.pad(
                            chunk_row_valid, (0, chunk_sq - chunk_row_valid.size(0)), value=False
                        )
                    main_attn_probs = main_attn_probs[:, :p_out_topk]
                    main_attn_probs = torch.nan_to_num(main_attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
                    main_attn_probs = main_attn_probs.masked_fill(~chunk_row_valid.unsqueeze(-1), 0.0)

                effective_topk = min(index_scores_cmp.size(-1), p_out_topk)
                index_scores_topk = (
                    index_scores_cmp[chunk_start:chunk_end, :effective_topk]
                    * self.indexer.softmax_scale
                )
                index_scores_topk = torch.nan_to_num(
                    index_scores_topk,
                    nan=torch.finfo(index_scores_topk.dtype).min,
                    posinf=torch.finfo(index_scores_topk.dtype).max,
                    neginf=torch.finfo(index_scores_topk.dtype).min,
                )
                if chunk_sq > index_scores_topk.size(0):
                    index_scores_topk = F.pad(
                        index_scores_topk, (0, 0, 0, chunk_sq - index_scores_topk.size(0)), value=0.0
                    )

                compress_valid = compress_topk_idxs[chunk_start:chunk_end, :effective_topk] >= 0
                if chunk_sq > compress_valid.size(0):
                    compress_valid = F.pad(
                        compress_valid, (0, 0, 0, chunk_sq - compress_valid.size(0)), value=False
                    )   
                index_scores_topk = torch.where(
                    compress_valid, index_scores_topk, torch.finfo(index_scores_topk.dtype).min
                )

                index_attn_probs = torch.softmax(index_scores_topk, dim=-1)
                index_attn_probs = torch.nan_to_num(index_attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
                index_attn_probs = index_attn_probs.masked_fill(~chunk_row_valid.unsqueeze(-1), 0.0)
                main_attn_probs = main_attn_probs[:, :effective_topk]

                per_row_loss = (
                    main_attn_probs
                    * ((main_attn_probs + 1e-8).log() - (index_attn_probs + 1e-8).log())
                ).sum(dim=-1)
                per_row_loss = torch.nan_to_num(per_row_loss, nan=0.0, posinf=0.0, neginf=0.0)
                per_row_loss = per_row_loss.masked_fill(~chunk_row_valid, 0.0)
                if sparse_loss:
                    denom = chunk_row_valid.float().sum().clamp_min(1.0)
                    indexer_loss = per_row_loss.sum() / denom
                else:
                    indexer_loss = per_row_loss.sum() / per_row_loss.numel()
                indexer_loss = indexer_loss * indexer_loss_coeff

        if indexer_loss_coeff > 0:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
            )
        output = output.unsqueeze(1)
        return output, indexer_loss

