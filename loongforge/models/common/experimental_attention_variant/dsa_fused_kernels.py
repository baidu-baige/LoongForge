# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Kernels for Deepseek Sparse Attention."""

import dataclasses
from typing import Optional, Tuple, Any
from packaging.version import Version as PkgVersion

import torch
from torch import Tensor

_DSA_FUSED_DEPS_HINT = (
    "dsa_fused requires optional dependencies. "
    "Install them with: pip install -r requirements_dsa_fused.txt"
)

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc

try:
    import deep_gemm
    import flashinfer
    import lightning_indexer_bwd
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.fusions.fused_mla_yarn_rope_apply import _get_thd_token_idx
from megatron.core.fusions.fused_mla_yarn_rope_apply import ApplyMLARotaryEmbQ as ApplyMLARotaryEmbQNonInterleaved

try:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc

try:
    from flash_mla_fwd import flash_mla_sparse_fwd
    from flash_mla_bwd import flash_mla_sparse_bwd
except ImportError as exc:
    raise ImportError(_DSA_FUSED_DEPS_HINT) from exc
from .sparse_mla_bwd import sparse_mla_bwd_interface


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["Q"],
)
@triton.jit
def rotary_fwd_q_kernel_interleaved(
    Q,
    COS,
    SIN,
    emb_offset,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    sp_offset,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel for forward pass - Interleaved mode.
    
    Interleaved layout:
    - Pairs adjacent elements: (x[0], x[1]), (x[2], x[3]), ..., (x[n-2], x[n-1])
    - For each pair (x_even, x_odd):
        x_even_new = x_even * cos - x_odd * sin
        x_odd_new = x_even * sin + x_odd * cos
    - Store back in the same interleaved positions

    Args:
        emb_offset: starting offset of the emb_dim RoPE region within the head dimension.
            For nope-first layout [nope, pe]: emb_offset = qk_head_dim.
            For PE-first layout [pe, nope]: emb_offset = 0.
        sp_offset: global token offset for sequence parallelism (TP rank * local_seqlen).
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m + sp_offset, seq_num, cp_rank, cp_size)

    Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + emb_offset
    mask = x_off < head_num * stride_x_nheads

    # Extract even and odd indices: x[0::2] and x[1::2]
    # We need to apply rotation to pairs (x[2i], x[2i+1])
    x_even_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_odd_off = x_even_off + 1
    x_even = tl.load(Q + x_even_off, mask=mask)
    x_odd = tl.load(Q + x_odd_off, mask=mask)
    
    # Get corresponding cos/sin values for BOTH even and odd positions
    cos_even = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
    sin_even = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
    cos_odd = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)
    sin_odd = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)
    
    cos_even = cos_even.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_even = sin_even.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_odd = cos_odd.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_odd = sin_odd.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    
    # Apply rotation transformation
    # x_even_new = x_even * cos_even - x_odd * sin_even
    # x_odd_new = x_odd * cos_odd + x_even * sin_odd
    x_even_new = x_even * cos_even - x_odd * sin_even
    x_odd_new = x_odd * cos_odd + x_even * sin_odd
    
    # Store back in interleaved positions
    tl.store(Q + x_even_off, x_even_new, mask=mask)
    tl.store(Q + x_odd_off, x_odd_new, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["DO"],
)
@triton.jit
def rotary_bwd_q_kernel_interleaved(
    DO,
    COS,
    SIN,
    emb_offset,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    sp_offset,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel for backward pass - Interleaved mode.
    
    Backward rotation (inverse transformation):
    - x_even_grad = x_even_new_grad * cos + x_odd_new_grad * sin
    - x_odd_grad = -x_even_new_grad * sin + x_odd_new_grad * cos

    Args:
        emb_offset: starting offset of the emb_dim RoPE region within the head dimension.
        sp_offset: global token offset for sequence parallelism (TP rank * local_seqlen).
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m + sp_offset, seq_num, cp_rank, cp_size)

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + emb_offset
    mask = x_off < head_num * stride_x_nheads
    
    # Load gradient values at even and odd positions
    x_even_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_odd_off = x_even_off + 1
    x_even_grad = tl.load(DO + x_even_off, mask=mask)
    x_odd_grad = tl.load(DO + x_odd_off, mask=mask)
    
    # Get corresponding cos/sin values for BOTH even and odd positions
    cos_even = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
    sin_even = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
    cos_odd = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)
    sin_odd = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)
    
    cos_even = cos_even.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_even = sin_even.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_odd = cos_odd.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_odd = sin_odd.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    
    # Apply inverse rotation
    # Forward was: x_even_new = x_even * cos_even - x_odd * sin_even
    #              x_odd_new = x_odd * cos_odd + x_even * sin_odd
    # Backward: x_even_grad = x_even_new_grad * cos_even + x_odd_new_grad * sin_odd
    #           x_odd_grad = -x_even_new_grad * sin_even + x_odd_new_grad * cos_odd
    x_even_out = x_even_grad * cos_even + x_odd_grad * sin_odd
    x_odd_out = -x_even_grad * sin_even + x_odd_grad * cos_odd
    
    # Store back
    tl.store(DO + x_even_off, x_even_out, mask=mask)
    tl.store(DO + x_odd_off, x_odd_out, mask=mask)


class ApplyMLARotaryEmbQInterleaved(torch.autograd.Function):
    """
    Autograd function for applying interleaved YARN RoPE to MLA's query.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        cos,
        sin,
        qk_head_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        emb_offset=None,
        sp_offset=0,
    ):
        """
        Forward function for ApplyMLARotaryEmbQInterleaved.

        Args:
            q: [seq_len, batch_size, head_num, qk_head_dim + emb_dim]
                or [total_seq_len, head_num, qk_head_dim + emb_dim]
            cos/sin: [max_seq_len, 1, 1, emb_dim]
            cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
            emb_offset: starting offset of RoPE region. None defaults to qk_head_dim (nope-first).
            sp_offset: global token offset for sequence parallelism (TP rank * local_seqlen).
        """
        if emb_offset is None:
            emb_offset = qk_head_dim
        max_seqlen = None
        batch_size = None
        seq_num = None
        if cu_seqlens_q is None:
            # sbhd
            max_seqlen, batch_size, nheads, headdim = q.shape
            q = q.view(-1, nheads, headdim)
            total_seqlen = q.shape[0]
        else:
            # thd
            total_seqlen, nheads, headdim = q.shape
            seq_num = len(cu_seqlens_q) - 1
        assert q.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert headdim == qk_head_dim + emb_dim
        assert emb_dim % 4 == 0

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        rotary_fwd_q_kernel_interleaved[grid](
            q,
            cos,
            sin,
            emb_offset,
            emb_dim,
            nheads,
            batch_size,
            seq_num,
            cu_seqlens_q,
            q.stride(0),
            q.stride(1),
            cp_rank,
            cp_size,
            sp_offset,
        )
        ctx.save_for_backward(cos, sin)
        ctx.emb_offset = emb_offset
        ctx.emb_dim = emb_dim
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.sp_offset = sp_offset
        if cu_seqlens_q is None:
            q = q.view(max_seqlen, batch_size, nheads, headdim)
        return q

    @staticmethod
    def backward(ctx, grad):
        """
        Backward function for ApplyMLARotaryEmbQInterleaved.

        Args:
            grad: [seq_len, batch_size, head_num, qk_head_dim + emb_dim]
                or [total_seq_len, head_num, qk_head_dim + emb_dim]
        """
        cos, sin = ctx.saved_tensors
        max_seqlen = None
        batch_size = None
        seq_num = None
        if ctx.cu_seqlens_q is None:
            max_seqlen, batch_size, nheads, headdim = grad.shape
            grad = grad.contiguous().view(-1, nheads, headdim)
            total_seqlen = grad.shape[0]
        else:
            seq_num = len(ctx.cu_seqlens_q) - 1
            total_seqlen, nheads, headdim = grad.shape
        assert grad.stride(-1) == 1

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        rotary_bwd_q_kernel_interleaved[grid](
            grad,
            cos,
            sin,
            ctx.emb_offset,
            ctx.emb_dim,
            nheads,
            batch_size,
            seq_num,
            ctx.cu_seqlens_q,
            grad.stride(0),
            grad.stride(1),
            ctx.cp_rank,
            ctx.cp_size,
            ctx.sp_offset,
        )
        if ctx.cu_seqlens_q is None:
            grad = grad.view(max_seqlen, batch_size, nheads, headdim)
        return grad, None, None, None, None, None, None, None, None, None


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["Q"],
)
@triton.jit
def rotary_fwd_q_kernel_non_interleaved(
    Q,
    COS,
    SIN,
    emb_offset,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    sp_offset,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel for forward pass - Non-interleaved mode with configurable emb_offset.

    Same math as Loong-Megatron's rotary_fwd_q_kernel, but uses emb_offset instead of
    hardcoded qk_head_dim so it works for both nope-first and PE-first layouts.

    Args:
        emb_offset: starting offset of the emb_dim RoPE region within the head dimension.
        sp_offset: global token offset for sequence parallelism (TP rank * local_seqlen).
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m + sp_offset, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + emb_offset
    mask = x_off < head_num * stride_x_nheads
    # non-interleaved: x1 = t[..., :emb_dim//2], x2 = t[..., emb_dim//2:]
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_2_off = x_off + emb_dim // 2 + tl.arange(0, emb_dim // 2)[None, :]
    x_1 = tl.load(Q + x_1_off, mask=mask)
    x_2 = tl.load(Q + x_2_off, mask=mask)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right

    tl.store(Q + x_1_off, x_left, mask=mask)
    tl.store(Q + x_2_off, x_right, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "head_num"],
    restore_value=["DO"],
)
@triton.jit
def rotary_bwd_q_kernel_non_interleaved(
    DO,
    COS,
    SIN,
    emb_offset,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    sp_offset,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel for backward pass - Non-interleaved mode with configurable emb_offset.

    Args:
        emb_offset: starting offset of the emb_dim RoPE region within the head dimension.
        sp_offset: global token offset for sequence parallelism (TP rank * local_seqlen).
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m + sp_offset, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + emb_offset
    mask = x_off < head_num * stride_x_nheads
    # non-interleaved: read front half and back half
    x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
    x_2_off = x_off + emb_dim // 2 + tl.arange(0, emb_dim // 2)[None, :]
    x_left = tl.load(DO + x_1_off, mask=mask)
    x_right = tl.load(DO + x_2_off, mask=mask)

    x_1 = x_left * cos_left + x_right * sin_right
    x_2 = -x_left * sin_left + x_right * cos_right

    tl.store(DO + x_1_off, x_1, mask=mask)
    tl.store(DO + x_2_off, x_2, mask=mask)


class ApplyMLARotaryEmbQNonInterleavedWithOffset(torch.autograd.Function):
    """
    Autograd function for applying non-interleaved YARN RoPE to MLA's query,
    with configurable emb_offset for PE-first layouts.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        cos,
        sin,
        qk_head_dim,
        emb_dim,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        emb_offset=None,
        sp_offset=0,
    ):
        """Apply non-interleaved YARN RoPE to MLA query tensor in-place with configurable embedding offset."""
        if emb_offset is None:
            emb_offset = qk_head_dim
        max_seqlen = None
        batch_size = None
        seq_num = None
        if cu_seqlens_q is None:
            max_seqlen, batch_size, nheads, headdim = q.shape
            q = q.view(-1, nheads, headdim)
            total_seqlen = q.shape[0]
        else:
            total_seqlen, nheads, headdim = q.shape
            seq_num = len(cu_seqlens_q) - 1
        assert q.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert headdim == qk_head_dim + emb_dim
        assert emb_dim % 4 == 0

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        rotary_fwd_q_kernel_non_interleaved[grid](
            q,
            cos,
            sin,
            emb_offset,
            emb_dim,
            nheads,
            batch_size,
            seq_num,
            cu_seqlens_q,
            q.stride(0),
            q.stride(1),
            cp_rank,
            cp_size,
            sp_offset,
        )
        ctx.save_for_backward(cos, sin)
        ctx.emb_offset = emb_offset
        ctx.emb_dim = emb_dim
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.sp_offset = sp_offset
        if cu_seqlens_q is None:
            q = q.view(max_seqlen, batch_size, nheads, headdim)
        return q

    @staticmethod
    def backward(ctx, grad):
        """Compute backward pass by applying inverse non-interleaved YARN RoPE to the query gradient."""
        cos, sin = ctx.saved_tensors
        max_seqlen = None
        batch_size = None
        seq_num = None
        if ctx.cu_seqlens_q is None:
            max_seqlen, batch_size, nheads, headdim = grad.shape
            grad = grad.contiguous().view(-1, nheads, headdim)
            total_seqlen = grad.shape[0]
        else:
            seq_num = len(ctx.cu_seqlens_q) - 1
            total_seqlen, nheads, headdim = grad.shape
        assert grad.stride(-1) == 1

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        rotary_bwd_q_kernel_non_interleaved[grid](
            grad,
            cos,
            sin,
            ctx.emb_offset,
            ctx.emb_dim,
            nheads,
            batch_size,
            seq_num,
            ctx.cu_seqlens_q,
            grad.stride(0),
            grad.stride(1),
            ctx.cp_rank,
            ctx.cp_size,
            ctx.sp_offset,
        )
        if ctx.cu_seqlens_q is None:
            grad = grad.view(max_seqlen, batch_size, nheads, headdim)
        return grad, None, None, None, None, None, None, None, None, None

# Fused RoPE + Permute (HSD→SHD) + Cat kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["d_out", "emb_dim", "head_num"],
)
@triton.jit
def fused_rope_permute_cat_fwd_kernel_non_interleaved(
    Q_CONTENT,       # [h, S, d_out]  HSD (GroupedGEMM output)
    Q_POS_EMB,       # [S, h, d_pe]   SHD (split view, squeezed b)
    COS,             # [max_seq_len, emb_dim]
    SIN,             # [max_seq_len, emb_dim]
    OUTPUT,          # [S, h, d_total] SHD
    S,
    head_num: tl.constexpr,
    d_out: tl.constexpr,         # kv_lora_rank (e.g. 512)
    emb_dim: tl.constexpr,       # qk_pos_emb_head_dim (e.g. 64)
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_qc_h,                 # Q_CONTENT stride for head dim
    stride_qc_s,                 # Q_CONTENT stride for token dim
    stride_pe_s,                 # Q_POS_EMB stride for token dim
    stride_pe_h,                 # Q_POS_EMB stride for head dim
    stride_out_s,                # OUTPUT stride for token dim
    stride_out_h,                # OUTPUT stride for head dim
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    For each (token, head) element:
      1) Copy q_content from HSD layout to output's SHD layout (permute)
      2) Apply non-interleaved RoPE on q_pos_emb, write to output tail (RoPE + cat)
    """
    pid_m = tl.program_id(axis=0)       # token index
    pid_head = tl.program_id(axis=1)    # head block index

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    h_offs = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_offs < head_num

    d_total = d_out + emb_dim

    # Part 1: permute q_content (HSD → SHD)
    # Read  Q_CONTENT[h_offs, pid_m, d] and write to OUTPUT[pid_m, h_offs, d]
    for d_start in range(0, d_out, 64):
        d_offs = d_start + tl.arange(0, 64)
        d_mask = d_offs < d_out
        mask = h_mask[:, None] & d_mask[None, :]

        src = (h_offs[:, None] * stride_qc_h
               + pid_m * stride_qc_s
               + d_offs[None, :])
        vals = tl.load(Q_CONTENT + src, mask=mask)

        dst = (pid_m * stride_out_s
               + h_offs[:, None] * stride_out_h
               + d_offs[None, :])
        tl.store(OUTPUT + dst, vals, mask=mask)

    # Part 2: RoPE on q_pos_emb + cat
    half_emb: tl.constexpr = emb_dim // 2

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, half_emb))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, half_emb))
    cos_right = tl.load(COS + token_idx * emb_dim + half_emb + tl.arange(0, half_emb))
    sin_right = tl.load(SIN + token_idx * emb_dim + half_emb + tl.arange(0, half_emb))
    cos_left = cos_left[None, :]   # [1, half_emb]
    sin_left = sin_left[None, :]
    cos_right = cos_right[None, :]
    sin_right = sin_right[None, :]

    pe_base = pid_m * stride_pe_s + h_offs[:, None] * stride_pe_h  # [BLOCK_H, 1]

    # non-interleaved: x1 = x[..., 0::2], x2 = x[..., 1::2]
    x_1_off = pe_base + tl.arange(0, half_emb)[None, :] * 2
    x_2_off = x_1_off + 1
    x_1 = tl.load(Q_POS_EMB + x_1_off, mask=h_mask[:, None])
    x_2 = tl.load(Q_POS_EMB + x_2_off, mask=h_mask[:, None])

    out_left = x_1 * cos_left - x_2 * sin_left
    out_right = x_2 * cos_right + x_1 * sin_right

    # Write to OUTPUT[pid_m, h_offs, d_out : d_out+emb_dim]
    out_base = pid_m * stride_out_s + h_offs[:, None] * stride_out_h + d_out
    left_off = out_base + tl.arange(0, half_emb)[None, :]
    right_off = out_base + half_emb + tl.arange(0, half_emb)[None, :]
    tl.store(OUTPUT + left_off, out_left, mask=h_mask[:, None])
    tl.store(OUTPUT + right_off, out_right, mask=h_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["d_out", "emb_dim", "head_num"],
)
@triton.jit
def fused_rope_permute_cat_fwd_kernel_interleaved(
    Q_CONTENT,
    Q_POS_EMB,
    COS,
    SIN,
    OUTPUT,
    S,
    head_num: tl.constexpr,
    d_out: tl.constexpr,
    emb_dim: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_qc_h,
    stride_qc_s,
    stride_pe_s,
    stride_pe_h,
    stride_out_s,
    stride_out_h,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """Interleaved RoPE variant of fused_rope_permute_cat."""
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    h_offs = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < head_num

    d_total = d_out + emb_dim

    # Part 1: permute q_content (HSD → SHD)
    for d_start in range(0, d_out, 64):
        d_offs = d_start + tl.arange(0, 64)
        d_mask = d_offs < d_out
        mask = h_mask[:, None] & d_mask[None, :]

        src = (h_offs[:, None] * stride_qc_h
               + pid_m * stride_qc_s
               + d_offs[None, :])
        vals = tl.load(Q_CONTENT + src, mask=mask)

        dst = (pid_m * stride_out_s
               + h_offs[:, None] * stride_out_h
               + d_offs[None, :])
        tl.store(OUTPUT + dst, vals, mask=mask)

    # Part 2: interleaved RoPE on q_pos_emb + cat
    half_emb: tl.constexpr = emb_dim // 2

    cos_even = tl.load(COS + token_idx * emb_dim + tl.arange(0, half_emb) * 2)
    sin_even = tl.load(SIN + token_idx * emb_dim + tl.arange(0, half_emb) * 2)
    cos_odd = tl.load(COS + token_idx * emb_dim + tl.arange(0, half_emb) * 2 + 1)
    sin_odd = tl.load(SIN + token_idx * emb_dim + tl.arange(0, half_emb) * 2 + 1)
    cos_even = cos_even[None, :]
    sin_even = sin_even[None, :]
    cos_odd = cos_odd[None, :]
    sin_odd = sin_odd[None, :]

    pe_base = pid_m * stride_pe_s + h_offs[:, None] * stride_pe_h

    x_even_off = pe_base + tl.arange(0, half_emb)[None, :] * 2
    x_odd_off = x_even_off + 1
    x_even = tl.load(Q_POS_EMB + x_even_off, mask=h_mask[:, None])
    x_odd = tl.load(Q_POS_EMB + x_odd_off, mask=h_mask[:, None])

    x_even_new = x_even * cos_even - x_odd * sin_even
    x_odd_new = x_odd * cos_odd + x_even * sin_odd

    out_base = pid_m * stride_out_s + h_offs[:, None] * stride_out_h + d_out
    tl.store(OUTPUT + out_base + tl.arange(0, half_emb)[None, :] * 2,
             x_even_new, mask=h_mask[:, None])
    tl.store(OUTPUT + out_base + tl.arange(0, half_emb)[None, :] * 2 + 1,
             x_odd_new, mask=h_mask[:, None])


# ---- Backward kernels ----


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["d_out", "emb_dim", "head_num"],
)
@triton.jit
def fused_rope_permute_cat_bwd_kernel_non_interleaved(
    GRAD_OUTPUT,         # [S, h, d_total] SHD
    GRAD_Q_CONTENT,      # [h, S, d_out]   HSD
    GRAD_Q_POS_EMB,      # [S, h, d_pe]    SHD
    COS,
    SIN,
    S,
    head_num: tl.constexpr,
    d_out: tl.constexpr,
    emb_dim: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_go_s,
    stride_go_h,
    stride_gc_h,
    stride_gc_s,
    stride_gpe_s,
    stride_gpe_h,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """Backward: split grad_output → inverse-permute grad_q_content + inverse-RoPE grad_q_pos_emb."""
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    h_offs = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < head_num

    d_total = d_out + emb_dim

    # Part 1: grad_q_content: SHD → HSD (inverse permute)
    for d_start in range(0, d_out, 64):
        d_offs = d_start + tl.arange(0, 64)
        d_mask = d_offs < d_out
        mask = h_mask[:, None] & d_mask[None, :]

        src = (pid_m * stride_go_s
               + h_offs[:, None] * stride_go_h
               + d_offs[None, :])
        vals = tl.load(GRAD_OUTPUT + src, mask=mask)

        dst = (h_offs[:, None] * stride_gc_h
               + pid_m * stride_gc_s
               + d_offs[None, :])
        tl.store(GRAD_Q_CONTENT + dst, vals, mask=mask)

    # Part 2: inverse RoPE on grad for pos_emb region
    half_emb: tl.constexpr = emb_dim // 2

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, half_emb))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, half_emb))
    cos_right = tl.load(COS + token_idx * emb_dim + half_emb + tl.arange(0, half_emb))
    sin_right = tl.load(SIN + token_idx * emb_dim + half_emb + tl.arange(0, half_emb))
    cos_left = cos_left[None, :]
    sin_left = sin_left[None, :]
    cos_right = cos_right[None, :]
    sin_right = sin_right[None, :]

    # Read grad from output's pos_emb region (already in non-interleaved RoPE output layout)
    go_base = pid_m * stride_go_s + h_offs[:, None] * stride_go_h + d_out
    left_off = go_base + tl.arange(0, half_emb)[None, :]
    right_off = go_base + half_emb + tl.arange(0, half_emb)[None, :]
    g_left = tl.load(GRAD_OUTPUT + left_off, mask=h_mask[:, None])
    g_right = tl.load(GRAD_OUTPUT + right_off, mask=h_mask[:, None])

    # Inverse RoPE (non-interleaved):
    #   fwd: out_left  = x1 * cos_left  - x2 * sin_left
    #        out_right = x2 * cos_right + x1 * sin_right
    #   bwd: dx1 = g_left * cos_left  + g_right * sin_right
    #        dx2 = -g_left * sin_left + g_right * cos_right
    dx_1 = g_left * cos_left + g_right * sin_right
    dx_2 = -g_left * sin_left + g_right * cos_right

    # Write grad_q_pos_emb in original interleaved-storage layout: x[..., 0::2], x[..., 1::2]
    gpe_base = pid_m * stride_gpe_s + h_offs[:, None] * stride_gpe_h
    gpe_1_off = gpe_base + tl.arange(0, half_emb)[None, :] * 2
    gpe_2_off = gpe_1_off + 1
    tl.store(GRAD_Q_POS_EMB + gpe_1_off, dx_1, mask=h_mask[:, None])
    tl.store(GRAD_Q_POS_EMB + gpe_2_off, dx_2, mask=h_mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["d_out", "emb_dim", "head_num"],
)
@triton.jit
def fused_rope_permute_cat_bwd_kernel_interleaved(
    GRAD_OUTPUT,
    GRAD_Q_CONTENT,
    GRAD_Q_POS_EMB,
    COS,
    SIN,
    S,
    head_num: tl.constexpr,
    d_out: tl.constexpr,
    emb_dim: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_go_s,
    stride_go_h,
    stride_gc_h,
    stride_gc_s,
    stride_gpe_s,
    stride_gpe_h,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """Backward: interleaved RoPE variant."""
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_q is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)

    h_offs = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < head_num

    d_total = d_out + emb_dim

    # Part 1: inverse permute (SHD → HSD)
    for d_start in range(0, d_out, 64):
        d_offs = d_start + tl.arange(0, 64)
        d_mask = d_offs < d_out
        mask = h_mask[:, None] & d_mask[None, :]

        src = (pid_m * stride_go_s
               + h_offs[:, None] * stride_go_h
               + d_offs[None, :])
        vals = tl.load(GRAD_OUTPUT + src, mask=mask)

        dst = (h_offs[:, None] * stride_gc_h
               + pid_m * stride_gc_s
               + d_offs[None, :])
        tl.store(GRAD_Q_CONTENT + dst, vals, mask=mask)

    # Part 2: inverse interleaved RoPE
    half_emb: tl.constexpr = emb_dim // 2

    cos_even = tl.load(COS + token_idx * emb_dim + tl.arange(0, half_emb) * 2)
    sin_even = tl.load(SIN + token_idx * emb_dim + tl.arange(0, half_emb) * 2)
    cos_odd = tl.load(COS + token_idx * emb_dim + tl.arange(0, half_emb) * 2 + 1)
    sin_odd = tl.load(SIN + token_idx * emb_dim + tl.arange(0, half_emb) * 2 + 1)
    cos_even = cos_even[None, :]
    sin_even = sin_even[None, :]
    cos_odd = cos_odd[None, :]
    sin_odd = sin_odd[None, :]

    go_base = pid_m * stride_go_s + h_offs[:, None] * stride_go_h + d_out
    g_even = tl.load(GRAD_OUTPUT + go_base + tl.arange(0, half_emb)[None, :] * 2,
                     mask=h_mask[:, None])
    g_odd = tl.load(GRAD_OUTPUT + go_base + tl.arange(0, half_emb)[None, :] * 2 + 1,
                    mask=h_mask[:, None])

    # Inverse interleaved RoPE:
    #   fwd: x_even_new = x_even * cos_even - x_odd * sin_even
    #        x_odd_new  = x_odd  * cos_odd  + x_even * sin_odd
    #   bwd: dx_even = g_even * cos_even + g_odd * sin_odd
    #        dx_odd  = -g_even * sin_even + g_odd * cos_odd
    dx_even = g_even * cos_even + g_odd * sin_odd
    dx_odd = -g_even * sin_even + g_odd * cos_odd

    gpe_base = pid_m * stride_gpe_s + h_offs[:, None] * stride_gpe_h
    tl.store(GRAD_Q_POS_EMB + gpe_base + tl.arange(0, half_emb)[None, :] * 2,
             dx_even, mask=h_mask[:, None])
    tl.store(GRAD_Q_POS_EMB + gpe_base + tl.arange(0, half_emb)[None, :] * 2 + 1,
             dx_odd, mask=h_mask[:, None])


class FusedRopePermuteCat(torch.autograd.Function):
    """
    Fused: permute q_content (HSD→SBHD) + RoPE on q_pos_emb + cat.

    Forward:
        q_content  [h, S, d_out]  (HSD, from GroupedGEMM)
        q_pos_emb  [s, b, h, emb_dim]  (SBHD, raw, no RoPE applied yet)
        cos/sin    [max_seq_len, 1, 1, emb_dim]
        →  query   [s, b, h, d_out + emb_dim]

    Backward:
        grad_query  [s, b, h, d_out + emb_dim]
        →  grad_q_content  [h, S, d_out]   (HSD, feeds GroupedGEMM backward)
           grad_q_pos_emb  [s, b, h, emb_dim]  (SBHD, with inverse RoPE)
    """

    @staticmethod
    def forward(
        ctx,
        q_content,
        q_pos_emb,
        cos,
        sin,
        cu_seqlens_q,
        cp_rank,
        cp_size,
        rotary_interleaved=False,
    ):
        """Fuse HSD-to-SBHD permutation of q_content, RoPE on q_pos_emb, and concatenation into a single kernel pass."""
        if q_pos_emb.ndim == 4:
            s, b, nheads, emb_dim = q_pos_emb.shape
            S = s * b
        else:
            # thd: [total_seq_len, h, emb_dim]
            S, nheads, emb_dim = q_pos_emb.shape
            s, b = S, 1

        d_out = q_content.shape[-1]
        d_total = d_out + emb_dim

        # Save original shape so backward can restore it
        q_content_shape = q_content.shape

        # Flatten to 3D for uniform kernel interface
        qc_3d = q_content.reshape(nheads, S, d_out)
        pe_3d = q_pos_emb.reshape(S, nheads, emb_dim)
        cos_2d = cos.reshape(-1, emb_dim)
        sin_2d = sin.reshape(-1, emb_dim)

        output = q_content.new_empty(S, nheads, d_total)

        batch_size = b if cu_seqlens_q is None else None
        seq_num = (len(cu_seqlens_q) - 1) if cu_seqlens_q is not None else None

        grid = lambda META: (S, triton.cdiv(nheads, META["BLOCK_H"]))

        kernel = (fused_rope_permute_cat_fwd_kernel_interleaved
                  if rotary_interleaved
                  else fused_rope_permute_cat_fwd_kernel_non_interleaved)
        kernel[grid](
            qc_3d, pe_3d, cos_2d, sin_2d, output,
            S, nheads, d_out, emb_dim,
            batch_size, seq_num, cu_seqlens_q,
            qc_3d.stride(0), qc_3d.stride(1),
            pe_3d.stride(0), pe_3d.stride(1),
            output.stride(0), output.stride(1),
            cp_rank, cp_size,
        )

        ctx.save_for_backward(cos_2d, sin_2d)
        ctx.s, ctx.b, ctx.nheads = s, b, nheads
        ctx.d_out, ctx.emb_dim = d_out, emb_dim
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cp_rank, ctx.cp_size = cp_rank, cp_size
        ctx.rotary_interleaved = rotary_interleaved
        ctx.q_content_shape = q_content_shape
        ctx.q_pos_emb_ndim = q_pos_emb.ndim

        if q_pos_emb.ndim == 4:
            return output.view(s, b, nheads, d_total)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Split grad into q_content (SBHD-to-HSD) and q_pos_emb with inverse RoPE."""
        cos_2d, sin_2d = ctx.saved_tensors
        s, b, nheads = ctx.s, ctx.b, ctx.nheads
        d_out, emb_dim = ctx.d_out, ctx.emb_dim
        S = s * b

        grad_3d = grad_output.reshape(S, nheads, d_out + emb_dim).contiguous()

        grad_q_content = grad_output.new_empty(nheads, S, d_out)
        grad_q_pos_emb = grad_output.new_empty(S, nheads, emb_dim)

        batch_size = b if ctx.cu_seqlens_q is None else None
        seq_num = (len(ctx.cu_seqlens_q) - 1) if ctx.cu_seqlens_q is not None else None

        grid = lambda META: (S, triton.cdiv(nheads, META["BLOCK_H"]))

        kernel = (fused_rope_permute_cat_bwd_kernel_interleaved
                  if ctx.rotary_interleaved
                  else fused_rope_permute_cat_bwd_kernel_non_interleaved)
        kernel[grid](
            grad_3d, grad_q_content, grad_q_pos_emb,
            cos_2d, sin_2d,
            S, nheads, d_out, emb_dim,
            batch_size, seq_num, ctx.cu_seqlens_q,
            grad_3d.stride(0), grad_3d.stride(1),
            grad_q_content.stride(0), grad_q_content.stride(1),
            grad_q_pos_emb.stride(0), grad_q_pos_emb.stride(1),
            ctx.cp_rank, ctx.cp_size,
        )

        # Reshape grad_q_content to match forward input's original shape
        grad_q_content = grad_q_content.view(ctx.q_content_shape)

        if ctx.q_pos_emb_ndim == 4:
            grad_q_pos_emb = grad_q_pos_emb.view(s, b, nheads, emb_dim)

        # Returns: grad for (q_content, q_pos_emb, cos, sin, cu_seqlens_q, cp_rank, cp_size, rotary_interleaved)
        return grad_q_content, grad_q_pos_emb, None, None, None, None, None, None


def fused_rope_permute_cat(
    q_content: torch.Tensor,
    q_pos_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens_q=None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """
    Fused operation: permute q_content (HSD→SBHD) + RoPE on q_pos_emb + cat.

    This replaces the sequence:
        q_content = q_content.permute(1,2,0,3).contiguous()
        q_pos_emb = apply_rope(q_pos_emb)
        query = torch.cat([q_content, q_pos_emb], dim=-1)

    Args:
        q_content:  [h, s*b, d_out]  HSD layout from GroupedGEMM output
        q_pos_emb:  [s, b, h, emb_dim] or [total_seq_len, h, emb_dim]  raw (no RoPE yet)
        cos/sin:    [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_q: optional packed-sequence cumulative lengths
        rotary_interleaved: whether to use interleaved RoPE

    Returns:
        query: [s, b, h, d_out + emb_dim] or [total_seq_len, h, d_out + emb_dim]
    """
    return FusedRopePermuteCat.apply(
        q_content, q_pos_emb, cos, sin,
        cu_seqlens_q, cp_rank, cp_size, rotary_interleaved,
    )


def fused_apply_mla_rope(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qk_head_dim: int,
    emb_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
    pe_first: bool = False,
    sp_offset: int = 0,
):
    """
    Fused function for applying YARN RoPE to MLA's query.
    This function inplace modifies the input tensor t.
    It is an experimental feature and may change in future versions.
    It supports both sbhd and thd input formats.

    For the notations below, seq_len is the length of the sequence per batch for sbhd format,
    total_seq_len is the total length of the sequences for thd format.
    max_seq_len is the maximum length of the sequences in the input tensor.

    Args:
        t: [seq_len, batch_size, head_num, qk_head_dim + emb_dim]
            or [total_seq_len, head_num, qk_head_dim + emb_dim]
        cos/sin: [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_q: [seq_num + 1] accumulated sequence lengths for thd format
        rotary_interleaved: whether to apply RoPE interleaved (True) or non-interleaved (False)
        pe_first: if True, layout is [pe, nope] (emb_offset=0);
                  if False, layout is [nope, pe] (emb_offset=qk_head_dim). Default False.
        sp_offset: global token offset for sequence parallelism (TP rank * local_seqlen).

    Returns:
        t: inplace modified input tensor
    """
    emb_offset = 0 if pe_first else qk_head_dim
    if rotary_interleaved:
        return ApplyMLARotaryEmbQInterleaved.apply(
            t, cos, sin, qk_head_dim, emb_dim, cu_seqlens_q, cp_rank, cp_size, emb_offset, sp_offset
        )
    else:
        return ApplyMLARotaryEmbQNonInterleavedWithOffset.apply(
            t, cos, sin, qk_head_dim, emb_dim, cu_seqlens_q, cp_rank, cp_size, emb_offset, sp_offset
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "k_dim", "v_dim", "head_num"],
)
@triton.jit
def rotary_fwd_absorb_kv_kernel_non_interleaved(
    KV,
    K_POS_EMB,
    O_KEY,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_kv_seq,
    stride_kv_nheads,
    stride_emb_seq,
    stride_k_seq,
    stride_k_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the forward pass for applying YARN RoPE to MLA's key and value.
    It splits the input tensor KV into key and value,
    and concatenates the processed RoPE to the key.

    Input:
        KV: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        K_POS_EMB: [seq_len, batch_size, emb_dim] or [total_seq_len, emb_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size: batch size for sbhd format, not used for thd format
        seq_num: number of sequences for thd format, not used for sbhd format
        cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format

    Output:
        O_KEY: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        O_VALUE: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
    cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
    sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

    KV_ptr = KV + pid_m * stride_kv_seq + pid_head * BLOCK_H * stride_kv_nheads
    kv_off = tl.arange(0, BLOCK_H)[:, None] * stride_kv_nheads
    mask = kv_off < head_num * stride_kv_nheads
    k_in_off = kv_off + tl.arange(0, k_dim)[None, :]
    k = tl.load(KV_ptr + k_in_off, mask=mask)

    K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads

    k_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads + tl.arange(0, k_dim)[None, :]
    tl.store(K_ptr + k_out_off, k, mask=mask)

    EMB = K_POS_EMB + pid_m * stride_emb_seq
    # x1 = t[..., 0::2], x2 = t[..., 1::2]
    x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
    x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)

    x_left = x_1 * cos_left - x_2 * sin_left
    x_right = x_2 * cos_right + x_1 * sin_right
    x_left = x_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    x_right = x_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)

    x_left_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
        + k_dim
        + tl.arange(0, emb_dim // 2)[None, :]
    )
    x_right_off = x_left_off + emb_dim // 2
    tl.store(K_ptr + x_left_off, x_left, mask=mask)
    tl.store(K_ptr + x_right_off, x_right, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "k_dim", "v_dim", "head_num"],
)
@triton.jit
def rotary_bwd_absorb_kv_kernel_non_interleaved(
    dK,
    dKV,
    dEMB,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_dk_seq,
    stride_dk_nheads,
    stride_dkv_seq,
    stride_dkv_nheads,
    stride_demb_seq,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel of the backward pass for applying YARN RoPE to MLA's key and value.

    Input:
        dK: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        COS/SIN: [max_seq_len, emb_dim]

        batch_size, seq_num, and cu_seqlens_kv are the same as in the forward pass

    Output:
        dKV: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        dEMB: [seq_len, batch_size, emb_dim] or [total_seq_len, emb_dim]
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    dKV_ptr = dKV + pid_m * stride_dkv_seq + pid_head * BLOCK_H * stride_dkv_nheads
    dkv_off = tl.arange(0, BLOCK_H)[:, None] * stride_dkv_nheads
    mask = dkv_off < head_num * stride_dkv_nheads
    dk_out_off = dkv_off + tl.arange(0, k_dim)[None, :]

    dK_ptr = dK + pid_m * stride_dk_seq + pid_head * BLOCK_H * stride_dk_nheads
    dk_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + tl.arange(0, k_dim)[None, :]
    dk = tl.load(dK_ptr + dk_in_off, mask=mask)
    tl.store(dKV_ptr + dk_out_off, dk, mask=mask)

    if pid_head == 0:
        x_left_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        x_right_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
            dK_ptr = dK + pid_m * stride_dk_seq + i * BLOCK_H * stride_dk_nheads
            x_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + k_dim
            mask = x_off < head_num * stride_dk_nheads
            x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
            x_right_off = x_left_off + emb_dim // 2
            x_left = tl.load(dK_ptr + x_left_off, mask=mask)
            x_right = tl.load(dK_ptr + x_right_off, mask=mask)
            x_left_accum += x_left
            x_right_accum += x_right
        x_left_accum = tl.sum(x_left_accum, axis=0)
        x_right_accum = tl.sum(x_right_accum, axis=0)
        x_left_accum = x_left_accum.to(dEMB.dtype.element_ty)
        x_right_accum = x_right_accum.to(dEMB.dtype.element_ty)

        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))

        x_1 = x_left_accum * cos_left + x_right_accum * sin_right
        x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
        dEMB_ptr = dEMB + pid_m * stride_demb_seq
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2, x_1)
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2 + 1, x_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "k_dim", "v_dim", "head_num"],
)
@triton.jit
def rotary_fwd_absorb_kv_kernel_interleaved(
    KV,
    K_POS_EMB,
    O_KEY,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_kv_seq,
    stride_kv_nheads,
    stride_emb_seq,
    stride_k_seq,
    stride_k_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel for forward pass - Interleaved mode.
    
    Interleaved layout:
    - Pairs adjacent elements in K_POS_EMB: (x[0], x[1]), (x[2], x[3]), ...
    - For each pair (x_even, x_odd):
        x_even_new = x_even * cos_even - x_odd * sin_even
        x_odd_new = x_odd * cos_odd + x_even * sin_odd
    - Store back in the same interleaved positions
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    # Copy k_dim part from KV to O_KEY (unchanged)
    KV_ptr = KV + pid_m * stride_kv_seq + pid_head * BLOCK_H * stride_kv_nheads
    kv_off = tl.arange(0, BLOCK_H)[:, None] * stride_kv_nheads
    mask = kv_off < head_num * stride_kv_nheads
    k_in_off = kv_off + tl.arange(0, k_dim)[None, :]
    k = tl.load(KV_ptr + k_in_off, mask=mask)

    K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads
    k_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads + tl.arange(0, k_dim)[None, :]
    tl.store(K_ptr + k_out_off, k, mask=mask)

    # Load K_POS_EMB and apply interleaved RoPE
    EMB = K_POS_EMB + pid_m * stride_emb_seq
    
    # Extract even and odd indices from K_POS_EMB
    x_even = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
    x_odd = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)
    
    # Load cos/sin for BOTH even and odd positions
    cos_even = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
    sin_even = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
    cos_odd = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)
    sin_odd = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)
    
    # Apply rotation transformation
    x_even_new = x_even * cos_even - x_odd * sin_even
    x_odd_new = x_odd * cos_odd + x_even * sin_odd
    
    # Broadcast to BLOCK_H heads and store in interleaved positions
    x_even_new = x_even_new.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    x_odd_new = x_odd_new.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
    
    # Store back in interleaved layout
    x_even_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads + k_dim + tl.arange(0, emb_dim // 2)[None, :] * 2
    x_odd_off = x_even_off + 1
    tl.store(K_ptr + x_even_off, x_even_new, mask=mask)
    tl.store(K_ptr + x_odd_off, x_odd_new, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1}),
        triton.Config({"BLOCK_H": 2}),
        triton.Config({"BLOCK_H": 4}),
        triton.Config({"BLOCK_H": 8}),
        triton.Config({"BLOCK_H": 16}),
        triton.Config({"BLOCK_H": 32}),
        triton.Config({"BLOCK_H": 64}),
        triton.Config({"BLOCK_H": 128}),
    ],
    key=["emb_dim", "k_dim", "v_dim", "head_num"],
)
@triton.jit
def rotary_bwd_absorb_kv_kernel_interleaved(
    dK,
    dKV,
    dEMB,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_dk_seq,
    stride_dk_nheads,
    stride_dkv_seq,
    stride_dkv_nheads,
    stride_demb_seq,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
    """
    Triton kernel for backward pass - Interleaved mode.
    
    Accumulates gradients from all heads and applies inverse rotation
    in interleaved layout.
    """
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)

    if cu_seqlens_kv is None:
        token_idx = pid_m // batch_size
    else:
        token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)

    # Copy k_dim gradients from dK to dKV (unchanged)
    dKV_ptr = dKV + pid_m * stride_dkv_seq + pid_head * BLOCK_H * stride_dkv_nheads
    dkv_off = tl.arange(0, BLOCK_H)[:, None] * stride_dkv_nheads
    mask = dkv_off < head_num * stride_dkv_nheads
    dk_out_off = dkv_off + tl.arange(0, k_dim)[None, :]

    dK_ptr = dK + pid_m * stride_dk_seq + pid_head * BLOCK_H * stride_dk_nheads
    dk_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + tl.arange(0, k_dim)[None, :]
    dk = tl.load(dK_ptr + dk_in_off, mask=mask)
    tl.store(dKV_ptr + dk_out_off, dk, mask=mask)

    # Accumulate gradients for emb_dim (only in first head group)
    if pid_head == 0:
        x_even_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        x_odd_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
        
        # Accumulate from all heads
        for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
            dK_ptr = dK + pid_m * stride_dk_seq + i * BLOCK_H * stride_dk_nheads
            x_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + k_dim
            mask = x_off < head_num * stride_dk_nheads
            
            # Load gradients from interleaved positions
            x_even_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
            x_odd_off = x_even_off + 1
            x_even_grad = tl.load(dK_ptr + x_even_off, mask=mask)
            x_odd_grad = tl.load(dK_ptr + x_odd_off, mask=mask)
            
            x_even_accum += x_even_grad
            x_odd_accum += x_odd_grad
        
        # Sum across BLOCK_H dimension
        x_even_accum = tl.sum(x_even_accum, axis=0)
        x_odd_accum = tl.sum(x_odd_accum, axis=0)
        x_even_accum = x_even_accum.to(dEMB.dtype.element_ty)
        x_odd_accum = x_odd_accum.to(dEMB.dtype.element_ty)

        # Load cos/sin for both even and odd positions
        cos_even = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
        sin_even = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2)
        cos_odd = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)
        sin_odd = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2) * 2 + 1)

        # Apply inverse rotation
        # Forward was: x_even_new = x_even * cos_even - x_odd * sin_even
        #              x_odd_new = x_odd * cos_odd + x_even * sin_odd
        # Backward: x_even_grad = x_even_new_grad * cos_even + x_odd_new_grad * sin_odd
        #           x_odd_grad = -x_even_new_grad * sin_even + x_odd_new_grad * cos_odd
        x_even_out = x_even_accum * cos_even + x_odd_accum * sin_odd
        x_odd_out = -x_even_accum * sin_even + x_odd_accum * cos_odd
        
        # Store back in interleaved positions
        dEMB_ptr = dEMB + pid_m * stride_demb_seq
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2, x_even_out)
        tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2 + 1, x_odd_out)


class ApplyMLARotaryEmbAbsorbKV(torch.autograd.Function):
    """
    Autograd function for applying YARN RoPE to MLA's key and value.
    """

    @staticmethod
    def forward(
        ctx,
        kv,
        k_pos_emb,
        cos,
        sin,
        emb_dim,
        k_dim,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
        rotary_interleaved=False,
    ):
        """
        Forward function for ApplyMLARotaryEmbAbsorbKV.

        Args:
            kv: [seq_len, batch_size, head_num, k_dim + v_dim]
                or [total_seq_len, head_num, k_dim + v_dim]
            k_pos_emb: [seq_len, batch_size, 1, emb_dim] or [total_seq_len, 1, emb_dim]
            cos/sin: [max_seq_len, 1, 1, emb_dim]
            cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format
            rotary_interleaved: whether to apply RoPE interleaved (True) or non-interleaved (False)
        """
        max_seqlen = None
        batch_size = None
        seq_num = None
        if cu_seqlens_kv is None:
            # sbhd
            max_seqlen, batch_size, nheads, headdim = kv.shape
            kv = kv.view(-1, nheads, headdim)
            k_pos_emb = k_pos_emb.view(-1, emb_dim)
            total_seqlen = kv.shape[0]
        else:
            # thd
            seq_num = len(cu_seqlens_kv) - 1
            total_seqlen, nheads, headdim = kv.shape
        assert headdim == k_dim
        assert kv.stride(-1) == 1
        assert k_pos_emb.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert emb_dim % 4 == 0

        o_key = kv.new_empty(total_seqlen, nheads, emb_dim + k_dim)

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        
        # Choose kernel based on mode
        if rotary_interleaved:
            rotary_fwd_absorb_kv_kernel_interleaved[grid](
                kv,
                k_pos_emb,
                o_key,
                cos,
                sin,
                emb_dim,
                k_dim,
                nheads,
                batch_size,
                seq_num,
                cu_seqlens_kv,
                kv.stride(0),
                kv.stride(1),
                k_pos_emb.stride(0),
                o_key.stride(0),
                o_key.stride(1),
                cp_rank,
                cp_size,
            )
        else:
            rotary_fwd_absorb_kv_kernel_non_interleaved[grid](
                kv,
                k_pos_emb,
                o_key,
                cos,
                sin,
                emb_dim,
                k_dim,
                nheads,
                batch_size,
                seq_num,
                cu_seqlens_kv,
                kv.stride(0),
                kv.stride(1),
                k_pos_emb.stride(0),
                o_key.stride(0),
                o_key.stride(1),
                cp_rank,
                cp_size,
            )
        ctx.save_for_backward(cos, sin)
        ctx.rotary_interleaved = rotary_interleaved
        ctx.emb_dim = emb_dim
        ctx.k_dim = k_dim
        ctx.cu_seqlens_kv = cu_seqlens_kv
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        if cu_seqlens_kv is None:
            o_key = o_key.view(max_seqlen, -1, nheads, emb_dim + k_dim)
        return o_key

    @staticmethod
    def backward(ctx, dk):
        """
        Backward function for ApplyMLARotaryEmbAbsorbKV.

        Args:
            dk: [seq_len, batch_size, head_num, emb_dim + k_dim]
                or [total_seq_len, head_num, emb_dim + k_dim]
        """
        cos, sin = ctx.saved_tensors
        max_seqlen = None
        batch_size = None
        seq_num = None
        if ctx.cu_seqlens_kv is None:
            # sbhd
            max_seqlen, batch_size, nheads, _ = dk.shape
            dk = dk.contiguous().view(-1, nheads, ctx.emb_dim + ctx.k_dim)
            total_seqlen = dk.shape[0]
        else:
            # thd
            seq_num = len(ctx.cu_seqlens_kv) - 1
            total_seqlen, nheads, _ = dk.shape
        assert dk.stride(-1) == 1

        d_kv = dk.new_empty(total_seqlen, nheads, ctx.k_dim)
        d_emb = dk.new_empty(total_seqlen, 1, ctx.emb_dim)

        grid = lambda META: (total_seqlen, triton.cdiv(nheads, META["BLOCK_H"]))
        
        # Choose kernel based on mode
        if ctx.rotary_interleaved:
            rotary_bwd_absorb_kv_kernel_interleaved[grid](
                dk,
                d_kv,
                d_emb,
                cos,
                sin,
                ctx.emb_dim,
                ctx.k_dim,
                nheads,
                batch_size,
                seq_num,
                ctx.cu_seqlens_kv,
                dk.stride(0),
                dk.stride(1),
                d_kv.stride(0),
                d_kv.stride(1),
                d_emb.stride(0),
                ctx.cp_rank,
                ctx.cp_size,
            )
        else:
            rotary_bwd_absorb_kv_kernel_non_interleaved[grid](
                dk,
                d_kv,
                d_emb,
                cos,
                sin,
                ctx.emb_dim,
                ctx.k_dim,
                nheads,
                batch_size,
                seq_num,
                ctx.cu_seqlens_kv,
                dk.stride(0),
                dk.stride(1),
                d_kv.stride(0),
                d_kv.stride(1),
                d_emb.stride(0),
                ctx.cp_rank,
                ctx.cp_size,
            )
        if ctx.cu_seqlens_kv is None:
            d_kv = d_kv.view(max_seqlen, batch_size, nheads, ctx.k_dim)
            d_emb = d_emb.view(max_seqlen, batch_size, 1, ctx.emb_dim)
        return d_kv, d_emb, None, None, None, None, None, None, None, None


def fused_apply_mla_rope_for_absorb_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    emb_dim: int,
    k_dim: int,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
):
    """
    Fused function for applying YARN RoPE to MLA's key and value.
    It splits the input tensor kv into key and value,
    and concatenates the processed RoPE to the key.

    For the notations below, seq_len is the length of sequence per batch for sbhd format,
    total_seq_len is the total length of the sequences for thd format.
    max_seq_len is the maximum length of the sequences in the input tensor.

    Args:
        kv: [seq_len, batch_size, head_num, k_dim + v_dim]
            or [total_seq_len, head_num, k_dim + v_dim]
        k_pos_emb: [seq_len, batch_size, 1, emb_dim] or [total_seq_len, 1, emb_dim]
        cos/sin: [max_seq_len, 1, 1, emb_dim]
        cu_seqlens_kv: [seq_num + 1] accumulated sequence lengths for thd format
        rotary_interleaved: whether to apply RoPE interleaved (True) or non-interleaved (False)

    Returns:
        key: [seq_len, batch_size, head_num, emb_dim + k_dim]
            or [total_seq_len, head_num, emb_dim + k_dim]
        value: [seq_len, batch_size, head_num, v_dim] or [total_seq_len, head_num, v_dim]
    """
    return ApplyMLARotaryEmbAbsorbKV.apply(
        kv,
        k_pos_emb,
        cos,
        sin,
        emb_dim,
        k_dim,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
        rotary_interleaved,
    )


@triton.jit
def triton_attn_dist_kernel(
    p_out_ptr,
    output_ptr,
    sm_scale,
    s_q, topk,
    stride_p_s: tl.int64, stride_p_h: tl.int64, stride_p_k: tl.int64,
    stride_o_s: tl.int64, stride_o_k: tl.int64,
    H_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for computing attention distribution over heads.

    This kernel computes the attention distribution by:
    1) Iterating over each query position and head
    2) Applying softmax over the topk dimension (after scaling by sm_scale)
    3) Averaging the resulting probabilities across heads

    Args:
        p_out_ptr: Pointer to the attention logits tensor of shape [s_q, h_q, topk]
        output_ptr: Pointer to the output tensor of shape [s_q, topk]
        sm_scale: Scaling factor for logits before softmax
        s_q: Number of query positions, Not used.
        topk: Number of top elements to consider, Not used.
        stride_p_s: Stride for p_out_ptr along sequence dimension
        stride_p_h: Stride for p_out_ptr along head dimension
        stride_p_k: Stride for p_out_ptr along topk dimension
        stride_o_s: Stride for output_ptr along sequence dimension
        stride_o_k: Stride for output_ptr along topk dimension
        H_Q: Number of heads (constexpr)
        BLOCK_K: Block size for topk dimension (constexpr, must equal topk)
    """
    s_idx = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK_K)    
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for h_idx in range(H_Q):
        p_ptrs = p_out_ptr + s_idx * stride_p_s + h_idx * stride_p_h + k_offs * stride_p_k

        p = tl.load(p_ptrs)
        attn_score = p * sm_scale

        max_val = tl.max(attn_score, axis=0)
        exp_val = tl.exp(attn_score - max_val)
        sum_exp = tl.sum(exp_val, axis=0)
        attn_prob = exp_val / sum_exp  # [BLOCK_K]

        acc += attn_prob

    o_ptrs = output_ptr + s_idx * stride_o_s + k_offs * stride_o_k
    tl.store(o_ptrs, acc / H_Q)


def triton_attn_dist(p_out: torch.Tensor, sm_scale) -> torch.Tensor:
    """
    Compute an averaged attention distribution over heads for a *top-k* attention tensor.

    This is a convenience wrapper around a Triton kernel that:
      1) For each query position `s`, iterates over all heads `h`.
      2) Applies a per-head softmax over the `topk` dimension to convert logits to probabilities:
           attn_prob[s, h, :] = softmax(p_out[s, h, :] * sm_scale)
      3) Averages the resulting probability distributions across heads:
           output[s, :] = mean_h(attn_prob[s, h, :])

    Args:
        p_out (torch.Tensor):
            Attention logits of shape (s_q, h_q, topk). Typically `float16`/`bfloat16`.
            The softmax is computed independently for each (s, h) over the last dimension.
        sm_scale (float | torch.Tensor):
            Scaling factor applied to logits before softmax (commonly 1/sqrt(d_k)).
            Must be broadcastable to a scalar in the Triton kernel.

    Returns:
        torch.Tensor:
            Averaged attention distribution of shape (s_q, topk), same dtype/device as `p_out`.

    Shape:
        Input:  (s_q, h_q, topk)
        Output: (s_q, topk)

    Notes:
        - The kernel computes softmax in fp32 for improved numerical stability.
        - `topk` must be a power of 2, enforced by:
              assert topk == triton.next_power_of_2(topk)
          If your `topk` is not a power of 2, pad `p_out` along the last dimension first.
        - The kernel assumes `BLOCK_K == topk` and processes the full last dimension in one program.
        - This function returns the *mean* probability across heads (not sum).

    Example:
        >>> p_out = torch.randn(1024, 16, 128, device="cuda", dtype=torch.float16)
        >>> sm_scale = (128 ** -0.5)
        >>> dist = triton_attn_dist(p_out, sm_scale)
        >>> dist.shape
        torch.Size([1024, 128])
    """
    s_q, h_q, topk = p_out.shape
    output = torch.empty((s_q, topk), device=p_out.device, dtype=p_out.dtype)

    assert topk == triton.next_power_of_2(topk)

    grid = (s_q,)

    triton_attn_dist_kernel[grid](
        p_out, output, sm_scale,
        s_q, topk,
        p_out.stride(0), p_out.stride(1), p_out.stride(2),
        output.stride(0), output.stride(1),
        H_Q=h_q,
        BLOCK_K=topk,
    )
    return output


def padded_flashinfer_topk(logits, topk, sk, *, sorted=True):
    """
    Compute top-k values and indices from logits tensor, with padding support.

    This function computes the top-k values and indices from the last dimension of the logits tensor.
    If the requested topk is larger than the last dimension, it first computes full top-d and then pads
    the results with -infinity for values and sk (sequence length) for indices.

    Args:
        logits (torch.Tensor): Input tensor of shape [..., d] containing logits
        topk (int): Number of top elements to return
        sk (int): Sequence length, used for padding indices when topk > d
        sorted (bool, optional): Whether to sort the returned values and indices. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - vals: Top-k values tensor of shape [..., topk]
            - idx: Top-k indices tensor of shape [..., topk]

    Examples:
        >>> logits = torch.randn(10, 20)  # [batch, seq_len]
        >>> vals, idx = padded_flashinfer_topk(logits, 5, 20)
        >>> vals.shape, idx.shape
        (torch.Size([10, 5]), torch.Size([10, 5]))

        >>> # Case where topk > seq_len
        >>> vals, idx = padded_flashinfer_topk(logits, 25, 20)
        >>> vals.shape, idx.shape
        (torch.Size([10, 25]), torch.Size([10, 25]))
    """
    d = logits.size(-1)
    topk = int(topk)
    if topk <= d:
        return flashinfer.top_k(logits, topk, sorted=sorted)

    # compute full top-d, then pad
    vals, idx = flashinfer.top_k(logits, d, sorted=sorted)
    pad = topk - d
    vals = torch.cat([vals, vals.new_full((*vals.shape[:-1], pad), float("-inf"))], dim=-1)
    idx  = torch.cat([idx, idx.new_full((*idx.shape[:-1], pad), sk)], dim=-1)  # use key length to fill
    return vals, idx


class DSADotProductAttentionFunction(torch.autograd.Function):
    """
    Tilelang's sparse_mla interface for BF16.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        indices,
        chunk_offset=0,
        sm_scale=None,
        d_v=512,
        return_p_out=False,
        packed_seq_params=None
    ):
        """
        Flash MLA sparse_mla_forward.
        """
        # q_flash [sq, n, d]
        # kv_flash [skv, 1, d]
        # indices_flash [sq, 1, topk]
        q_flash = q
        kv_flash = kv
        indices_flash = indices

        sq = q.size(0)
        
        out, _, lse, *p_out = flash_mla_sparse_fwd(
            q_flash,  # q: [s_q, h_q, d_qk], bfloat16
            kv_flash,  # kv: [s_kv, h_kv, d_qk], bfloat16
            indices_flash,  # [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
            sm_scale,
            d_v,
            q_start_index_s=chunk_offset,
            write_p_out=return_p_out
        )

        ctx.save_for_backward(q_flash, kv_flash, indices_flash, out, lse)
        ctx.sm_scale = sm_scale
        ctx.chunk_offset = chunk_offset
        ctx.sq = sq

        out = out.unsqueeze(0)

        if return_p_out:
            return out, p_out[0]
        else:
            return out

    @staticmethod
    def backward(ctx, grad_out, grad_p_out=None):
        """
        TileLang's sparse_mla_backward.
        """

        q, kv, indices, out, lse = ctx.saved_tensors
        grad_out = grad_out.squeeze(0).contiguous()
        major, _ = torch.cuda.get_device_capability()

        if major == 10:
            grad_q, grad_kv = flash_mla_sparse_bwd(
                q, kv, out, grad_out, indices, lse,
                sm_scale=ctx.sm_scale,
                q_start_index_s=ctx.chunk_offset,
            )
        else:
            log2e = 1.44269504
            offsets = torch.tensor([0, ctx.sq], dtype=torch.int32, device="cuda")
            grad_q, grad_kv = sparse_mla_bwd_interface(
                q,
                kv,
                out,
                grad_out,
                indices,
                lse / log2e,
                offsets,
                chunk_offset=ctx.chunk_offset,
                sm_scale=ctx.sm_scale,
                return_kernel=False,
                delta=None
            )
            
        return grad_q, grad_kv, None, None, None, None, None, None


class DSADotProductAttention(MegatronModule):
    """
    Wrapper for TileLang's `sparse_mla_interface` that supports DeepSeek sparse attention.

    This class provides an interface for computing sparse attention using the TileLang
    sparse attention operations optimized for DeepSeek models.
    """

    cp_stream: torch.cuda.Stream = None

    def __init__(
        self,
        config: TransformerConfig,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.qkv_format: str = 'sbhd'
        # layer_number is not used.
        # attention_type is not used.
        # attention_dropout is not used.
        self.softmax_scale = softmax_scale
        # k_channels is not used.
        # v_channels is not used.
        # cp_comm_type is not used.

        self.kept_packed_seq_params = set(field.name for field in dataclasses.fields(PackedSeqParams))
        if get_te_version() < PkgVersion("1.3.0"):
            # TE 1.3.0 introduces precomputing max_seqlen to remove unnecessary kernels and D2H
            # copies (#555)
            # These two arguments did not exist prior to 1.3.0
            self.kept_packed_seq_params.discard("max_seqlen_q")
            self.kept_packed_seq_params.discard("max_seqlen_kv")

        if get_te_version() < PkgVersion("1.10.0"):
            # TE 1.8.0 introduces cu_seqlens_padded which is the cu_seqlens with paddings counted
            # in each individual sequence in THD format dataset
            # These two arguments did not exist prior to 1.8.0. Full support added in 1.10.0 (#1012)
            self.kept_packed_seq_params.discard("cu_seqlens_q_padded")
            self.kept_packed_seq_params.discard("cu_seqlens_kv_padded")

    def forward(
        self,
        query: Tensor,
        kv: Tensor,
        indices: Tensor,
        chunk_offset: int,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        return_p_out: bool = False,
    ):
        """Forward."""
        if attn_mask_type is None:
            attn_mask_type = AttnMaskType.causal

        assert (attn_mask_type == AttnMaskType.causal or attn_mask_type == AttnMaskType.padding_causal), (
            "DSADotProductAttention only support causal attention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, "Attention bias is not supported for DSADotProductAttention."

        assert self.config.qk_pos_emb_head_dim == 64, (
            f"DSADotProductAttention only support qk_pos_emb_head_dim 64, but got {self.config.qk_pos_emb_head_dim}."
        )

        packed_seq_kwargs = (
            {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
            if packed_seq_params is not None
            else {}
        )
        # overwrite self.qkv_format depending on self.config.apply_rope_fusion, which can be set
        # after init
        if self.config.apply_rope_fusion and is_te_min_version("0.13.0", check_equality=False):
            self.qkv_format = 'bshd'

        qkv_format = packed_seq_kwargs.get('qkv_format', self.qkv_format)

        # if qkv_format == 'bshd'
        #     query: [b, sq, n, kv_lora_rank + qk_pos_emb_head_dim]
        #     kv: [b, skv, kv_lora_rank + qk_pos_emb_head_dim]
        if self.config.apply_rope_fusion and qkv_format == 'bshd':
            query = query.transpose(0, 1).contiguous()
            kv = kv.unsqueeze(2).transpose(0, 1).contiguous()

        # Case: sequence packing
        #   query: [sq*b, n, kv_lora_rank + qk_pos_emb_head_dim]
        #   kv: [skv*b, kv_lora_rank + qk_pos_emb_head_dim]
        # FlashMLA accept query [s, h, d] and kv [s, 1, d], so no need to add extra batch dim
        elif qkv_format == 'thd':
            kv = kv.unsqueeze(1)  # add dummy head dim for kv

        # if qkv_format == 'sbhd':
        #     query: [sq, b, n, kv_lora_rank + qk_pos_emb_head_dim]
        #     kv: [skv, b, kv_lora_rank + qk_pos_emb_head_dim]
        elif qkv_format == 'sbhd':
            query = query.transpose(0, 1).contiguous()
            kv = kv.unsqueeze(2).transpose(0, 1).contiguous()

        # indices: [b, sq, topk] -> [b, sq, 1, topk]
        assert indices is not None, "DSADotProductAttention need topk_indices."
        indices = indices.unsqueeze(2)

        # FlashMLA does not support batched input
        if query.ndim == 4:
            query = query.squeeze(0)  # [b, sq, h, d] -> [sq, h, d]
        if kv.ndim == 4:
            kv = kv.squeeze(0)  # [b, skv, h, d] -> [skv, h, d]
        if indices.ndim == 4:
            indices = indices.squeeze(0)  # [b, sq, 1, topk] -> [sq, 1, topk]

        args = (
            query,
            kv,
            indices,
            chunk_offset,
            self.softmax_scale,
            self.config.kv_lora_rank,
            return_p_out,
            packed_seq_params,
        )
        # core_attn_out [b, s/TP, h, d_v], p_out: list of [s/TP, h, topk]
        core_attn_out, *p_out = DSADotProductAttentionFunction.apply(*args)
        
        if qkv_format == 'sbhd':
            core_attn_out = core_attn_out.contiguous()
        elif qkv_format == 'thd':
            # [1, t, h, d_v] -> [t, h, d_v]
            core_attn_out = core_attn_out.squeeze(0).contiguous()
        else:
            core_attn_out = core_attn_out.contiguous()

        if return_p_out:
            return core_attn_out, p_out[0]
        else:
            return core_attn_out


class DSAIndexerKernelFunction(torch.autograd.Function):
    """
    Autograd function for DeepGEMM FP8Indexer kernel.

    This function implements the forward and backward passes for computing
    sparse attention indices using quantized query and key tensors with
    Float8BlockQuantizer.

    The forward pass:
    1. Quantizes input query and key tensors using Float8BlockQuantizer
    2. Computes scaled dot product between query and key in FP8
    3. Computes top-k indices from the resulting score matrix
    4. Handles sequence packing and causal masking

    The backward pass:
    1. Computes gradients for query, key and weights
    2. Handles the scaling factors properly during gradient computation

    Attributes:
        quantizer (Float8BlockQuantizer): FP8 quantizer with blockwise quantization
    """
    quantizer = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        amax_epsilon=1e-12,
        force_pow_2_scales=True,
        block_scaling_dim=1
    )

    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,  # [s, h, d]
        index_k: torch.Tensor,  # [s, d]
        weights: torch.Tensor,  # [s, h]
        index_topk: int,
        chunk_offset: int,
        packed_seq_params: Optional[PackedSeqParams],
    ):
        """
        DeepGEMM FP8Indexer forward.

        Quantizer: Float8BlockQuantizer with blockwise = 128.
        """
        assert index_q.ndim == 3 and index_k.ndim == 2 and weights.ndim == 2
        seq_q, head, dim = index_q.size()
        seq_k, _dim = index_k.size()
        assert dim == _dim, "Query and Key have diff dim."
        assert dim == 128, "Only support dim with size 128."
        device = index_q.device
        
        softmax_scale = (dim ** -0.5)

        quantized_q = DSAIndexerKernelFunction.quantizer.quantize(index_q)
        quantized_k = DSAIndexerKernelFunction.quantizer.quantize(index_k)
        q_fp8 = quantized_q.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        k_fp8 = quantized_k.get_data_tensors(rowwise_data=True, columnwise_data=False).view(torch.float8_e4m3fn)
        q_scale = quantized_q._rowwise_scale_inv.reshape(index_q.shape[:-1])  # [seq_q, head, 1] -> [seq_q, head]
        k_scale = quantized_k._rowwise_scale_inv.reshape(index_k.shape[:-1])  # [seq_k, head, 1] -> [seq_k, head]

        if packed_seq_params is None:
            k_start = torch.zeros(seq_q, dtype=torch.int, device=device)
        else:
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            seqlens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            full_seq_ids = torch.repeat_interleave(torch.arange(len(seqlens), device=device, dtype=torch.int), seqlens)
            local_seq_ids = full_seq_ids[chunk_offset:chunk_offset + seq_q]
            k_start = cu_seqlens_kv[local_seq_ids]

        k_end = torch.arange(seq_q, dtype=torch.int, device=device) + chunk_offset + 1

        weight_scaled = weights * q_scale * softmax_scale  # absorb the `sf_q` and `softmax_scale` into weights

        if packed_seq_params is None:
            # index_score [sq, sk]
            index_score = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weight_scaled, k_start, k_end)
        else:
            # index_score [sq, max_seqlen_k]
            max_seqlen_k = 0 if packed_seq_params is None else packed_seq_params.max_seqlen_kv
            index_score = deep_gemm.fp8_mqa_logits(
                q_fp8,
                (k_fp8, k_scale),
                weight_scaled,
                k_start, k_end,
                clean_logits=False,
                max_seqlen_k=max_seqlen_k
            )
            # Post-process to clean logits, apply causal mask, k_start is all zeros so omit here
            mask = torch.arange(max_seqlen_k, device='cuda')[None, :] < (k_end - k_start)[:, None]
            index_score = index_score.masked_fill(~mask, float('-inf'))

        index_score_topk, topk_indices = padded_flashinfer_topk(index_score.contiguous(), index_topk, seq_k)

        # In sft packing case, index_score is of shape [sq, max_seqlen_kv],
        # the topk_indices is relative indices within its sequence, convert to global indices by adding k_start
        if packed_seq_params is not None:
            topk_indices = topk_indices + k_start.unsqueeze(1)  # index may exceed sk, sparse_attn will handle that

        ctx.softmax_scale = softmax_scale
        ctx.index_topk = index_topk
        ctx.save_for_backward(q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, k_start, k_end)

        return index_score_topk, topk_indices

    @staticmethod
    def backward(ctx, grad_score, grad_topk):
        """
        DeepGEMM FP8Indexer backward.

        Computes gradients for:
        1. Quantized query tensor (d_q)
        2. Quantized key tensor (d_k)
        3. Weights tensor (d_weights)

        Args:
            ctx: Autograd context containing saved tensors from forward pass
            grad_score: Gradient of loss wrt index_score_topk (output from forward pass)
            grad_topk: Gradient of loss wrt topk_indices (output from forward pass)

        Returns:
            Tuple containing gradients for:
            - d_q: Gradient wrt quantized query tensor
            - d_k: Gradient wrt quantized key tensor
            - d_weights: Gradient wrt weights tensor
            - None: Placeholder for index_topk gradient (not differentiable)
            - None: Placeholder for chunk_offset gradient (not differentiable)
            - None: Placeholder for packed_seq_params gradient (not differentiable)
        """
        q_fp8, k_fp8, q_scale, k_scale, weight_scaled, topk_indices, ks, ke = ctx.saved_tensors

        d_q, d_k, d_weights = lightning_indexer_bwd.fp8_mqa_logits_bwd(
            grad_score.contiguous(),
            q_fp8,
            (k_fp8, k_scale),
            weight_scaled,
            ks,
            ke,
            topk_indices=topk_indices.int(),
            topk=ctx.index_topk
        )

        d_weights = d_weights * q_scale * ctx.softmax_scale
        d_q = d_q / q_scale.unsqueeze(-1)
        d_k = d_k / k_scale.unsqueeze(-1)

        return d_q, d_k, d_weights, None, None, None


class DSAIndexerKernel(torch.nn.Module):
    """
    Wrapper for DeepGEMM FP8Indexer kernel that computes sparse attention indices.

    This class provides an interface for computing quantized query-key dot products
    and extracting top-k indices using FP8 quantization for efficiency.

    The computation involves:
    1. Quantizing query and key tensors using FP8 block quantization
    2. Computing scaled dot product in FP8 precision
    3. Extracting top-k indices with optional sequence packing support
    4. Handling causal masking for autoregressive models

    Attributes:
        quantizer (Float8BlockQuantizer): FP8 quantizer with blockwise quantization
    """

    def forward(self, index_q, index_k, weights, index_topk, chunk_offset, packed_seq_params):
        """
        Call to the autograd function.
        """
        args = (
            index_q,
            index_k,
            weights,
            index_topk,
            chunk_offset,
            packed_seq_params,
        )
        return DSAIndexerKernelFunction.apply(*args)
