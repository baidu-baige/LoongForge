# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fused interleaved RoPE Triton kernel for WAN model."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_interleaved_kernel(
    OUT,
    X,
    COS,
    SIN,
    seqlen,
    rotary_dim,
    seqlen_ro,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
    rk_repeat = rk // 2

    X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
    X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
    COS_ptr = COS + (rm[:, None] * rotary_dim_half + rk_repeat[None, :])
    SIN_ptr = SIN + (rm[:, None] * rotary_dim_half + rk_repeat[None, :])

    seq_mask = rm[:, None] < seqlen
    k_mask = rk[None, :] < rotary_dim
    cs_mask = (rm[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half)

    cos = tl.load(COS_ptr, mask=cs_mask, other=1.0)
    sin = tl.load(SIN_ptr, mask=cs_mask, other=0.0)
    x0 = tl.load(X0, mask=seq_mask & k_mask, other=0.0).to(tl.float32)
    x1 = tl.load(X1, mask=seq_mask & (rk_swap[None, :] < rotary_dim), other=0.0).to(tl.float32)

    x0_cos = x0 * cos
    x1_sin = x1 * sin
    out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)

    OUT_ptr = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
    tl.store(OUT_ptr, out.to(tl.float32), mask=seq_mask & k_mask)


def _apply_rotary_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Apply interleaved rotary embedding (fused Triton kernel).

    Arguments:
        x: (batch, seqlen, nheads, headdim) - input tensor
        cos: (seqlen_ro, rotary_dim / 2) - cosine values
        sin: (seqlen_ro, rotary_dim / 2) - sine values
    Returns:
        output: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    seqlen_ro = cos.shape[0]
    rotary_dim = cos.shape[1] * 2

    output = torch.empty_like(x)
    if rotary_dim < headdim:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = triton.next_power_of_2(rotary_dim)
    BLOCK_M = 4
    grid = (triton.cdiv(seqlen, BLOCK_M), batch, nheads)

    with torch.cuda.device(x.device.index):
        _rotary_interleaved_kernel[grid](
            output,
            x,
            cos,
            sin,
            seqlen,
            rotary_dim,
            seqlen_ro,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            BLOCK_K=BLOCK_K,
            BLOCK_M=BLOCK_M,
            num_warps=4,
            num_stages=1,
        )
    return output


class _FusedWanRope(torch.autograd.Function):
    """Autograd wrapper around the fused interleaved-RoPE Triton kernel.

    Wraps :func:`_apply_rotary_interleaved` so that gradients propagate back
    through the fused kernel. The interleaved rotary is a per-pair rotation, so
    the Jacobian w.r.t. ``x`` is the same forward kernel evaluated with
    ``sin`` negated. ``cos``/``sin`` are position tables and do not
    receive gradients.
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        """Apply interleaved rotary embedding to ``x`` and save cos/sin for backward."""
        ctx.save_for_backward(cos, sin)
        return _apply_rotary_interleaved(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient by applying the forward kernel with ``sin`` negated."""
        cos, sin = ctx.saved_tensors
        grad_input = _apply_rotary_interleaved(grad_output.contiguous(), cos, -sin)
        return grad_input, None, None


def apply_rotary_interleaved(x, cos, sin, **kwargs):
    """Apply fused interleaved RoPE with autograd support."""
    return _FusedWanRope.apply(x, cos, sin)
