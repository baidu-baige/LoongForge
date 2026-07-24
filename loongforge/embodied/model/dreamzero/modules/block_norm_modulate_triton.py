# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Default-off Triton block norm+modulate helper for DreamZero."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - runtime fallback handles this.
    triton = None
    tl = None


def _eager_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    normed = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    return normed * (1 + scale) + shift


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _norm_modulate_fwd_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        mean_ptr,
        rstd_ptr,
        scale_stride_b: tl.constexpr,
        scale_stride_l: tl.constexpr,
        scale_stride_d: tl.constexpr,
        shift_stride_b: tl.constexpr,
        shift_stride_l: tl.constexpr,
        shift_stride_d: tl.constexpr,
        seq_len: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch_idx = row // seq_len
        seq_idx = row - batch_idx * seq_len
        offsets = tl.arange(0, block)
        mask = offsets < n_cols
        base = row * n_cols + offsets
        scale_base = (
            batch_idx * scale_stride_b
            + seq_idx * scale_stride_l
            + offsets * scale_stride_d
        )
        shift_base = (
            batch_idx * shift_stride_b
            + seq_idx * shift_stride_l
            + offsets * shift_stride_d
        )

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + scale_base, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + shift_base, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) / n_cols
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)
        normed = centered * rstd
        out = normed * (1.0 + scale) + shift

        tl.store(out_ptr + base, out, mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _norm_modulate_bwd_kernel(
        grad_ptr,
        x_ptr,
        scale_ptr,
        mean_ptr,
        rstd_ptr,
        grad_x_ptr,
        grad_scale_ptr,
        grad_shift_ptr,
        scale_stride_b: tl.constexpr,
        scale_stride_l: tl.constexpr,
        scale_stride_d: tl.constexpr,
        grad_scale_stride_b: tl.constexpr,
        grad_scale_stride_l: tl.constexpr,
        grad_scale_stride_d: tl.constexpr,
        grad_shift_stride_b: tl.constexpr,
        grad_shift_stride_l: tl.constexpr,
        grad_shift_stride_d: tl.constexpr,
        seq_len: tl.constexpr,
        n_cols: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch_idx = row // seq_len
        seq_idx = row - batch_idx * seq_len
        offsets = tl.arange(0, block)
        mask = offsets < n_cols
        base = row * n_cols + offsets
        scale_base = (
            batch_idx * scale_stride_b
            + seq_idx * scale_stride_l
            + offsets * scale_stride_d
        )
        grad_scale_base = (
            batch_idx * grad_scale_stride_b
            + seq_idx * grad_scale_stride_l
            + offsets * grad_scale_stride_d
        )
        grad_shift_base = (
            batch_idx * grad_shift_stride_b
            + seq_idx * grad_shift_stride_l
            + offsets * grad_shift_stride_d
        )

        grad = tl.load(grad_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + scale_base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)

        normed = (x - mean) * rstd
        dnorm = tl.where(mask, grad * (1.0 + scale), 0.0)
        normed = tl.where(mask, normed, 0.0)
        sum_dnorm = tl.sum(dnorm, axis=0)
        sum_dnorm_normed = tl.sum(dnorm * normed, axis=0)
        grad_x = (dnorm - sum_dnorm / n_cols - normed * sum_dnorm_normed / n_cols) * rstd
        grad_scale = grad * normed
        grad_shift = grad

        tl.store(grad_x_ptr + base, grad_x, mask=mask)
        tl.store(grad_scale_ptr + grad_scale_base, grad_scale, mask=mask)
        tl.store(grad_shift_ptr + grad_shift_base, grad_shift, mask=mask)


def _can_use_triton(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    if triton is None:
        return False
    if not (x.is_cuda and scale.is_cuda and shift.is_cuda):
        return False
    if x.dim() != 3 or scale.dim() != 3 or shift.dim() != 3:
        return False
    if scale.shape != x.shape or shift.shape != x.shape:
        return False
    if x.dtype != scale.dtype or x.dtype != shift.dtype:
        return False
    if x.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        return False
    return x.is_contiguous() and scale.stride(-1) == 1 and shift.stride(-1) == 1


class _TritonBlockNormModulate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        eps: float,
        num_warps: int,
    ) -> torch.Tensor:
        """Run fused block normalization and modulation in the forward pass."""
        if not _can_use_triton(x, scale, shift):
            ctx.fallback = True
            ctx.eps = float(eps)
            ctx.save_for_backward(x, scale, shift)
            return _eager_norm_modulate(x, scale, shift, eps)

        ctx.fallback = False
        dim = int(x.shape[-1])
        seq_len = int(x.shape[-2])
        rows = int(x.numel() // dim)
        block = _next_power_of_2(dim)
        out = torch.empty_like(x)
        mean = torch.empty((rows,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
        _norm_modulate_fwd_kernel[(rows,)](
            x,
            scale,
            shift,
            out,
            mean,
            rstd,
            int(scale.stride(0)),
            int(scale.stride(1)),
            int(scale.stride(2)),
            int(shift.stride(0)),
            int(shift.stride(1)),
            int(shift.stride(2)),
            seq_len,
            dim,
            float(eps),
            block,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, scale, mean, rstd)
        ctx.dim = dim
        ctx.seq_len = seq_len
        ctx.block = block
        ctx.num_warps = int(num_warps)
        ctx.scale_stride = tuple(scale.stride())
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Compute gradients for fused block normalization and modulation."""
        if ctx.fallback:
            x, scale, shift = ctx.saved_tensors
            with torch.enable_grad():
                x_ref = x.detach().requires_grad_(ctx.needs_input_grad[0])
                scale_ref = scale.detach().requires_grad_(ctx.needs_input_grad[1])
                shift_ref = shift.detach().requires_grad_(ctx.needs_input_grad[2])
                out_ref = _eager_norm_modulate(x_ref, scale_ref, shift_ref, ctx.eps)
            grads = torch.autograd.grad(
                out_ref,
                (x_ref, scale_ref, shift_ref),
                grad_out,
                retain_graph=False,
                allow_unused=True,
            )
            return grads[0], grads[1], grads[2], None, None

        x, scale, mean, rstd = ctx.saved_tensors
        grad = grad_out.contiguous() if grad_out.stride(-1) != 1 else grad_out
        grad_x = torch.empty_like(x)
        grad_scale = torch.empty_like(scale, memory_format=torch.contiguous_format)
        grad_shift = torch.empty_like(grad, memory_format=torch.contiguous_format)
        rows = int(x.numel() // int(ctx.dim))
        _norm_modulate_bwd_kernel[(rows,)](
            grad,
            x,
            scale,
            mean,
            rstd,
            grad_x,
            grad_scale,
            grad_shift,
            int(ctx.scale_stride[0]),
            int(ctx.scale_stride[1]),
            int(ctx.scale_stride[2]),
            int(grad_scale.stride(0)),
            int(grad_scale.stride(1)),
            int(grad_scale.stride(2)),
            int(grad_shift.stride(0)),
            int(grad_shift.stride(1)),
            int(grad_shift.stride(2)),
            int(ctx.seq_len),
            int(ctx.dim),
            int(ctx.block),
            num_warps=ctx.num_warps,
        )
        return grad_x, grad_scale, grad_shift, None, None


def triton_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    num_warps: int = 8,
) -> torch.Tensor:
    """Apply block normalization followed by scale and shift modulation."""
    return _TritonBlockNormModulate.apply(x, scale, shift, float(eps), int(num_warps))
