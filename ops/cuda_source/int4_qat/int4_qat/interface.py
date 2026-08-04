"""
Python interface for INT4 fake quantization.

Two levels of API:
  1. fake_int4_quant()              — low-level: quantize only → (codes, scale, zero)
  2. fake_int4_quantize_dequantize() — mid-level: quant+dequant → fake-quantized tensor
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Try to import the CUDA extension; fall back to None if not built.
try:
    import int4_qat.cuda as _C
except ImportError:
    _C = None

# Try to import the fused CUDA extension (3.4x faster quant+dequant in one pass).
try:
    import int4_qat.cuda_fused as _C_fused
except ImportError:
    _C_fused = None


def fake_int4_quant(
    x: torch.Tensor,
    block_size: List[int],
    sym: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Low-level INT4 quantization (quantize only, no dequant).

    Returns integer codes stored in the original floating dtype.

    Args:
        x: 2D CUDA tensor [M, N].
        block_size: [block_m, block_n]. Must satisfy block_m * block_n % 32 == 0.
        sym: Symmetric ([-7, 7]) or asymmetric ([0, 15]) quantization.

    Returns:
        (q, scale, zero):
            q:     [M, N], integer codes in float dtype
            scale: [ceil(M/block_m), ceil(N/block_n)]
            zero:  same shape as scale (only meaningful for asymmetric)
    """
    if _C is not None:
        return _C.fake_int4_quant_cuda(x.contiguous(), block_size, sym)
    else:
        return _fake_int4_quant_pytorch(x, block_size, sym)


def _fake_int4_quant_pytorch(
    x: torch.Tensor,
    block_size: List[int],
    sym: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation of INT4 quantization."""
    assert x.dim() == 2, "Input must be 2D"
    M, N = x.shape
    block_m, block_n = block_size

    scale_rows = math.ceil(M / block_m)
    scale_cols = math.ceil(N / block_n)

    # Pad to block-aligned shape
    M_pad = scale_rows * block_m
    N_pad = scale_cols * block_n
    if M_pad > M or N_pad > N:
        x_pad = torch.zeros(M_pad, N_pad, dtype=x.dtype, device=x.device)
        x_pad[:M, :N] = x
    else:
        x_pad = x

    # Reshape into blocks: [scale_rows, block_m, scale_cols, block_n]
    x_blocks = x_pad.view(scale_rows, block_m, scale_cols, block_n)

    if sym:
        abs_max = x_blocks.abs().float().amax(dim=(1, 3))  # [scale_rows, scale_cols]
        scale = torch.clamp(abs_max / 7.0, min=1e-5).to(x.dtype)
        # Quantize: divide, round, clamp to [-7, 7]
        q_blocks = torch.round(
            x_blocks.float() / scale[:, None, :, None].float()
        ).clamp(-7, 7).to(x.dtype)
        zero = torch.zeros_like(scale)
    else:
        block_min = x_blocks.float().amin(dim=(1, 3))
        block_max = x_blocks.float().amax(dim=(1, 3))
        scale = torch.clamp((block_max - block_min) / 15.0, min=1e-5).to(x.dtype)
        zero = torch.clamp(
            torch.round(-block_min / scale.float()), min=0.0, max=15.0
        ).to(x.dtype)
        q_blocks = (
            torch.round(x_blocks.float() / scale[:, None, :, None].float())
            + zero[:, None, :, None].float()
        ).clamp(0, 15).to(x.dtype)

    q = q_blocks.view(M_pad, N_pad)[:M, :N].contiguous()
    return q, scale, zero


def fake_int4_quantize_dequantize(
    weight: torch.Tensor,
    group_size: int = 128,
    sym: bool = True,
) -> torch.Tensor:
    """Fake INT4 quantize-then-dequantize.

    Quantizes the weight to INT4 codes, then dequantizes back to original dtype.
    The output has the same shape and dtype as input but values are restricted to
    the INT4 quantization grid.

    When the fused CUDA kernel is available and conditions are met (symmetric mode,
    group_size=32, 2D contiguous CUDA tensor), uses a single-pass fused kernel
    that is ~3.4x faster than the two-pass approach.

    Args:
        weight: 2D tensor [out_features, in_features].
        group_size: Quantization group size along in_features (block_size = [1, group_size]).
        sym: Symmetric or asymmetric quantization.

    Returns:
        Fake-quantized weight of same shape and dtype.
    """
    # Fast path: fused CUDA kernel (single-pass, ~3.4x faster)
    if (
        _C_fused is not None
        and sym
        and group_size == 32
        and weight.is_cuda
        and weight.dim() == 2
        and weight.is_contiguous()
    ):
        return _C_fused.fused_fake_int4_quantize_dequantize_cuda(weight)

    # Fallback: two-pass (quant kernel + PyTorch dequant)
    M, N = weight.shape
    block_size = [1, group_size]
    scale_cols = math.ceil(N / group_size)

    q, scale, zero = fake_int4_quant(weight, block_size, sym)

    # Dequantize: broadcast scale from [M, scale_cols] → [M, N]
    scale_expanded = scale.unsqueeze(-1).expand(M, scale_cols, group_size)
    scale_full = scale_expanded.reshape(M, scale_cols * group_size)[:, :N]

    if sym:
        return (q * scale_full).to(weight.dtype)
    else:
        zero_expanded = zero.unsqueeze(-1).expand(M, scale_cols, group_size)
        zero_full = zero_expanded.reshape(M, scale_cols * group_size)[:, :N]
        return ((q - zero_full) * scale_full).to(weight.dtype)
