# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fused blockwise FP8 kernels for optimizer-state-shard owner parameter publication."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - validated before enabling the path.
    triton = None
    tl = None

E4M3_MAX = 448.0
DEFAULT_BLOCK = 256
NUM_BLOCKS_PER_TILE = 8


def validate_runtime(device: torch.device, backend: str) -> None:
    """Fail early unless the selected process group supports fused FP8."""
    device = torch.device(device)
    if device.type != "cuda" or "nccl" not in str(backend).lower():
        raise RuntimeError(
            "fp8_e4m3_delta parameter sync requires a CUDA device and NCCL"
        )
    if triton is None or tl is None or not hasattr(tl, "float8e4nv"):
        raise RuntimeError(
            "fp8_e4m3_delta parameter sync requires Triton tl.float8e4nv"
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("fp8_e4m3_delta requires PyTorch E4M3 support")
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("fp8_e4m3_delta requires compute capability >= 8.9")


if triton is not None:

    @triton.jit
    def _quantize_kernel(
        X,
        Q,
        S,
        numel,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Quantize contiguous fp32 deltas to E4M3 with one scale per block."""
        pid = tl.program_id(0).to(tl.int64)
        block0 = pid * NB
        block_offsets = tl.arange(0, NB).to(tl.int64)
        element_offsets = tl.arange(0, BLOCK).to(tl.int64)
        offsets = (block0 + block_offsets)[:, None] * BLOCK + element_offsets[None, :]
        mask = offsets < numel
        block_mask = block0 + block_offsets < (numel + BLOCK - 1) // BLOCK
        values = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=1)
        scale = amax / 448.0
        inverse = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        quantized = (values * inverse[:, None]).to(tl.float8e4nv)
        tl.store(Q + offsets, quantized.to(tl.uint8, bitcast=True), mask=mask)
        tl.store(S + block0 + block_offsets, scale, mask=block_mask)

    @triton.jit
    def _dequantize_kernel(
        Q,
        S,
        OUT,
        numel,
        BLOCK: tl.constexpr,
        NB: tl.constexpr,
    ):
        """Dequantize E4M3 bytes into a contiguous fp32 delta buffer."""
        pid = tl.program_id(0).to(tl.int64)
        block0 = pid * NB
        block_offsets = tl.arange(0, NB).to(tl.int64)
        element_offsets = tl.arange(0, BLOCK).to(tl.int64)
        offsets = (block0 + block_offsets)[:, None] * BLOCK + element_offsets[None, :]
        mask = offsets < numel
        block_mask = block0 + block_offsets < (numel + BLOCK - 1) // BLOCK
        bits = tl.load(Q + offsets, mask=mask, other=0).to(tl.uint8)
        values = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scales = tl.load(
            S + block0 + block_offsets, mask=block_mask, other=0.0
        )
        tl.store(OUT + offsets, values * scales[:, None], mask=mask)


def _grid(numel: int, block: int) -> tuple[int]:
    tile = block * NUM_BLOCKS_PER_TILE
    return ((numel + tile - 1) // tile,)


def quantize_into(delta, quantized, scales, block=DEFAULT_BLOCK) -> None:
    """Quantize a contiguous fp32 delta into preallocated buffers."""
    if triton is None:
        raise RuntimeError("fp8_e4m3_delta requires Triton")
    if not delta.is_contiguous() or delta.dtype is not torch.float32:
        raise ValueError("FP8 parameter delta must be contiguous fp32")
    _quantize_kernel[_grid(delta.numel(), block)](
        delta,
        quantized,
        scales,
        delta.numel(),
        BLOCK=block,
        NB=NUM_BLOCKS_PER_TILE,
    )


def dequantize_into(quantized, scales, output, block=DEFAULT_BLOCK) -> None:
    """Dequantize a contiguous FP8 payload into preallocated fp32 storage."""
    if triton is None:
        raise RuntimeError("fp8_e4m3_delta requires Triton")
    _dequantize_kernel[_grid(output.numel(), block)](
        quantized,
        scales,
        output,
        output.numel(),
        BLOCK=block,
        NB=NUM_BLOCKS_PER_TILE,
    )


__all__ = ["dequantize_into", "quantize_into", "validate_runtime"]
