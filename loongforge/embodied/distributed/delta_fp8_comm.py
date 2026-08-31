# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for FSDP2 delta-FP8 AllGather.

FSDP2 all-gathers unsharded BF16 parameters every step. This path quantizes
the per-rank delta against a persistent unsharded BF16 reference instead of
the absolute weight:

    y_t = y_{t-1} + dequant(fp8(x_t - y_{t-1}[rank]))

``y`` is identical on every rank, so every rank adds the same dequantized
deltas and stays bit-identical. The next delta is measured against the
reconstruction, so quantization error is corrected instead of accumulated.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - install path raises a clearer error.
    triton = None
    tl = None

import torch

E4M3_MAX = 448.0
DEFAULT_BLOCK = 256


def require_triton() -> None:
    """Raise if the delta-FP8 kernels cannot be compiled or launched."""
    if triton is None:
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires Triton to be installed"
        )


def _backend_for_device(backend: str, device_type: str) -> str:
    """Resolve a plain or device-scoped backend to ``device_type``."""
    entries = [entry.strip() for entry in str(backend).lower().split(",")]
    scoped = {}
    unscoped = None
    for entry in entries:
        if ":" in entry:
            scope, name = entry.rsplit(":", 1)
            scoped[scope] = name
        elif entry:
            unscoped = entry
    return scoped.get(device_type, unscoped or "")


def validate_runtime(device: torch.device, backend: str) -> None:
    """Validate the runtime used by the CUDA-only delta-FP8 kernels.

    The kernels use Triton's NVIDIA ``float8e4nv`` type and are launched on the
    same CUDA stream as NCCL's FSDP all-gather.  Failing here keeps unsupported
    XPU/CPU devices, non-NCCL process groups, and pre-FP8 NVIDIA GPUs from
    failing much later at the first parameter unshard.
    """
    device = torch.device(device)
    backend_name = str(backend).lower()
    device_backend = _backend_for_device(backend_name, device.type)
    context = f"device={device}, backend={backend_name}"
    if device.type != "cuda":
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires a CUDA device and NCCL; "
            f"got {context}"
        )
    if device_backend != "nccl":
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires the NCCL backend; "
            f"got {context} (resolved {device.type}:{device_backend or 'unknown'})"
        )
    if triton is None:
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires Triton to be installed; "
            f"got {context}"
        )
    if tl is None or not hasattr(tl, "float8e4nv"):
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires Triton FP8 type "
            "tl.float8e4nv; "
            f"got {context}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires CUDA to be available; "
            f"got {context}"
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires PyTorch FP8 E4M3 support; "
            f"got {context}"
        )

    try:
        capability = torch.cuda.get_device_capability(device)
    except Exception as exc:
        raise RuntimeError(
            "Unable to query CUDA compute capability for "
            f"--fsdp-delta-fp8-allgather ({context})"
        ) from exc
    if capability < (8, 9):
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires an NVIDIA GPU with "
            f"compute capability >= 8.9 for tl.float8e4nv; got {capability} "
            f"({context})"
        )

    try:
        target = triton.runtime.driver.active.get_current_target()
        triton_backend = str(target.backend).lower()
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize Triton's CUDA backend for "
            f"--fsdp-delta-fp8-allgather ({context})"
        ) from exc
    if triton_backend != "cuda":
        raise RuntimeError(
            "--fsdp-delta-fp8-allgather requires Triton's CUDA backend; "
            f"got triton_backend={triton_backend!r} ({context})"
        )


if triton is not None:

    @triton.jit
    def _quantize_delta_kernel(X, YREF, Q, S, numel, BLOCK: tl.constexpr):
        """Quantize ``x - yref`` into per-block scaled FP8 stored as uint8."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        yref = tl.load(YREF + offs, mask=mask, other=0.0).to(tl.float32)
        delta = x - yref
        # Per-block scaling keeps the FP8 payload at one byte per element.
        amax = tl.max(tl.abs(delta), axis=0)
        scale = amax / 448.0
        inv_scale = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        quantized = (delta * inv_scale).to(tl.float8e4nv)
        tl.store(S + pid, scale)
        tl.store(Q + offs, quantized.to(tl.uint8, bitcast=True), mask=mask)

    @triton.jit
    def _dequantize_add_kernel(Y, Q, S, shard_numel, num_blocks, BLOCK: tl.constexpr):
        """Accumulate ``dequant(q)`` into the persistent reference in place.

        The block index is grid dimension 0 because CUDA caps grid dimensions
        1 and 2 at 65535 and a single FSDP unit can hold far more than 65535
        scale blocks.
        """
        pid = tl.program_id(0)
        rank = tl.program_id(1)
        block_offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = block_offs < shard_numel
        offs = rank * shard_numel + block_offs
        quantized = tl.load(Q + offs, mask=mask, other=0).to(tl.uint8)
        delta = quantized.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale = tl.load(S + rank * num_blocks + pid)
        y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
        # Update this rank's replica of the global reference in place; the next
        # step measures its delta against this reconstructed value.
        tl.store(Y + offs, (y + delta * scale).to(tl.bfloat16), mask=mask)


def quantize_delta(x, yref, block=DEFAULT_BLOCK):
    """Quantize one shard delta; also used by the standalone kernel test."""
    require_triton()
    numel = x.numel()
    num_blocks = (numel + block - 1) // block
    quantized = torch.empty(numel, dtype=torch.uint8, device=x.device)
    scales = torch.empty(num_blocks, dtype=torch.float32, device=x.device)
    _quantize_delta_kernel[(num_blocks,)](x, yref, quantized, scales, numel, BLOCK=block)
    return quantized, scales


def dequantize_add(y, quantized, scales, shard_numel, world_size, block=DEFAULT_BLOCK):
    """Add all ranks' dequantized deltas into the persistent reference."""
    require_triton()
    num_blocks = (shard_numel + block - 1) // block
    _dequantize_add_kernel[(num_blocks, world_size)](
        y, quantized, scales, shard_numel, num_blocks, BLOCK=block
    )


def quantize_delta_into(x, yref, quantized, scales, block=DEFAULT_BLOCK):
    """Quantize ``x - yref`` into preallocated buffers."""
    require_triton()
    numel = x.numel()
    num_blocks = (numel + block - 1) // block
    _quantize_delta_kernel[(num_blocks,)](x, yref, quantized, scales, numel, BLOCK=block)
