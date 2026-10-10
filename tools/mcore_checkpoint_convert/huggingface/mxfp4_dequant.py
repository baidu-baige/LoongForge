#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""On-the-fly MXFP4 (OCP Microscaling FP4) dequantization for HuggingFace checkpoints.

DeepSeek-V4 (and similar) ship routed-expert weights as packed MXFP4:

  * weight: E2M1 4-bit values, two packed per byte (safetensors dtype ``I8``).
  * scale:  per-block E8M0 factor, one per ``block_size=32`` consecutive E2M1
            values (safetensors dtype ``F8_E8M0``).

This module materializes such weights into ordinary floating-point ``.weight``
tensors so the downstream common/mcore conversion can treat experts as plain
BF16/FP16 linears. It is intentionally a pure-PyTorch, element-wise + gather
implementation — it only runs once during checkpoint bridging, so kernel-level
throughput is not a concern.

Ported from ModelOpt's ``MXFP4QTensor.dequantize``
(``modelopt.torch.quantization.qtensor.mxfp4_tensor``), flattened to stateless
functions and inlined here to avoid pulling ModelOpt in as a LoongForge dep.
"""

import logging
import time
from typing import Iterable, Optional

import torch

__all__ = ["dequantize_mxfp4", "dequantize_mxfp4_state_dict", "progress_print"]

LOGGER = logging.getLogger(__name__)


def progress_print(message: str) -> None:
    """Progress line visible in training logs: rank 0 only under
    torch.distributed (all ranks print when it is not initialized, e.g. the
    offline conversion tools). ``logging.info`` is swallowed by the default
    WARNING threshold in the training entrypoints, so progress must use print.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(message, flush=True)

# OCP MXFP4: 32 E2M1 values share one E8M0 block scale.
DEFAULT_MXFP4_BLOCK_SIZE = 32

# E2M1 representable magnitudes indexed by the 3 magnitude bits (bit2..0).
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# E8M0 is exponent-only with bias 127: effective scale = 2 ** (raw_uint - 127).
_E8M0_BIAS = 127

# 8-bit signed/unsigned integer dtypes whose raw bytes can be reinterpreted as
# uint8 for nibble / bit operations. safetensors ``I8`` loads as torch.int8.
_UINT8_VIEW_COMPATIBLE = (torch.uint8, torch.int8)

# Float8 dtypes safetensors may hand us for E8M0 / E4M3 scales (torch >= 2.1).
try:
    _FLOAT8_DTYPES = (torch.float8_e8m0fnu, torch.float8_e4m3fn, torch.float8_e5m2)
except AttributeError:  # torch < 2.1 has no float8 dtypes
    _FLOAT8_DTYPES = ()


def _as_uint8(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Reinterpret an 8-bit tensor's raw bytes as uint8 without copying.

    Accepts uint8/int8 (safetensors ``I8``) and the float8 dtypes safetensors may
    use for E8M0 scales. The bit-level unpack below is defined on the unsigned
    byte, so we view rather than cast — no values are rounded or sign-extended.
    """
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype in _UINT8_VIEW_COMPATIBLE or tensor.dtype in _FLOAT8_DTYPES:
        return tensor.view(torch.uint8)
    raise TypeError(
        f"{name}: expected an 8-bit packed/scale dtype "
        f"(uint8/int8/float8_e8m0fnu/e4m3fn/e5m2), got {tensor.dtype}."
    )


def dequantize_mxfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = DEFAULT_MXFP4_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
    compute_device: Optional[str] = None,
) -> torch.Tensor:
    """Dequantize a single MXFP4 packed tensor (E2M1 in uint8 + E8M0 block scale).

    Args:
        weight: packed E2M1 weights, two values per byte. Any leading dims; the
            last dim is the contraction (in_features) axis stored as
            ``in_features // 2`` bytes. Accepted as uint8/int8 (safetensors ``I8``)
            — reinterpreted, not rounded.
        scale: per-block E8M0 scale of shape ``... x (in_features // block_size)``.
            Accepted as uint8/int8/float8_e8m0fnu (safetensors ``F8_E8M0``).
        block_size: E2M1 values covered by one block scale. OCP MXFP4 = 32.
        dtype: output dtype (bf16/fp16/fp32).
        compute_device: optional device to route the computation through (e.g.
            ``"cuda"``). The packed tensors are copied there, dequantized, and
            the result is copied back to the input device — the state dict
            stays host-resident, only ~a few tens of MB transit the GPU at a
            time. All ops involved (bit masks, LUT gather, exp2, mul, bf16
            round) are exact/IEEE, so the result is bit-identical to the CPU
            path. None = compute in place on the input device.

    Returns:
        Dequantized tensor of shape ``... x in_features`` in ``dtype``, on the
        input's original device.
    """
    out_device = weight.device
    weight = _as_uint8(weight.contiguous(), "weight")
    scale = _as_uint8(scale.contiguous(), "scale")
    if compute_device is not None and torch.device(compute_device) != out_device:
        weight = weight.to(compute_device)
        scale = scale.to(compute_device)

    out_last = weight.shape[-1] * 2
    scale_last = scale.shape[-1]
    if out_last != scale_last * block_size:
        raise ValueError(
            f"MXFP4 shape mismatch: weight unpacked last dim {out_last} != "
            f"scale last dim {scale_last} * block_size {block_size}."
        )
    if list(weight.shape[:-1]) != list(scale.shape[:-1]):
        raise ValueError(
            f"MXFP4 shape mismatch: weight leading dims {list(weight.shape[:-1])} "
            f"!= scale leading dims {list(scale.shape[:-1])}."
        )

    # 1) Split each byte into its two E2M1 nibbles -> uint4 indices (last dim x2).
    low_nibble = weight & 0x0F
    high_nibble = (weight >> 4) & 0x0F
    idx = torch.empty(
        weight.shape[:-1] + (out_last,), dtype=torch.uint8, device=weight.device
    )
    idx[..., 0::2] = low_nibble
    idx[..., 1::2] = high_nibble

    # 2) Map each 4-bit pattern (sign | 3 magnitude bits) -> float via E2M1 LUT.
    lut = torch.tensor(_E2M1_MAGNITUDES, device=weight.device, dtype=torch.float32)
    magnitude_bits = (idx & 0b0111).to(torch.long)
    values = lut[magnitude_bits]
    sign = 1.0 - 2.0 * ((idx & 0b1000) >> 3).to(torch.float32)
    x = sign * values

    # 3) Apply per-block E8M0 scale: effective = 2 ** (raw_uint - 127).
    sf = torch.exp2(scale.to(torch.float32) - _E8M0_BIAS)
    x = x.reshape(-1, block_size) * sf.reshape(-1, 1)
    # Cast to the target dtype before the device copy-back so only half the
    # bytes cross PCIe when routed through a GPU.
    return x.reshape(idx.shape).to(dtype).to(out_device)


def dequantize_mxfp4_state_dict(
    state_dict: dict,
    output_dtype: torch.dtype = torch.bfloat16,
    target_weight_keys: Optional[Iterable[str]] = None,
    block_size: int = DEFAULT_MXFP4_BLOCK_SIZE,
    compute_device: Optional[str] = "auto",
) -> int:
    """Materialize MXFP4 (E2M1 + E8M0) expert weights in-place into ``output_dtype``.

    A key pair matches when ``<base>.weight`` is an 8-bit int dtype (uint8/int8)
    AND a companion ``<base>.scale`` exists. On match the dequantized tensor
    replaces ``<base>.weight`` and ``<base>.scale`` is dropped, so downstream
    conversion sees a plain floating-point linear. BF16/FP8 weights and any
    tensor without a paired ``.scale`` are left untouched.

    Args:
        state_dict: HF state dict to mutate in place.
        output_dtype: target dtype for the materialized weight.
        target_weight_keys: optional allow-list of ``<base>.weight`` keys to
            convert (e.g. only routed-expert weights). Entries that are not
            8-bit or lack a ``.scale`` are silently skipped. If None, every
            matching uint8/int8 ``.weight`` + ``.scale`` pair is converted.
        block_size: MXFP4 block size (default 32).
        compute_device: device to route per-tensor dequant math through.
            ``"auto"`` (default) uses the current CUDA device when available —
            ~200x faster than CPU at DSV4 expert sizes — falling back to the
            tensors' own device otherwise. Results are bit-identical either
            way; the state dict itself stays on its original (host) device.

    Returns:
        Number of tensors dequantized.
    """
    if compute_device == "auto":
        compute_device = "cuda" if torch.cuda.is_available() else None
    if target_weight_keys is None:
        weight_keys = sorted(
            k
            for k in state_dict
            if k.endswith(".weight")
            and state_dict[k].dtype in _UINT8_VIEW_COMPATIBLE
            and f"{k[:-len('.weight')]}.scale" in state_dict
        )
    else:
        weight_keys = []
        missing = []
        for weight_key in sorted(target_weight_keys):
            if not weight_key.endswith(".weight"):
                raise ValueError(
                    f"MXFP4 dequant target must end with '.weight', got: {weight_key}"
                )
            if weight_key not in state_dict:
                missing.append(weight_key)
                continue
            if state_dict[weight_key].dtype not in _UINT8_VIEW_COMPATIBLE:
                continue  # not MXFP4 (already float / FP8) — skip silently
            scale_key = f"{weight_key[:-len('.weight')]}.scale"
            if scale_key not in state_dict:
                continue
            weight_keys.append(weight_key)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
            raise KeyError(
                "MXFP4 dequant targeted weight(s) not loaded in state_dict: "
                f"{preview}{suffix}"
            )

    if not weight_keys:
        return 0

    total = len(weight_keys)
    progress_interval = max(1, total // 20)
    start = time.perf_counter()
    progress_print(
        f"[mxfp4-dequant] start: {total} packed weight(s) -> {output_dtype} "
        f"(compute on {compute_device or 'input device'})"
    )

    converted = 0
    for weight_key in weight_keys:
        base = weight_key[: -len(".weight")]
        scale_key = f"{base}.scale"
        tensor = dequantize_mxfp4(
            state_dict[weight_key],
            state_dict[scale_key],
            block_size=block_size,
            dtype=output_dtype,
            compute_device=compute_device,
        )
        state_dict[weight_key] = tensor.contiguous()
        state_dict.pop(scale_key, None)
        converted += 1
        if converted == total or converted % progress_interval == 0:
            elapsed = time.perf_counter() - start
            eta = elapsed / converted * (total - converted)
            progress_print(
                f"[mxfp4-dequant] {converted}/{total} ({converted / total:.0%}), "
                f"elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m"
            )

    if converted and compute_device is not None and str(compute_device).startswith("cuda"):
        # Return the transient dequant buffers to the allocator pool before
        # training-time allocations begin.
        torch.cuda.empty_cache()

    return converted
