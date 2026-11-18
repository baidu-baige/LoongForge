# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Default-off frozen native BF16 LayerNorm with original FP32 statistics.

The validated native CUDA path retains FP32 Welford/statistics and output
rounding. Avoiding full FP32 input/output tensors does not lower statistics
precision. This is restricted to the exercised frozen video block path.
"""
import atexit
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

_subjects = []  # Scalar records only: never retain a model or tensor at exit.


def enable(expert):
    if expert.training or any(p.requires_grad for p in expert.parameters()):
        raise ValueError("Native video LayerNorm requires an eval, frozen expert")
    norms = [getattr(block, name) for block in expert.blocks for name in ("norm1", "norm2", "norm3")]
    if not norms:
        raise ValueError("Native video LayerNorm requires video blocks")
    for norm in norms:
        if (not isinstance(norm, torch.nn.LayerNorm) or norm.normalized_shape != (5120,)
                or norm.eps != 1e-6 or hasattr(norm, "_perf_native_bf16_ln")):
            raise ValueError("Unsupported or already configured video LayerNorm")
    record = {"tagged_norms": len(norms), "calls": 0, "affine_calls": 0,
              "nonaffine_calls": 0, "statistics": "native-fp32", "fallback": "none-fail-closed"}
    for norm in norms:
        norm._perf_native_bf16_ln = record
    _subjects.append(record)


def forward(norm, x):
    if (torch.is_grad_enabled() or norm.training or torch.is_autocast_enabled()
            or not x.is_cuda or x.dtype != torch.bfloat16 or x.requires_grad
            or x.ndim != 3 or tuple(x.shape[1:]) != (1960, 5120) or not x.is_contiguous()):
        raise ValueError("Native video LayerNorm requires frozen no-grad CUDA BF16 [B,1960,5120] without autocast")
    affine = norm.weight is not None
    if affine != (norm.bias is not None):
        raise ValueError("Unsupported video LayerNorm affine parameters")
    if affine and any(p.requires_grad or p.dtype != torch.bfloat16 or p.device != x.device
                      or not p.is_contiguous() for p in (norm.weight, norm.bias)):
        raise ValueError("Native video LayerNorm requires frozen BF16 affine parameters")
    result = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    record = norm._perf_native_bf16_ln
    record["calls"] += 1
    record["affine_calls" if affine else "nonaffine_calls"] += 1
    return result


def _health():
    root = os.environ.get("OUTPUT_DIR")
    if not root or not _subjects:
        return
    target = Path(root) / ("video_norm_health_rank" + str(os.environ.get("RANK", "0")) + ".json")
    target.write_text(json.dumps({"rank": int(os.environ.get("RANK", "0")),
        "subjects": _subjects, "scope": "frozen video block LayerNorm only"}, indent=2) + "\n")


atexit.register(_health)
