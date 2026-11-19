# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Default-off frozen-video RMS pointwise fusion; native FP32 reduction.

Input conversion/square and post-scale are fused separately. The original
native mean, epsilon addition and rsqrt remain unchanged. Normalization is
rounded to BF16 before the learned BF16 weight multiplication, as in WanRMSNorm.
"""
import atexit
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

_subjects = []  # Scalar health records only; never retain modules or tensors.
_spec = (('self_attn.norm_q', 1960), ('self_attn.norm_k', 1960),
         ('cross_attn.norm_q', 1960), ('cross_attn.norm_k', 129),
         ('cross_attn.norm_k_img', 257))


@triton.jit
def _square(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(S + i, x * x, i < N)


@triton.jit
def _post(X, INV, W, Y, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    inv = tl.load(INV + i // WIDTH, i < N, 0)
    w = tl.load(W + i % WIDTH, i < N, 0).to(tl.float32)
    normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, normalized * w, i < N)


def enable(expert):
    from .wan_video_dit import WanRMSNorm
    if expert.training or any(p.requires_grad for p in expert.parameters()):
        raise ValueError('Video RMS fusion requires an eval, frozen expert')
    norms = [(block.get_submodule(name), length) for block in expert.blocks for name, length in _spec]
    if not norms:
        raise ValueError('Video RMS fusion requires video blocks')
    for norm, length in norms:
        if (not isinstance(norm, WanRMSNorm) or norm.eps != 1e-6 or norm.weight.shape != (5120,)
                or hasattr(norm, '_perf_rms_pointwise')):
            raise ValueError('Unsupported or already configured video RMSNorm')
    record = dict(tagged_norms=len(norms), calls=0, video_calls=0, text_calls=0, image_calls=0,
        square='fp32-elementwise', statistics='native-fp32-mean-eps-rsqrt',
        rounding='bf16-normalized-then-bf16-weight-product', fallback='none-fail-closed')
    for norm, length in norms:
        norm._perf_rms_pointwise = record
        norm._perf_rms_length = length
    _subjects.append(record)


def forward(norm, x):
    weight = norm.weight
    length = norm._perf_rms_length
    shape_ok = (
        x.ndim == 3 and x.shape[-1] == 5120
        and x.shape[1] == length
    )
    if (torch.is_grad_enabled() or norm.training or torch.is_autocast_enabled()
            or not x.is_cuda or x.dtype != torch.bfloat16 or x.requires_grad
            or not shape_ok or not x.is_contiguous()
            or norm.eps != 1e-6 or weight.requires_grad or weight.dtype != torch.bfloat16
            or weight.device != x.device or weight.shape != (5120,) or not weight.is_contiguous()):
        raise ValueError('Video RMS fusion requires frozen no-grad contiguous CUDA BF16 actual shape/weight without autocast')
    piece = x.contiguous()
    squared = torch.empty_like(piece, dtype=torch.float32)
    _square[(triton.cdiv(piece.numel(), 512),)](
        piece, squared, piece.numel(), 512, num_warps=4, enable_fp_fusion=False
    )
    inv = torch.rsqrt(squared.mean(dim=-1, keepdim=True) + norm.eps)
    del squared
    y = torch.empty_like(x)
    _post[(triton.cdiv(piece.numel(), 512),)](
        piece, inv, weight, y, piece.numel(), 5120, 512,
        num_warps=4, enable_fp_fusion=False
    )
    record = norm._perf_rms_pointwise
    record['calls'] += 1
    record[{1960: 'video_calls', 129: 'text_calls', 257: 'image_calls'}[length]] += 1
    return y


def _health():
    root = os.environ.get('OUTPUT_DIR')
    if not root or not _subjects:
        return
    target = Path(root) / ('video_rms_health_rank' + str(os.environ.get('RANK', '0')) + '.json')
    target.write_text(json.dumps(dict(rank=int(os.environ.get('RANK', '0')),
        subjects=_subjects, scope='frozen video RMSNorm only'), indent=2) + '\n')


atexit.register(_health)
