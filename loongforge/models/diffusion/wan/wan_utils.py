# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import torch
from einops import rearrange
from megatron.core.parallel_state import get_context_parallel_group
from .communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from megatron.training.utils import print_rank_0


# ---------------------------------------------------------------------------
# Fused Triton RoPE
# ---------------------------------------------------------------------------
try:
    from .custom_ops import apply_rotary_interleaved
    _TRITON_ROPE_AVAILABLE = True
    print_rank_0("triton available")
except Exception as _e:
    _TRITON_ROPE_AVAILABLE = False
    print_rank_0("triton not available")


def wan_rope_apply(
    x,
    freqs,
    config,
    cu_seqlens=None,
    rotary_interleaved=False,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
):
    """wan rope apply — uses VeOmni fused Triton fp32 kernel when available"""
    heads = x.shape[2]

    need_cp_gather = (
        config.context_parallel_size > 1 and freqs is not None and freqs.shape[0] != x.shape[0]
    )
    if need_cp_gather:
        x = gather_forward_split_backward(
            x, get_context_parallel_group(), dim=0, grad_scale=None
        )
    x = rearrange(x, "s b n d -> b s n d")

    use_triton_rope = _TRITON_ROPE_AVAILABLE and getattr(config, "use_fused_wan_rope", False)
    if use_triton_rope:
        seq_len = x.shape[1]
        cos = freqs[:seq_len].real.squeeze(1).contiguous()
        sin = freqs[:seq_len].imag.squeeze(1).contiguous()
        x_out = apply_rotary_interleaved(x.contiguous(), cos, sin).flatten(2)
    else:
        x_out = torch.view_as_complex(
            x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
        )
        x_out = torch.view_as_real(x_out * freqs).flatten(2)

    if need_cp_gather:
        x_out = split_forward_gather_backward(
            x_out, get_context_parallel_group(), dim=1, grad_scale=None
        )
    x_out = rearrange(x_out, "b s (n d) -> s b n d", n=heads).to(x.dtype)
    # clone(contiguous_format) forces a real reallocation with canonical strides.
    return x_out.clone(memory_format=torch.contiguous_format)
