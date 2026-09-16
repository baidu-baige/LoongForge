"""Flash Attention 3 interface for DINOv3.

Batched fixed-length API (B, N, H, D), unlike attn_base's varlen API.
Supports causal masking for future causal video attention.
"""

import torch
from torch import Tensor

try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func
except ImportError:
    _flash_attn_func = None


def flash_attention3(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Tensor:
    """Flash Attention 3 for DINOv3 (Hopper H100/H200).

    Args:
        q: (B, N, H, D)
        k: (B, N, H, D)
        v: (B, N, H, D)
        causal: if True, apply causal (autoregressive) masking
        softmax_scale: defaults to 1/sqrt(D)

    Returns:
        out: (B, N, H, D)
    """
    assert _flash_attn_func is not None, (
        "flash_attn_interface not installed. "
        "Install flash-attn-3 for Hopper GPUs: pip install flash-attn-3"
    )
    out = _flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    return out
