"""Flash Attention 3 varlen (variable-length packed) interface for DINOv3 NaViT.

Packed API: (total_tokens, H, D) with cu_seqlens, unlike the batched API in
flash_attention.py which uses (B, N, H, D).

Note: implementation kept byte-for-byte compatible with the call form in
``lumos.transformer.attn_base.attention.flash_attention3`` (which is known
to work under torch.compile + activation checkpointing + FSDP2). No extra
kwargs with None defaults, no dynamic isinstance branches — both confuse
dynamo + autograd interaction in NO_REENTRANT recompute.
"""

import torch
from torch import Tensor

try:
    from flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:
    _flash_attn_varlen_func = None


def flash_attention3_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """Flash Attention 3 varlen for NaViT packed sequences (Hopper H100/H200).

    Args:
        q: (total_tokens, H, D)
        k: (total_tokens, H, D)
        v: (total_tokens, H, D)
        cu_seqlens: (n_docs + 1,) int32 — cumulative sequence lengths
        max_seqlen: maximum document length in the batch

    Returns:
        out: (total_tokens, H, D)
    """
    assert _flash_attn_varlen_func is not None, (
        "flash_attn_interface not installed. "
        "Install flash-attn-3 for Hopper GPUs: pip install flash-attn-3"
    )
    out: Tensor = _flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )
    return out
