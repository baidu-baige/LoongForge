"""
flash mla backward interface
"""
from typing import Optional, Tuple

import torch

import flash_mla_bwd.cuda as flash_mla_cuda


def flash_mla_sparse_bwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    dO: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: Optional[float] = None,
    d_v: int = 512,
    topk_length: Optional[torch.Tensor] = None,
    q_start_index_s: int = 0,
    fast_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse attention backward kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16 - Query tensor
        kv: [s_kv, h_kv, d_qk], bfloat16 - Key/Value tensor
        o: [s_q, h_q, d_v], bfloat16 - Forward output
        dO: [s_q, h_q, d_v], bfloat16 - Output gradient
        indices: [s_q, h_kv, topk], int32 - TopK indices
        lse: [s_q, h_q], float32 - Log-Sum-Exp (from forward)
        sm_scale: float - Softmax scaling factor
        d_v: int - Value dimension, must be 512
        topk_length: optional, [s_q], int32 - Optional TopK length
        q_start_index_s: The starting position of the current chunk in the global sequence (used for causal masking)
        fast_mode: bool - If True, use fused kernel path (only for h_q=128)

    Returns:
        (dQ, dKV)
        - dQ: [s_q, h_q, d_qk], bfloat16 - Query gradient
        - dKV: [s_kv, h_kv, d_qk], bfloat16 - KV gradient
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** (-0.5)

    results = flash_mla_cuda.sparse_prefill_bwd(
        q, kv, o, dO, indices, lse, sm_scale, d_v, topk_length, q_start_index_s, fast_mode
    )
    return results