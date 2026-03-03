"""
flash mla interface
"""
from typing import Optional, Tuple

import torch

import flash_mla_fwd.cuda as flash_mla_cuda


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    q_start_index_s: int = 0,
    write_p_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        q_start_index_s: The starting position of the current chunk in the global sequence (used for causal masking)
        write_p_out: bool. Whether to write p_out to global memory.

    Returns:
        (output, max_logits, lse, p_out)
        About the definition of output, max_logits and lse, please refer to README.md
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, 2-based log-sum-exp
        - p_out: [s_q, h_q, topk], float32, probability（write_p_out=False 时为None）
    """
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, q_start_index_s, write_p_out
    )
    return results

