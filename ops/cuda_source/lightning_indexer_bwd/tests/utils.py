"""Shared test utilities for lightning_indexer_bwd tests."""

import torch
from deep_gemm.utils import per_custom_dims_cast_to_fp8


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def ref_fp8_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                       cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor):
    """Pure-PyTorch reference for fp8_mqa_logits.

    Args:
        q:            [seq_len, num_heads, head_dim]  float
        kv:           [seq_len_kv, head_dim]           float
        weights:      [seq_len, num_heads]             float32
        cu_seqlen_ks: [seq_len]  int32  KV range start per query token
        cu_seqlen_ke: [seq_len]  int32  KV range end   per query token

    Returns:
        logits: [seq_len, seq_len_kv]  float32, -inf at masked positions
        cost:   scalar int, number of active (unmasked) token pairs
    """
    seq_len_kv = kv.shape[0]
    q = q.float()
    k = kv.float()

    mask_lo = torch.arange(0, seq_len_kv, device=q.device)[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device=q.device)[None, :] <  cu_seqlen_ke[:, None]
    mask = mask_lo & mask_hi

    # score: [num_heads, seq_len, seq_len_kv]
    score = torch.einsum('mhd,nd->hmn', q, k)
    # weighted sum over heads -> [seq_len, seq_len_kv]
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float('-inf')), mask.sum()


# ---------------------------------------------------------------------------
# KV-range helpers
# ---------------------------------------------------------------------------

def packed_q_to_k_ranges(cu_seqlens_q: torch.Tensor,
                         cu_seqlens_kv: torch.Tensor = None,
                         causal: bool = True):
    """Compute per-token KV [start, end) ranges for packed sequences.

    Args:
        cu_seqlens_q:  [num_seqs + 1] int32, cumulative query sequence lengths
        cu_seqlens_kv: [num_seqs + 1] int32, cumulative KV sequence lengths
                       (defaults to cu_seqlens_q for self-attention)
        causal:        if True, ke[i] = seq_kv_start + pos_in_seq + 1
                       (each token only sees its own past, not future tokens)

    Returns:
        ks: [total_tokens] int32  KV range start per query token
        ke: [total_tokens] int32  KV range end   per query token
    """
    if cu_seqlens_kv is None:
        cu_seqlens_kv = cu_seqlens_q

    total_q = int(cu_seqlens_q[-1])
    q_idx   = torch.arange(total_q, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype)
    q_seq   = torch.searchsorted(cu_seqlens_q[1:], q_idx, right=True)

    k_start = cu_seqlens_kv[q_seq]
    k_end   = cu_seqlens_kv[q_seq + 1]

    if causal:
        pos_in_seq = q_idx - cu_seqlens_q[q_seq]
        k_end = torch.minimum(k_start + pos_in_seq + 1, k_end)

    return k_start, k_end


# ---------------------------------------------------------------------------
# Test-data generators
# ---------------------------------------------------------------------------

def generate_cp_test_data(seq_len: int, seq_len_kv: int):
    """Generate KV ranges for Context Parallelism (CP) mode.

    Each query token attends to two disjoint sub-windows of the KV sequence
    corresponding to a specific CP rank's responsibility.

    Args:
        seq_len:    number of query tokens (must be even)
        seq_len_kv: total KV sequence length (must be a multiple of seq_len)

    Returns:
        ks: [seq_len] int32  KV range start per query token
        ke: [seq_len] int32  KV range end   per query token
    """
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size    = seq_len_kv // seq_len
    cp_id      = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    ke = torch.zeros(seq_len, dtype=torch.int, device='cuda')
    for i in range(chunk_size):
        ke[i]              = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def generate_sft_test_data(total_seq_len: int, num_heads: int, head_dim: int,
                           seq_lengths: list):
    """Generate packed-sequence (SFT) inputs with causal per-sequence masking.

    Multiple sequences are concatenated into a single packed buffer.  Each
    query token attends only to preceding tokens within its own sequence
    (cross-sequence attention is masked out).

    Args:
        total_seq_len: total tokens across all sequences; must equal
                       sum(seq_lengths) and be a multiple of 128 (BLOCK_KV)
        num_heads:     number of attention heads
        head_dim:      head dimension
        seq_lengths:   list of individual sequence lengths in the packed batch

    Returns:
        q, kv, weights  – raw BF16 tensors
        q_fp8, kv_fp8   – FP8-quantised versions
        ks, ke          – per-token KV ranges (causal within each sequence)
        cu_seqlens      – [num_seqs + 1] int32 cumulative lengths
    """
    assert sum(seq_lengths) == total_seq_len, \
        f'seq_lengths sum {sum(seq_lengths)} != total_seq_len {total_seq_len}'
    assert total_seq_len % 128 == 0, \
        f'total_seq_len {total_seq_len} must be 128-aligned (BLOCK_KV constraint)'

    q       = torch.randn(total_seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv      = torch.randn(total_seq_len, head_dim,            device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(total_seq_len, num_heads,            device='cuda', dtype=torch.float32)

    q_fp8  = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)

    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seq_lengths).cumsum(0).tolist()),
        dtype=torch.int32, device='cuda'
    )
    ks, ke = packed_q_to_k_ranges(cu_seqlens)

    return q, kv, weights, q_fp8, kv_fp8, ks, ke, cu_seqlens
