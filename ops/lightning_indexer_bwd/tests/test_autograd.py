"""Lightning Indexer Backward Kernel Tests"""

import torch
import deep_gemm
import lightning_indexer_bwd
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_custom_dims_cast_to_fp8


def ref_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Reference implementation of FP8 MQA logits forward pass."""
    seq_len_kv = kv.shape[0]
    q, k = q.float(), kv.float()

    mask_lo = torch.arange(seq_len_kv, device='cuda')[None, :] >= cu_seqlen_ks[:, None]
    mask_hi = torch.arange(seq_len_kv, device='cuda')[None, :] < cu_seqlen_ke[:, None]

    score = torch.einsum('mhd,nd->hmn', q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~(mask_lo & mask_hi), float('-inf'))


def packed_q_to_k_ranges(cu_seqlens_q, cu_seqlens_kv=None, causal=True):
    """Compute KV start/end indices for each query token."""
    if cu_seqlens_kv is None:
        cu_seqlens_kv = cu_seqlens_q

    total_q = int(cu_seqlens_q[-1])
    q_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype)
    q_seq = torch.searchsorted(cu_seqlens_q[1:], q_idx, right=True)

    k_start = cu_seqlens_kv[q_seq]
    k_end = cu_seqlens_kv[q_seq + 1]

    if causal:
        q_seq_start = cu_seqlens_q[q_seq]
        pos_in_seq = q_idx - q_seq_start
        k_end = torch.minimum(k_start + pos_in_seq + 1, k_end)

    return k_start, k_end


def bwd_accuracy_test(
    num_heads=64,
    head_dim=128,
    topk=2048,
    seq_len=1024 * 8,
    kv_seq_len=1024 * 8,
):
    """Test backward pass accuracy against reference implementation."""
    torch.manual_seed(42)

    # Prepare test data
    q = torch.randn(1, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_seq_len, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weights = (torch.randn(seq_len, 1, num_heads, device="cuda") * 10 + 10).requires_grad_(True)

    # FP8 conversion
    q_fp8 = q.squeeze(0).to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(k.squeeze(0), (0,), False)
    softmax_scale = head_dim ** -0.5
    dg_w = weights.squeeze(1) * softmax_scale

    # KV ranges
    cu_seqlens_q = torch.tensor([0, 1028, 1533, 3820, 4860, 32 * 256], device='cuda', dtype=torch.int32)
    ks, ke = packed_q_to_k_ranges(cu_seqlens_q)

    # Generate topk indices
    topk_indices = torch.full((seq_len, topk), kv_seq_len - 1, dtype=torch.int32, device="cuda")
    for t in range(seq_len):
        indices = (torch.randperm(max(1, ke[t] - ks[t]), device='cuda') + ks[t])[:topk]
        topk_indices[t, : len(indices)] = indices

    # Reference backward
    ref_logits = torch.compile(ref_fp8_mqa_logits)(q.squeeze(0), k.squeeze(0), dg_w, ks, ke)
    loss = torch.gather(ref_logits, dim=-1, index=topk_indices.to(torch.int64)).sum()
    loss.backward()

    ref_dq = q.grad.clone().squeeze(0)
    ref_dk = k.grad.clone().squeeze(0)
    ref_dw = weights.grad.clone().squeeze(1)
    q.grad = k.grad = weights.grad = None

    # Lightning indexer backward
    d_output = torch.ones(seq_len, topk, device='cuda')
    d_q, d_k, d_w = lightning_indexer_bwd.fp8_mqa_logits_bwd(
        d_output, q_fp8, kv_fp8, dg_w, ks, ke, topk_indices=topk_indices, topk=topk
    )
    d_k = torch.einsum('nd,n->nd', d_k, 1 / kv_fp8[1])
    d_w = d_w * softmax_scale

    # Compare results
    print(f"> diff(dq): {calc_diff(d_q, ref_dq)}")
    print(f"> diff(dk): {calc_diff(d_k, ref_dk)}")
    print(f"> diff(dw): {calc_diff(d_w, ref_dw)}")


if __name__ == "__main__":
    bwd_accuracy_test()
