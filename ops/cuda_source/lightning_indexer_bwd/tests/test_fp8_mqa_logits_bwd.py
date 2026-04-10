"""Lightning Indexer Backward Kernel Tests"""

import torch
import lightning_indexer_bwd
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes, ignore_env, get_arch_major
from deep_gemm.utils import per_custom_dims_cast_to_fp8

from utils import ref_fp8_mqa_logits, packed_q_to_k_ranges


# ---------------------------------------------------------------------------
# Peak BF16 Tensor Core TFLOPS lookup (dense, no sparsity)
# Used for MFU calculation.  Add entries as needed.
# ---------------------------------------------------------------------------

_PEAK_BF16_TFLOPS = {
    'B200': 4500.0,   # B200 SXM BF16 TC dense
    'B100': 3500.0,   # B100 SXM BF16 TC dense (approximate)
    'H200': 1979.0,   # H200 SXM BF16 TC dense
    'H100': 1979.0,   # H100 SXM 80GB BF16 TC dense
    'A100': 312.0,    # A100 SXM 80GB BF16 TC dense
}


def get_peak_bf16_tflops():
    """Return peak BF16 TC TFLOPS for the current device, or None if unknown."""
    name = torch.cuda.get_device_name(0).upper()
    for key, val in _PEAK_BF16_TFLOPS.items():
        if key in name:
            return val
    return _PEAK_BF16_TFLOPS['B200']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bwd_inputs(seq_len, kv_seq_len, num_heads, head_dim, topk):
    """Create all inputs needed to run fp8_mqa_logits_bwd."""
    torch.manual_seed(42)
    q_fp8    = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    kv_fp8   = per_custom_dims_cast_to_fp8(
        torch.randn(kv_seq_len, head_dim, device='cuda', dtype=torch.bfloat16), (0,), False
    )
    softmax_scale = head_dim ** -0.5
    weights  = torch.randn(seq_len, num_heads, device='cuda') * softmax_scale
    d_output = torch.ones(seq_len, topk, device='cuda')

    # 5 packed sequences summing to seq_len
    boundaries = [0, seq_len // 8, seq_len // 4, seq_len // 2, seq_len * 3 // 4, seq_len]
    cu_seqlens_q = torch.tensor(boundaries, device='cuda', dtype=torch.int32)
    ks, ke = packed_q_to_k_ranges(cu_seqlens_q)

    topk_indices = torch.full((seq_len, topk), kv_seq_len - 1, dtype=torch.int32, device='cuda')
    for t in range(seq_len):
        window = max(1, int(ke[t].item()) - int(ks[t].item()))
        idx = (torch.randperm(window, device='cuda') + int(ks[t].item()))[:topk]
        topk_indices[t, :len(idx)] = idx

    return q_fp8, kv_fp8, weights, d_output, ks, ke, topk_indices


# ---------------------------------------------------------------------------
# Accuracy test
# ---------------------------------------------------------------------------

@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_accuracy():
    """Test backward pass accuracy against reference implementation."""
    print('Testing FP8 MQA Logits Backward accuracy:')
    head_dim = 128
    topk     = 1024
    seq_len  = 1024 * 4

    for num_heads in (64, 32):
        for kv_seq_len in (1024 * 4, 1024 * 8):
            q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(kv_seq_len, head_dim,        device='cuda', dtype=torch.bfloat16, requires_grad=True)
            weights_raw = (torch.randn(seq_len, num_heads, device='cuda') * 10 + 10).requires_grad_(True)

            q_fp8  = q.to(torch.float8_e4m3fn)
            kv_fp8 = per_custom_dims_cast_to_fp8(k, (0,), False)
            softmax_scale = head_dim ** -0.5
            dg_w   = weights_raw * softmax_scale

            boundaries   = [0, seq_len // 8, seq_len // 4, seq_len // 2, seq_len * 3 // 4, seq_len]
            cu_seqlens_q = torch.tensor(boundaries, device='cuda', dtype=torch.int32)
            ks, ke       = packed_q_to_k_ranges(cu_seqlens_q)

            topk_indices = torch.full((seq_len, topk), kv_seq_len - 1, dtype=torch.int32, device='cuda')
            for t in range(seq_len):
                window = max(1, int(ke[t].item()) - int(ks[t].item()))
                idx = (torch.randperm(window, device='cuda') + int(ks[t].item()))[:topk]
                topk_indices[t, :len(idx)] = idx

            # Reference backward
            ref_logits, _ = ref_fp8_mqa_logits(q, k, dg_w, ks, ke)
            loss = torch.gather(ref_logits, dim=-1, index=topk_indices.to(torch.int64)).sum()
            loss.backward()
            ref_dq = q.grad.clone(); ref_dk = k.grad.clone(); ref_dw = weights_raw.grad.clone()
            q.grad = k.grad = weights_raw.grad = None

            # Kernel backward
            d_output = torch.ones(seq_len, topk, device='cuda')
            d_q, d_k, d_w = lightning_indexer_bwd.fp8_mqa_logits_bwd(
                d_output, q_fp8, kv_fp8, dg_w, ks, ke, topk_indices=topk_indices, topk=topk
            )
            d_k = torch.einsum('nd,n->nd', d_k, 1 / kv_fp8[1])
            d_w = d_w * softmax_scale

            dq_diff = calc_diff(d_q, ref_dq)
            dk_diff = calc_diff(d_k, ref_dk)
            dw_diff = calc_diff(d_w, ref_dw)
            assert dq_diff < 0.01, f'dq diff too large: {dq_diff:.4f}'
            assert dk_diff < 0.01, f'dk diff too large: {dk_diff:.4f}'
            assert dw_diff < 1e-3, f'dw diff too large: {dw_diff:.2e}'
            print(f' > H={num_heads:2}, S={seq_len:4}, SKV={kv_seq_len:5}: '
                  f'dq={dq_diff:.2e}  dk={dk_diff:.2e}  dw={dw_diff:.2e}  OK')
    print()


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------

@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_perf():
    """Benchmark fp8_mqa_logits_bwd kernel."""
    print('Testing FP8 MQA Logits Backward performance:')
    head_dim  = 128
    peak_bf16 = get_peak_bf16_tflops()
    peak_info = f'(peak BF16 TC = {peak_bf16:.0f} TFLOPS)' if peak_bf16 else '(peak unknown)'
    print(f'  Device: {torch.cuda.get_device_name(0)}  {peak_info}')

    for num_heads in (64, 32):
        for seq_len in (2048, 4096):
            for kv_seq_len in (4096, 8192):
                topk = min(kv_seq_len, 1024)
                q_fp8, kv_fp8, weights, d_output, ks, ke, topk_indices = make_bwd_inputs(
                    seq_len, kv_seq_len, num_heads, head_dim, topk
                )

                # FLOP count: actual active (q, kv) pairs, accounting for causal mask via ks/ke
                #   FP8  (recompute logit): 2 * active_pairs * num_heads * head_dim
                #   BF16 (compute grad):    4 * active_pairs * num_heads * head_dim
                active_pairs = (
                    (ke.clamp(min=0, max=kv_seq_len) - ks.clamp(min=0, max=kv_seq_len))
                    .clamp(min=0, max=topk).sum().item()
                )
                tflops_fp8  = 2 * active_pairs * num_heads * head_dim / 1e12
                tflops_bf16 = 4 * active_pairs * num_heads * head_dim / 1e12

                # Bandwidth: read (q_fp8, kv_fp8, weights, d_output, topk_indices)
                #            write (d_q, d_k, d_w — atomicAdd, approximate)
                read_bytes  = (count_bytes(q_fp8, kv_fp8, weights, d_output, topk_indices))
                write_bytes = (seq_len * num_heads * head_dim * 4      # d_q  float32
                             + kv_seq_len * head_dim * 4                # d_kv float32 (full, atomic)
                             + seq_len * num_heads * 4)                 # d_w  float32
                io_bytes = read_bytes + write_bytes

                t = bench_kineto(
                    lambda: lightning_indexer_bwd.fp8_mqa_logits_bwd(
                        d_output, q_fp8, kv_fp8, weights, ks, ke,
                        topk_indices=topk_indices, topk=topk
                    ),
                    'fp8_mqa_logits_bwd'
                )

                throughput_fp8  = tflops_fp8  / t
                throughput_bf16 = tflops_bf16 / t
                mfu_fp8  = throughput_fp8  / peak_bf16        if peak_bf16 else None
                mfu_bf16 = throughput_bf16 / (peak_bf16 / 2)  if peak_bf16 else None
                print(f' > H={num_heads:2}, S={seq_len:4}, SKV={kv_seq_len:5}, topk={topk:4}: '
                      f'{t * 1e6:5.0f} us, {io_bytes / t / 1e9:5.0f} GB/s')
                print(f'   FP8(recompute logit): {throughput_fp8:5.0f} TFLOPS, '
                      f'MFU: {mfu_fp8 * 100:4.1f}%' if mfu_fp8 is not None else
                      f'   FP8(recompute logit): {throughput_fp8:5.0f} TFLOPS')
                print(f'   BF16(compute grad):   {throughput_bf16:5.0f} TFLOPS, '
                      f'MFU: {mfu_bf16 * 100:4.1f}%' if mfu_bf16 is not None else
                      f'   BF16(compute grad):   {throughput_bf16:5.0f} TFLOPS')
    print()


# ---------------------------------------------------------------------------
# End-to-end scenario performance test
# (seq_len, kv_seq_len, topk) matching real production workloads
# ---------------------------------------------------------------------------

_E2E_CONFIGS = [
    # (seq_len, kv_seq_len, topk)
    (8192,  65536,  2048),
    (16384, 131072, 2048),
]


@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_perf_e2e():
    """Benchmark fp8_mqa_logits_bwd on end-to-end production scenarios."""
    print('Testing FP8 MQA Logits Backward performance (end-to-end scenarios):')
    head_dim  = 128
    peak_bf16 = get_peak_bf16_tflops()
    peak_info = f'(peak BF16 TC = {peak_bf16:.0f} TFLOPS)' if peak_bf16 else '(peak unknown)'
    print(f'  Device: {torch.cuda.get_device_name(0)}  {peak_info}')

    for num_heads in (64, 32):
        for seq_len, kv_seq_len, topk in _E2E_CONFIGS:
            q_fp8, kv_fp8, weights, d_output, ks, ke, topk_indices = make_bwd_inputs(
                seq_len, kv_seq_len, num_heads, head_dim, topk
            )

            active_pairs = (
                (ke.clamp(min=0, max=kv_seq_len) - ks.clamp(min=0, max=kv_seq_len))
                .clamp(min=0, max=topk).sum().item()
            )
            tflops_fp8  = 2 * active_pairs * num_heads * head_dim / 1e12
            tflops_bf16 = 4 * active_pairs * num_heads * head_dim / 1e12

            read_bytes  = count_bytes(q_fp8, kv_fp8, weights, d_output, topk_indices)
            write_bytes = (seq_len * num_heads * head_dim * 4
                         + kv_seq_len * head_dim * 4
                         + seq_len * num_heads * 4)
            io_bytes = read_bytes + write_bytes

            t = bench_kineto(
                lambda: lightning_indexer_bwd.fp8_mqa_logits_bwd(
                    d_output, q_fp8, kv_fp8, weights, ks, ke,
                    topk_indices=topk_indices, topk=topk
                ),
                'fp8_mqa_logits_bwd'
            )

            throughput_fp8  = tflops_fp8  / t
            throughput_bf16 = tflops_bf16 / t
            mfu_fp8  = throughput_fp8  / peak_bf16        if peak_bf16 else None
            mfu_bf16 = throughput_bf16 / (peak_bf16 / 2)  if peak_bf16 else None
            print(f' > H={num_heads:2}, S={seq_len:5}, SKV={kv_seq_len:6}, topk={topk}: '
                  f'{t * 1e6:6.0f} us, {io_bytes / t / 1e9:5.0f} GB/s')
            print(f'   FP8(recompute logit): {throughput_fp8:5.0f} TFLOPS, '
                  f'MFU: {mfu_fp8 * 100:4.1f}%' if mfu_fp8 is not None else
                  f'   FP8(recompute logit): {throughput_fp8:5.0f} TFLOPS')
            print(f'   BF16(compute grad):   {throughput_bf16:5.0f} TFLOPS, '
                  f'MFU: {mfu_bf16 * 100:4.1f}%' if mfu_bf16 is not None else
                  f'   BF16(compute grad):   {throughput_bf16:5.0f} TFLOPS')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_accuracy()
    test_perf()
    test_perf_e2e()
