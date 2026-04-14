"""Tests for sm100_fp8_mqa_logits forward kernel (via deep_gemm).

Covers three attention patterns:
  - Full-window (no CP):   each query attends to a fixed contiguous KV window
  - CP (Context Parallel): each query attends to two disjoint KV sub-windows
  - SFT (packed causal):   multiple sequences packed together; each query
                            attends only to preceding tokens within its own
                            sequence (cross-sequence attention is masked out)
"""

import torch
import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff,
    count_bytes,
    ignore_env, get_arch_major,
)

from utils import (
    ref_fp8_mqa_logits,
    packed_q_to_k_ranges,
    generate_cp_test_data,
    generate_sft_test_data,
)

# ---------------------------------------------------------------------------
# Peak TFLOPS lookup (dense, no sparsity).  Used for MFU calculation.
# Forward kernel uses FP8 TC; add entries as needed.
# ---------------------------------------------------------------------------

_PEAK_FP8_TFLOPS = {
    'B200': 9000.0,   # B200 SXM FP8 TC dense
    'B100': 7000.0,   # B100 SXM FP8 TC dense (approximate)
    'H200': 3958.0,   # H200 SXM FP8 TC dense
    'H100': 3958.0,   # H100 SXM 80GB FP8 TC dense
    'A100': 312.0,    # A100 SXM 80GB (no native FP8 TC; BF16 peak used as proxy)
}


def get_peak_fp8_tflops():
    """Return peak FP8 TC TFLOPS for the current device, or None if unknown."""
    name = torch.cuda.get_device_name(0).upper()
    for key, val in _PEAK_FP8_TFLOPS.items():
        if key in name:
            return val
    return _PEAK_FP8_TFLOPS['B200']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(seq_len, seq_len_kv, num_heads, head_dim, disable_cp):
    from deep_gemm.utils import per_custom_dims_cast_to_fp8
    q       = torch.randn(seq_len,    num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv      = torch.randn(seq_len_kv, head_dim,            device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(seq_len,    num_heads,            device='cuda', dtype=torch.float32)
    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
        ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)
    else:
        ks, ke = generate_cp_test_data(seq_len, seq_len_kv)
    q_fp8  = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
    return q, kv, weights, q_fp8, kv_fp8, ks, ke


def run_kernel(q_fp8, kv_fp8, weights, ks, ke, compressed_logits):
    """Run the kernel and return full [seq_len, seq_len_kv] logits."""
    seq_len    = q_fp8.shape[0]
    seq_len_kv = kv_fp8[0].shape[0]
    if compressed_logits:
        max_seqlen_k = int((ke - ks).max().item())
        out = deep_gemm.fp8_mqa_logits(
            q_fp8, kv_fp8, weights, ks, ke,
            max_seqlen_k=max_seqlen_k, clean_logits=False
        )
        assert out.size() == (seq_len, max_seqlen_k), \
            f'shape mismatch: {out.size()} vs ({seq_len}, {max_seqlen_k})'
        # Expand back to full shape for accuracy comparison
        full = torch.full((seq_len, seq_len_kv), float('-inf'), device='cuda')
        for i in range(seq_len):
            full[i, ks[i]:ke[i]] = out[i, :ke[i] - ks[i]]
        return full
    else:
        return deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)


# ---------------------------------------------------------------------------
# Accuracy test
# ---------------------------------------------------------------------------

@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_accuracy():
    print('Testing FP8 MQA Logits accuracy:')
    head_dim = 128

    for num_heads in (64, 32):
        for seq_len in (2048, 4096):
            for seq_len_kv in (4096, 8192):
                for disable_cp in (False, True):
                    for compressed_logits in (False, True):
                        q, kv, weights, q_fp8, kv_fp8, ks, ke = make_inputs(
                            seq_len, seq_len_kv, num_heads, head_dim, disable_cp
                        )
                        logits        = run_kernel(q_fp8, kv_fp8, weights, ks, ke, compressed_logits)
                        ref_logits, _ = ref_fp8_mqa_logits(q, kv, weights, ks, ke)

                        ref_neginf = (ref_logits == float('-inf'))
                        neginf     = (logits     == float('-inf'))
                        assert torch.equal(neginf, ref_neginf), \
                            f'neginf mask mismatch: {num_heads=}, {seq_len=}, {seq_len_kv=}, ' \
                            f'cp={not disable_cp}, {compressed_logits=}'

                        diff = calc_diff(logits.masked_fill(neginf, 0),
                                         ref_logits.masked_fill(ref_neginf, 0))
                        assert diff < 1e-3, \
                            f'diff={diff:.5f}, {num_heads=}, {seq_len=}, {seq_len_kv=}, ' \
                            f'cp={not disable_cp}, {compressed_logits=}'

                        cp_flag = 0 if disable_cp else 1
                        print(f' > H={num_heads:2}, S={seq_len:4}, SKV={seq_len_kv:5}, CP={cp_flag}, '
                              f'compressed={int(compressed_logits)}: diff={diff:.2e}  OK')
    print()


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------

@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_perf():
    print('Testing FP8 MQA Logits performance:')
    head_dim  = 128
    peak_fp8  = get_peak_fp8_tflops()
    peak_info = f'(peak FP8 TC = {peak_fp8:.0f} TFLOPS)' if peak_fp8 else '(peak unknown)'
    print(f'  Device: {torch.cuda.get_device_name(0)}  {peak_info}')

    for num_heads in (64, 32):
        for seq_len in (2048, 4096):
            for seq_len_kv in (4096, 8192):
                for disable_cp in (False, True):
                    for compressed_logits in (False, True):
                        _, _, _, q_fp8, kv_fp8, ks, ke = make_inputs(
                            seq_len, seq_len_kv, num_heads, head_dim, disable_cp
                        )
                        weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
                        _, ref_cost = ref_fp8_mqa_logits(
                            q_fp8.float(), kv_fp8[0].float(), weights, ks, ke
                        )
                        tflops      = 2 * ref_cost * num_heads * head_dim / 1e12
                        input_bytes = count_bytes(q_fp8, kv_fp8, weights, ks, ke) + ref_cost * 4
                        cp_flag     = 0 if disable_cp else 1

                        if compressed_logits:
                            max_seqlen_k = int((ke - ks).max().item())
                            t = bench_kineto(
                                lambda: deep_gemm.fp8_mqa_logits(
                                    q_fp8, kv_fp8, weights, ks, ke,
                                    max_seqlen_k=max_seqlen_k, clean_logits=False
                                ),
                                'fp8_mqa_logits'
                            )
                            achieved = tflops / t
                            mfu_str  = f'{achieved / peak_fp8 * 100:4.1f}%' if peak_fp8 else '   -'
                            print(f' > H={num_heads:2}, S={seq_len:4}, SKV={seq_len_kv:5}, CP={cp_flag}, compressed=1: '
                                  f'{achieved:4.0f} TFLOPS, {t * 1e6:4.0f} us, '
                                  f'{input_bytes / t / 1e9:4.0f} GB/s, MFU={mfu_str}')
                        else:
                            t, clean_t = bench_kineto(
                                lambda: deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke),
                                ('fp8_mqa_logits', 'clean_logits')
                            )
                            clean_bytes = (seq_len * seq_len_kv - ref_cost) * 4 + count_bytes(ks, ke)
                            achieved    = tflops / t
                            mfu_str     = f'{achieved / peak_fp8 * 100:4.1f}%' if peak_fp8 else '   -'
                            print(f' > H={num_heads:2}, S={seq_len:4}, SKV={seq_len_kv:5}, CP={cp_flag}, compressed=0: '
                                  f'{achieved:4.0f} TFLOPS, {t * 1e6:4.0f} us, '
                                  f'{input_bytes / t / 1e9:4.0f} GB/s, MFU={mfu_str}'
                                  f' | clean: {clean_t * 1e6:3.0f} us, '
                                  f'{clean_bytes / clean_t / 1e9:4.0f} GB/s')
    print()


# ---------------------------------------------------------------------------
# SFT accuracy test
# ---------------------------------------------------------------------------

# Packed-sequence configurations: (total_tokens, [per-sequence lengths])
# All totals must be 128-aligned.
_SFT_CONFIGS = [
    # Many short sequences: simulates a typical SFT micro-batch
    (4096,  [512,  768, 1024,  512,  640,  640]),
    # Fewer but longer sequences
    (4096,  [1024, 1024, 1024, 1024]),
    # Uneven lengths — stress-tests boundary handling
    (4224,  [128,  384,  512, 1024,  768,  512,  384,  512]),
    # Single long sequence — degenerate case (identical to causal LM)
    (4096,  [4096]),
]


@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_accuracy_sft():
    print('Testing FP8 MQA Logits accuracy (SFT / packed-causal):')
    head_dim = 128

    for num_heads in (64, 32):
        for total_len, seq_lengths in _SFT_CONFIGS:
            q, kv, weights, q_fp8, kv_fp8, ks, ke, _ = generate_sft_test_data(
                total_len, num_heads, head_dim, seq_lengths
            )

            logits        = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
            ref_logits, _ = ref_fp8_mqa_logits(q, kv, weights, ks, ke)

            ref_neginf = (ref_logits == float('-inf'))
            neginf     = (logits     == float('-inf'))
            assert torch.equal(neginf, ref_neginf), \
                f'neginf mask mismatch: {num_heads=}, total_len={total_len}, seqs={seq_lengths}'

            diff = calc_diff(logits.masked_fill(neginf, 0),
                             ref_logits.masked_fill(ref_neginf, 0))
            assert diff < 1e-3, \
                f'diff={diff:.5f}, {num_heads=}, total_len={total_len}, seqs={seq_lengths}'

            num_seqs    = len(seq_lengths)
            active_frac = (~ref_neginf).float().mean().item()
            print(f' > H={num_heads:2}, total={total_len:4}, num_seqs={num_seqs}, '
                  f'lens={seq_lengths}: '
                  f'active={active_frac:.2%}, diff={diff:.2e}  OK')
    print()


# ---------------------------------------------------------------------------
# SFT performance test
# ---------------------------------------------------------------------------

# Larger configs for meaningful perf numbers; still 128-aligned.
_SFT_PERF_CONFIGS = [
    (4096,  [512,  768, 1024,  512,  640,  640]),
    (4096,  [1024, 1024, 1024, 1024]),
    (8192,  [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]),
    (8192,  [2048, 2048, 2048, 2048]),
]


@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_perf_sft():
    print('Testing FP8 MQA Logits performance (SFT / packed-causal):')
    head_dim  = 128
    peak_fp8  = get_peak_fp8_tflops()
    peak_info = f'(peak FP8 TC = {peak_fp8:.0f} TFLOPS)' if peak_fp8 else '(peak unknown)'
    print(f'  Device: {torch.cuda.get_device_name(0)}  {peak_info}')

    for num_heads in (64, 32):
        for total_len, seq_lengths in _SFT_PERF_CONFIGS:
            _, _, _, q_fp8, kv_fp8, ks, ke, _ = generate_sft_test_data(
                total_len, num_heads, head_dim, seq_lengths
            )
            weights = torch.randn(total_len, num_heads, device='cuda', dtype=torch.float32)

            _, ref_cost = ref_fp8_mqa_logits(
                q_fp8.float(), kv_fp8[0].float(), weights, ks, ke
            )
            tflops      = 2 * ref_cost * num_heads * head_dim / 1e12
            input_bytes = count_bytes(q_fp8, kv_fp8, weights, ks, ke) + ref_cost * 4

            t, clean_t = bench_kineto(
                lambda: deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke),
                ('fp8_mqa_logits', 'clean_logits')
            )
            clean_bytes = (total_len * total_len - ref_cost) * 4 + count_bytes(ks, ke)
            num_seqs    = len(seq_lengths)
            achieved    = tflops / t
            mfu_str     = f'{achieved / peak_fp8 * 100:4.1f}%' if peak_fp8 else '   -'
            print(f' > H={num_heads:2}, total={total_len:4}, num_seqs={num_seqs}: '
                  f'{achieved:4.0f} TFLOPS, {t * 1e6:4.0f} us, '
                  f'{input_bytes / t / 1e9:4.0f} GB/s, MFU={mfu_str}'
                  f' | clean: {clean_t * 1e6:3.0f} us, '
                  f'{clean_bytes / clean_t / 1e9:4.0f} GB/s')
    print()


# ---------------------------------------------------------------------------
# End-to-end scenario performance test
# (seq_len, seq_len_kv) matching real production workloads.
# Uses compressed_logits=True (production path) and CP ranges.
# ---------------------------------------------------------------------------

_E2E_CONFIGS = [
    # (seq_len, seq_len_kv)
    (8192,  65536),
    (16384, 131072),
]


@ignore_env('DG_JIT_PTXAS_CHECK', lambda: get_arch_major() == 10)
def test_perf_e2e():
    """Benchmark fp8_mqa_logits on end-to-end production scenarios."""
    print('Testing FP8 MQA Logits performance (end-to-end scenarios):')
    head_dim  = 128
    peak_fp8  = get_peak_fp8_tflops()
    peak_info = f'(peak FP8 TC = {peak_fp8:.0f} TFLOPS)' if peak_fp8 else '(peak unknown)'
    print(f'  Device: {torch.cuda.get_device_name(0)}  {peak_info}')

    for num_heads in (64, 32):
        for seq_len, seq_len_kv in _E2E_CONFIGS:
            _, _, _, q_fp8, kv_fp8, ks, ke = make_inputs(
                seq_len, seq_len_kv, num_heads, head_dim, disable_cp=False
            )
            weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)

            # Compute ref_cost directly to avoid OOM from full [seq_len, seq_len_kv] allocation
            ref_cost     = int((ke - ks).sum().item())
            tflops       = 2 * ref_cost * num_heads * head_dim / 1e12
            input_bytes  = count_bytes(q_fp8, kv_fp8, weights, ks, ke) + ref_cost * 4
            max_seqlen_k = int((ke - ks).max().item())

            t = bench_kineto(
                lambda: deep_gemm.fp8_mqa_logits(
                    q_fp8, kv_fp8, weights, ks, ke,
                    max_seqlen_k=max_seqlen_k, clean_logits=False
                ),
                'fp8_mqa_logits'
            )

            achieved = tflops / t
            mfu_str  = f'{achieved / peak_fp8 * 100:4.1f}%' if peak_fp8 else '   -'
            print(f' > H={num_heads:2}, S={seq_len:5}, SKV={seq_len_kv:6}, compressed=1: '
                  f'{achieved:4.0f} TFLOPS, {t * 1e6:6.0f} us, '
                  f'{input_bytes / t / 1e9:4.0f} GB/s, MFU={mfu_str}')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_accuracy()
    test_perf()
    test_accuracy_sft()
    test_perf_sft()
    test_perf_e2e()
