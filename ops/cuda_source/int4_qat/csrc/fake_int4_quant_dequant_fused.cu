/*
 * Fused INT4 fake quantize-dequantize CUDA kernel.
 *
 * Performs symmetric INT4 quantization and dequantization in a single GPU pass,
 * eliminating the intermediate q/scale DRAM round-trip of the two-pass approach.
 *
 * Optimized for block_size=[1, 32] (group_size=32) with BF16 MoE expert weights.
 * Uses multi-row-per-thread coarsening (ROWS_PER_THREAD=4) to hide warp shuffle
 * latency by interleaving loads across rows.
 *
 * Performance (B200, fc1 [262144, 7168], BF16, group_size=32):
 *   Two-pass baseline:  9.79 ms
 *   This fused kernel:  2.90 ms  (3.38x speedup)
 *
 * Correctness: bit-exact match with the two-pass pipeline. Scale is truncated to
 * BF16 precision before the dequant multiply to replicate the DRAM BF16 round-trip.
 *
 * NOTE: Do NOT compile with --use_fast_math — it changes division rounding and
 * breaks the bit-exact match with the original quantize kernel.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#define FINAL_MASK 0xFFFFFFFF

__device__ __forceinline__
int ceil_div_d(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__
float warpReduceMax(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

/* Truncate FP32 → BF16 → FP32 in registers.
 *
 * The original two-pass pipeline writes scale to DRAM as BF16, then reads it back
 * for the dequant multiply.  This truncation replicates that precision loss so the
 * fused kernel produces bit-identical results. */
__device__ __forceinline__
float bf16_trunc(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

/* Fused symmetric INT4 quantize-dequantize for one element.
 *
 *   qval  = rintf(val / scale)          — same FP32 division as original kernel
 *   scale = bf16_trunc(scale)           — replicate BF16 DRAM truncation
 *   out   = qval * scale_bf16           — dequantize in FP32, caller casts to BF16
 *
 * qval is an integer in [-7, 7] which is exact in BF16, so no truncation needed. */
__device__ __forceinline__
float fused_qd_sym(float val, float scale) {
    float qval = rintf(val / scale);
    float scale_bf16 = bf16_trunc(scale);
    return qval * scale_bf16;
}

/* ---------------------------------------------------------------------------
 * Main kernel: fused INT4 fake quant-dequant with multi-row coarsening
 * ---------------------------------------------------------------------------
 * Each block covers ROWS_PER_THREAD consecutive rows.  Each warp iterates over
 * groups of 32 columns, processing all ROWS_PER_THREAD rows per group before
 * advancing.  This layout has two benefits:
 *
 *   1. Loads from row N+1 can overlap with the warp-shuffle reduction of row N,
 *      hiding the 5-stage serial shuffle latency.
 *   2. Column address computation is reused across rows (29% fewer INT ops).
 *
 * Launch config: grid = ceil(M / ROWS_PER_THREAD), block = 1024 (32 warps).
 * Best ROWS_PER_THREAD=4 on B200 (28 regs/thread, 95% warp occupancy).
 */
template<typename scalar_t, int ROWS_PER_THREAD>
__global__
void fused_int4_qd_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const int M, const int N
) {
    constexpr int WARPS_PER_BLOCK = 32;
    constexpr float SYM_CONS = 1.0f / 7.0f;

    const int needed_warps = ceil_div_d(N, 32);
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    const int base_row = blockIdx.x * ROWS_PER_THREAD;

    for (int item = warp_id; item < needed_warps; item += WARPS_PER_BLOCK) {
        const int col = item * 32 + lane_id;

        /* Load ROWS_PER_THREAD values — interleaved loads hide memory latency. */
        float vals[ROWS_PER_THREAD];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            const int row = base_row + r;
            if (row < M && col < N) {
                vals[r] = static_cast<float>(x[row * N + col]);
            } else {
                vals[r] = 0.0f;
            }
        }

        /* Quant-dequant each row.  Shuffles from consecutive rows interleave,
         * so the scheduler can overlap shuffle stalls with useful work. */
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            float abs_val = fabsf(vals[r]);
            float block_max = warpReduceMax(abs_val);
            float scale = fmaxf(block_max * SYM_CONS, 1e-5f);

            float dqval = fused_qd_sym(vals[r], scale);

            const int row = base_row + r;
            if (row < M && col < N) {
                out[row * N + col] = static_cast<scalar_t>(dqval);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * C++ dispatch
 * ---------------------------------------------------------------------------
 * Hardcoded ROWS_PER_THREAD=4 (best on B200).  Accepts any 2D contiguous tensor.
 * group_size is always 32 (warp width); symmetric quantization only. */

static int host_ceil_div(int a, int b) { return (a + b - 1) / b; }

torch::Tensor fused_fake_int4_quantize_dequantize_cuda(torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D, got ", x.dim(), "D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int ROWS_PER_THREAD = 4;
    dim3 grid(host_ceil_div(M, ROWS_PER_THREAD));
    dim3 block(1024);  /* 32 warps */

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16, x.scalar_type(),
        "fused_fake_int4_quantize_dequantize_cuda", [&] {
        fused_int4_qd_kernel<scalar_t, ROWS_PER_THREAD>
            <<<grid, block, 0, stream>>>(
                x.const_data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                M, N);
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_fake_int4_quantize_dequantize_cuda",
          &fused_fake_int4_quantize_dequantize_cuda,
          "Fused INT4 fake quantize-dequantize (symmetric, group_size=32)");
}
