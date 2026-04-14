#pragma once

#include "phase1.h"

#include <cute/tensor.hpp>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"
#include "config.h"

namespace sm100::bwd::head128 {

using namespace cute;

/**
 * @brief Preprocess kernel to compute and store negated delta.
 * 
 * This kernel computes delta = sum(O * dO, dim=-1) and stores -delta
 * independently before the main backward kernel.
 * Each row uses THREADS_PER_ROW cooperative threads for reduction.
 * 
 * Grid: [s_q, head_tiles, 1]
 * Block: [128, 1, 1] - 128 threads, each THREADS_PER_ROW threads process one head row
 * 
 * @tparam D_QK Query/Key dimension (576)
 */
template<int D_QK>
__global__ void
preprocess_delta_kernel(const SparseAttnBwdParams params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    static_assert(D_QK == 576);  // Only support D_QK == 576
    
    constexpr int D_V = 512;  // Value dimension
    constexpr int B_H = 128;  // Number of heads
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ELEMENTS_PER_VEC = 8;  // uint4 = 8 bf16
    constexpr int ROWS_PER_BLOCK = B_H / THREADS_PER_ROW;
    static_assert(B_H % THREADS_PER_ROW == 0);
    
    const int s_q_idx = blockIdx.x;
    const int lane_in_row = threadIdx.x % THREADS_PER_ROW;
    const int row_in_block = threadIdx.x / THREADS_PER_ROW;
    const int head_idx = blockIdx.y * ROWS_PER_BLOCK + row_in_block;
    
    if (head_idx >= B_H) return;
    
    // Compute delta: delta[i] = sum_j(O[i,j] * dO[i,j]).
    // THREADS_PER_ROW threads cooperate on one row and reduce within the warp subgroup.
    const bf16* gO = params.o + s_q_idx * params.stride_o_s_q + head_idx * params.stride_o_h_q;
    const bf16* gdO = params.dO + s_q_idx * params.stride_dO_s_q + head_idx * params.stride_dO_h_q;
    
    float delta = 0.0f;
    
    // Accumulate O * dO using vectorized loads
    constexpr int COL_STEP = THREADS_PER_ROW * ELEMENTS_PER_VEC;
    CUTE_UNROLL
    for (int col = lane_in_row * ELEMENTS_PER_VEC; col < D_V; col += COL_STEP) {
        // Vectorized load of O (8 bf16 values = 128 bits)
        uint4 o_raw = __ldg((const uint4*)(gO + col));
        bf16x8 o_vec;
        *(uint4*)&o_vec = o_raw;
        
        // Vectorized load of dO (8 bf16 values = 128 bits)
        uint4 do_raw = __ldg((const uint4*)(gdO + col));
        bf16x8 do_vec;
        *(uint4*)&do_vec = do_raw;
        
        // Accumulate dot product
        delta += __bfloat162float(o_vec.a01.x) * __bfloat162float(do_vec.a01.x);
        delta += __bfloat162float(o_vec.a01.y) * __bfloat162float(do_vec.a01.y);
        delta += __bfloat162float(o_vec.a23.x) * __bfloat162float(do_vec.a23.x);
        delta += __bfloat162float(o_vec.a23.y) * __bfloat162float(do_vec.a23.y);
        delta += __bfloat162float(o_vec.a45.x) * __bfloat162float(do_vec.a45.x);
        delta += __bfloat162float(o_vec.a45.y) * __bfloat162float(do_vec.a45.y);
        delta += __bfloat162float(o_vec.a67.x) * __bfloat162float(do_vec.a67.x);
        delta += __bfloat162float(o_vec.a67.y) * __bfloat162float(do_vec.a67.y);
    }

    const uint32_t active_mask = __activemask();
    CUTE_UNROLL
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        delta += __shfl_xor_sync(active_mask, delta, offset);
    }
    
    // Write -delta to global memory so downstream can use add instead of subtract.
    if (lane_in_row == 0) {
        float* gDelta = params.delta + s_q_idx * params.stride_delta_s_q + head_idx * params.stride_delta_h_q;
        *gDelta = -delta;
    }
    
#else
    // Error handling for non-SM100 architectures
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

/**
 * @brief Host wrapper function to launch delta preprocessing kernel
 * @tparam D_QK Query/Key dimension (576)
 * @param params Attention computation parameter struct
 * 
 * Functionality:
 * 1. Parameter validation
 * 2. Launch preprocessing kernel to compute and store -delta
 * 
 * Grid: [s_q, head_tiles, 1]
 * Block: [128, 1, 1] - 128 threads, each THREADS_PER_ROW threads process one head row
 */
template<int D_QK>
void run_bwd_preprocess_delta_kernel(const SparseAttnBwdParams& params) {
    static_assert(D_QK == 576);  // Only support D_QK == 576 for backward kernel
    
    constexpr int B_H = 128;
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ROWS_PER_BLOCK = B_H / THREADS_PER_ROW;
    
    // === Parameter validation ===
    KU_ASSERT(params.h_q == B_H);  // Query head count must equal B_H
    
    // === Launch preprocessing kernel ===
    dim3 grid(params.s_q, (B_H + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    dim3 block(B_H, 1, 1);  // 128 threads: 16 rows/block, 8 threads/row
    
    preprocess_delta_kernel<D_QK><<<grid, block, 0, params.stream>>>(params);
    KU_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sm100::bwd::head128

namespace sm100::bwd::head64 {

using namespace cute;

/**
 * @brief Preprocess kernel to compute and store negated delta.
 *
 * This kernel computes delta = sum(O * dO, dim=-1) and stores -delta
 * independently before the main backward kernel.
 * Each row uses THREADS_PER_ROW cooperative threads for reduction.
 *
 * Grid: [s_q, head_tiles, 1]
 * Block: [64, 1, 1] - 64 threads, each THREADS_PER_ROW threads process one head row
 *
 * @tparam D_QK Query/Key dimension (576)
 */
template<int D_QK>
__global__ void
preprocess_delta_kernel(const SparseAttnBwdParams params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    static_assert(D_QK == 576);  // Only support D_QK == 576

    constexpr int D_V = 512;  // Value dimension
    constexpr int B_H = 64;   // Number of heads (single CTA architecture)
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ELEMENTS_PER_VEC = 8;  // uint4 = 8 bf16
    constexpr int ROWS_PER_BLOCK = B_H / THREADS_PER_ROW;
    static_assert(B_H % THREADS_PER_ROW == 0);

    const int s_q_idx = blockIdx.x;
    const int lane_in_row = threadIdx.x % THREADS_PER_ROW;
    const int row_in_block = threadIdx.x / THREADS_PER_ROW;
    const int head_idx = blockIdx.y * ROWS_PER_BLOCK + row_in_block;

    if (head_idx >= B_H) return;

    // Compute delta: delta[i] = sum_j(O[i,j] * dO[i,j]).
    // THREADS_PER_ROW threads cooperate on one row and reduce within the warp subgroup.
    const bf16* gO = params.o + s_q_idx * params.stride_o_s_q + head_idx * params.stride_o_h_q;
    const bf16* gdO = params.dO + s_q_idx * params.stride_dO_s_q + head_idx * params.stride_dO_h_q;

    float delta = 0.0f;

    // Accumulate O * dO using vectorized loads
    constexpr int COL_STEP = THREADS_PER_ROW * ELEMENTS_PER_VEC;
    CUTE_UNROLL
    for (int col = lane_in_row * ELEMENTS_PER_VEC; col < D_V; col += COL_STEP) {
        // Vectorized load of O (8 bf16 values = 128 bits)
        uint4 o_raw = __ldg((const uint4*)(gO + col));
        bf16x8 o_vec;
        *(uint4*)&o_vec = o_raw;

        // Vectorized load of dO (8 bf16 values = 128 bits)
        uint4 do_raw = __ldg((const uint4*)(gdO + col));
        bf16x8 do_vec;
        *(uint4*)&do_vec = do_raw;

        // Accumulate dot product
        delta += __bfloat162float(o_vec.a01.x) * __bfloat162float(do_vec.a01.x);
        delta += __bfloat162float(o_vec.a01.y) * __bfloat162float(do_vec.a01.y);
        delta += __bfloat162float(o_vec.a23.x) * __bfloat162float(do_vec.a23.x);
        delta += __bfloat162float(o_vec.a23.y) * __bfloat162float(do_vec.a23.y);
        delta += __bfloat162float(o_vec.a45.x) * __bfloat162float(do_vec.a45.x);
        delta += __bfloat162float(o_vec.a45.y) * __bfloat162float(do_vec.a45.y);
        delta += __bfloat162float(o_vec.a67.x) * __bfloat162float(do_vec.a67.x);
        delta += __bfloat162float(o_vec.a67.y) * __bfloat162float(do_vec.a67.y);
    }

    const uint32_t active_mask = __activemask();
    CUTE_UNROLL
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
        delta += __shfl_xor_sync(active_mask, delta, offset);
    }

    // Write -delta to global memory so downstream can use add instead of subtract.
    if (lane_in_row == 0) {
        float* gDelta = params.delta + s_q_idx * params.stride_delta_s_q + head_idx * params.stride_delta_h_q;
        *gDelta = -delta;
    }

#else
    // Error handling for non-SM100 architectures
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

/**
 * @brief Host wrapper function to launch delta preprocessing kernel
 * @tparam D_QK Query/Key dimension (576)
 * @param params Attention computation parameter struct
 *
 * Functionality:
 * 1. Parameter validation
 * 2. Launch preprocessing kernel to compute and store -delta
 *
 * Grid: [s_q, head_tiles, 1]
 * Block: [64, 1, 1] - 64 threads, each THREADS_PER_ROW threads process one head row
 */
template<int D_QK>
void run_bwd_preprocess_delta_kernel(const SparseAttnBwdParams& params) {
    static_assert(D_QK == 576);  // Only support D_QK == 576 for backward kernel

    constexpr int B_H = 64;   // Number of heads (single CTA architecture)
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ROWS_PER_BLOCK = B_H / THREADS_PER_ROW;

    // === Parameter validation ===
    KU_ASSERT(params.h_q == B_H);  // Query head count must equal B_H

    // === Launch preprocessing kernel ===
    dim3 grid(params.s_q, (B_H + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    dim3 block(B_H, 1, 1);  // 64 threads: 8 rows/block, 8 threads/row

    preprocess_delta_kernel<D_QK><<<grid, block, 0, params.stream>>>(params);
    KU_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sm100::bwd::head64
