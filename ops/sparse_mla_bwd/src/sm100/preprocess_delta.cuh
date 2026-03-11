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
 * @brief Preprocess kernel to compute delta = sum(O * dO, dim=-1)
 * 
 * This kernel computes delta independently before the main backward kernel.
 * Each block processes B_H=128 rows (all heads) for one query token.
 * 
 * Grid: [s_q, 1, 1] - one block per query token
 * Block: [128, 1, 1] - 128 threads, each thread processes one row (one head)
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
    
    const int s_q_idx = blockIdx.x;
    const int head_idx = threadIdx.x;  // Each thread processes one head
    
    if (head_idx >= B_H) return;
    
    // Compute delta: delta[i] = sum_j(O[i,j] * dO[i,j])
    // Each thread processes one row (one head)
    const bf16* gO = params.o + s_q_idx * params.stride_o_s_q + head_idx * params.stride_o_h_q;
    const bf16* gdO = params.dO + s_q_idx * params.stride_dO_s_q + head_idx * params.stride_dO_h_q;
    
    float delta = 0.0f;
    
    // Accumulate O * dO using vectorized loads
    CUTE_UNROLL
    for (int col = 0; col < D_V; col += 8) {
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
    
    // Write delta to global memory
    float* gDelta = params.delta + s_q_idx * params.stride_delta_s_q + head_idx * params.stride_delta_h_q;
    *gDelta = delta;
    
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
 * 2. Launch preprocessing kernel to compute delta
 * 
 * Grid: [s_q, 1, 1] - one block per query token
 * Block: [128, 1, 1] - 128 threads (one per head)
 */
template<int D_QK>
void run_bwd_preprocess_delta_kernel(const SparseAttnBwdParams& params) {
    static_assert(D_QK == 576);  // Only support D_QK == 576 for backward kernel
    
    constexpr int B_H = 128;
    
    // === Parameter validation ===
    KU_ASSERT(params.h_q == B_H);  // Query head count must equal B_H
    
    // === Launch preprocessing kernel ===
    dim3 grid(params.s_q, 1, 1);      // One block per query token
    dim3 block(B_H, 1, 1);           // 128 threads (one per head)
    
    preprocess_delta_kernel<D_QK><<<grid, block, 0, params.stream>>>(params);
    KU_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sm100::bwd::head128