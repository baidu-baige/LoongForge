#pragma once

#include "phase1.h"

#include <cstring>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

#include "config.h"

namespace sm100::bwd::head128 {

using namespace cute;

CUTE_DEVICE
float2 float2_sub(const float2& a, const float2& b) {
    return ku::float2_add(a, ku::float2_neg(b));
}

CUTE_DEVICE
void atomic_add_32floats_unrolled(float* dst, const float* src) {
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 0), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 4), "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 8), "f"(src[8]), "f"(src[9]), "f"(src[10]), "f"(src[11]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 12), "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 16), "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 20), "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 24), "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]) : "memory");
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 28), "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]) : "memory");
}

CUTE_DEVICE
int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return val;
}

template<typename TmaParamsType>
__global__ __launch_bounds__(NUM_THREADS, 1) void test_mla_bwd_kernel(
    __grid_constant__ const SparseAttnBwdParams params,
    __grid_constant__ const TmaParamsType tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    // Use cute namespace inside kernel to avoid conflicts with PyTorch's at::Layout
    using namespace cute;
    
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);
    
    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int s_q_idx = blockIdx.x / 2;
    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();  // Global warp index
    const int lane_idx = threadIdx.x % 32;
    if (s_q_idx >= params.s_q) {
        return;
    }
    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int32_t* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    const float* lse_s = params.lse + (int64_t)s_q_idx * params.h_q;
    const float* delta_s = params.delta + (int64_t)s_q_idx * params.stride_delta_s_q;
    
    // Determine warpgroup: 4 warpgroups, 128 threads each
    // WG0: threads 0-127 (warpgroup_idx = 0, warp_idx = 0-3)
    // WG1: threads 128-255 (warpgroup_idx = 1, warp_idx = 4-7)
    // WG2: threads 256-383 (warpgroup_idx = 2, warp_idx = 8-11)
    // WG3: threads 384-511 (warpgroup_idx = 3, warp_idx = 12-15)
    const int warpgroup_idx = __shfl_sync(0xffffffff, tid / 128, 0);
    const int idx_in_warpgroup = tid % 128;
    const bool is_wg0 = (warpgroup_idx == 0);
    const bool is_wg1 = (warpgroup_idx == 1);
    const bool is_wg2 = (warpgroup_idx == 2);
    const bool is_wg3 = (warpgroup_idx == 3);
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
    }

    // Initialize barriers (warp 0 in CTA0)
    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_prologue_q_nope.init(1);
        plan.bar_prologue_q_rope.init(1);
        plan.bar_prologue_kv.init(1);
        plan.bar_prologue_dO.init(1);
        plan.bar_p_ready.init(1);        // WG3 notifies WG0 that p is ready (2CTA sync)
        plan.bar_dp_ready.init(1);       // WG3 notifies WG0 that dp is ready (2CTA sync)
        plan.bar_s_ready.init(128);      // WG0 notifies WG3 that s is ready
        plan.bar_ds_ready.init(128);     // WG0 notifies WG3 that ds is ready
        // WG3-WG2 barriers for dKV computation
        plan.bar_dkv_part0_ready.init(1);
        plan.bar_dkv_part1_ready.init(1);
        plan.bar_dkv_part2_ready.init(1);
        plan.bar_dkv_part0_done.init(128);
        plan.bar_dkv_part1_done.init(128);
        plan.bar_dkv_part2_done.init(128);
        // WG1-WG3 barriers for kv_peer cp_async
        plan.bar_kv_peer_cp_async.init(1);      // cp_async transaction barrier
        plan.bar_kv_peer_ready.init(1);         // WG1 notifies WG3 kv_peer is ready
        // WG3-WG0 barrier for dQ computation
        plan.bar_k_valid_ready.init(8);
        plan.bar_k_valid_free.init(128);
        plan.bar_dq_ready.init(1);              // WG3 notifies WG0 dQ is ready
        fence_barrier_init();
    }
    
    // Cluster sync before accessing peer SMEM - all CTAs must participate
    __syncthreads();
    cluster_sync();
    
    // Construct SMEM Tensors
    // Q and K are split into NoPE and RoPE parts
    // Q NoPE: [B_H/2, D_V] = [64, 512], Q RoPE: [B_H/2, D_ROPE] = [64, 64]
    // K NoPE: [B_TOPK/2, D_V] = [32, 512], K RoPE: [B_TOPK/2, D_ROPE] = [32, 64]
    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sKNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKNoPE{});
    Tensor sKRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    // kv_peer: [B_TOPK/2, D_K] = [32, 576] for cross-CTA K loading
    Tensor sKV_peer = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKV{});

    // Launch prologue TMA loads and allocate TMEM (warp 0 in each CTA)
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Q_NoPE: [B_H, D_V] split by CTA on first dim
            Tensor gQNoPE = flat_divide(
                tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQNoPE, sQNoPE, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            // Q_RoPE: [B_H, D_ROPE] split by CTA on first dim
            Tensor gQRoPE = flat_divide(
                tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_rope, gQRoPE, sQRoPE, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);

            // dO: [B_H, D_V] split by CTA on first dim
            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);

            // arrive_and_expect_tx is issued in MMA warp before actual use
        }

        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();  // Wait for TMEM allocation

    // ========================================
    // Warpgroup 0: Softmax and dS Computation (WG0)
    // Responsibility: Compute softmax(P), load delta, compute ds
    // ========================================
    if (is_wg0) {
        cutlass::arch::warpgroup_reg_alloc<184>();
        const int global_row_idx = cta_idx * (B_H/2) + idx_in_warpgroup % (B_H/2);
        // Forward stores LSE in natural-log domain. Convert to log2 domain because
        // backward reconstructs softmax with exp2(P*scale_log2e - lse_log2).
        const float row_lse = __ldg(lse_s + global_row_idx) * 1.44269504f;
        const float delta_val = __ldg(delta_s + global_row_idx);

        const float sm_scale = params.sm_scale;
        const float scale = params.sm_scale_div_log2;

        const float2 neg_lse_f2 = make_float2(-row_lse, -row_lse);
        const float2 scale_f2 = make_float2(scale, scale);
        const float2 delta_f2 = make_float2(delta_val, delta_val);
        const float2 sm_scale_f2 = make_float2(sm_scale, sm_scale);

        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
        Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutS{});

        const uint32_t tmem_base = plan.tmem_start_addr.data()[0];
        const uint32_t tmem_lane = idx_in_warpgroup % (B_H/2);
        const uint32_t tmem_p_addr = tmem_base + (tmem_lane << 16) + tmem_cols::P;
        const uint32_t tmem_dp_addr = tmem_base + (tmem_lane << 16) + tmem_cols::dP;
        const int col_offset = (idx_in_warpgroup / 64) * 32;
        const int s_row = idx_in_warpgroup % 64;

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;

            // Step 1: Wait for WG3 to compute P for current block
            plan.bar_p_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            // Step 2: Load P from TMEM (current tile) and reuse p[] as softmax result.
            float2 p[(B_TOPK/2)/2];
            ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_p_addr, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            plan.bar_k_valid_ready.wait(phase);
            const uint32_t is_k_valid_lo =
                *(uint32_t*)(plan.is_k_valid + (idx_in_warpgroup >= 64 ? B_TOPK/8/2 : 0));
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK/2; ++i) {
                if (!(is_k_valid_lo >> i & 1))
                    p_float[i] = -CUDART_INF_F;
            }
            plan.bar_k_valid_free.arrive();

            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                float2 scaled_p = ku::float2_fma(p[i], scale_f2, neg_lse_f2);
                p[i].x = exp2f(scaled_p.x);
                p[i].y = exp2f(scaled_p.y);
            }

            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                sS(i * 2 + col_offset, s_row) = bf16(p[i].x);
                sS(i * 2 + 1 + col_offset, s_row) = bf16(p[i].y);
            }
            fence_view_async_shared();
            __threadfence_block();

            // Step 5: Notify WG3 that s is ready.
            if (cta_idx == 0) {
                plan.bar_s_ready.arrive(0u);
            } else {
                plan.bar_s_ready.arrive(1u);
            }

            // Step 7: Wait for WG3 to compute dP for current block
            plan.bar_dp_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            // Steps 8-10: Stream dP in small chunks and compute ds directly to SMEM.
            constexpr int DP_CHUNK_F2 = 4;
            constexpr int NUM_DP_CHUNKS = (B_TOPK/2)/2 / DP_CHUNK_F2;
            CUTE_UNROLL
            for (int ch = 0; ch < NUM_DP_CHUNKS; ++ch) {
                float2 dp[DP_CHUNK_F2];
                ku::tmem_ld_32dp32bNx<DP_CHUNK_F2 * 2>(tmem_dp_addr + ch * DP_CHUNK_F2 * 2, dp);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < DP_CHUNK_F2; ++i) {
                    const int idx = ch * DP_CHUNK_F2 + i;
                    float2 dp_minus_delta = float2_sub(dp[i], delta_f2);
                    float2 ds_val = ku::float2_mul(ku::float2_mul(p[idx], dp_minus_delta), sm_scale_f2);
                    sDS(idx * 2 + col_offset, s_row) = bf16(ds_val.x);
                    sDS(idx * 2 + 1 + col_offset, s_row) = bf16(ds_val.y);
                }
            }
            fence_view_async_shared();
            __threadfence_block();

            // Step 11: Notify WG3 that ds is ready.
            if (cta_idx == 0) {
                plan.bar_ds_ready.arrive(0u);
            } else {
                plan.bar_ds_ready.arrive(1u);
            }
        }

        // ========================================
        // WG0 Step 12-13: Wait for dQ and transfer to global memory
        // ========================================
        const int final_phase = (num_k_blocks - 1) & 1;
        plan.bar_dq_ready.wait(final_phase);
        ku::tcgen05_after_thread_sync();

        // Read dQ (fp32) from TMEM, convert to bf16 in SMEM, then TMA-store to global memory.
        // dQ shape: [B_H, D_Q] = [128, 576], each CTA handles [B_H/2, D_Q] = [64, 576].
        {
            constexpr int dQ_ROWS = B_H / 2;
            constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;
            constexpr int NOPE_CHUNKS = 8;
            constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;
            constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;
            constexpr int ROPE_FLOAT2_PER_ROW = D_ROPE / 2 / 2;

            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dq.data()), SmemLayoutQ{});

            const int row_in_cta = idx_in_warpgroup % dQ_ROWS;
            const int col_half = idx_in_warpgroup / dQ_ROWS;

            const uint32_t tmem_addr_dq0 = tmem_base + (row_in_cta << 16) + tmem_cols::dQ;
            const uint32_t tmem_addr_dq1 = tmem_base + (row_in_cta << 16) + (tmem_cols::dQ + 128);

            // dQ_NoPE part0: cols [0, 255]
            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq0 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            // dQ_NoPE part1: cols [256, 511]
            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq1 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = 256 + col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            // dQ_RoPE: cols [512, 575] loaded in chunks to reduce peak register use.
            constexpr int ROPE_F2_CHUNK = 8;
            constexpr int ROPE_NUM_CHUNKS = ROPE_FLOAT2_PER_ROW / ROPE_F2_CHUNK;
            const uint32_t tmem_addr_dq_rope = tmem_base + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
            CUTE_UNROLL
            for (int rch = 0; rch < ROPE_NUM_CHUNKS; ++rch) {
                float2 dq_rope_chunk[ROPE_F2_CHUNK];
                ku::tmem_ld_32dp32bNx<ROPE_F2_CHUNK * 2>(
                    tmem_addr_dq_rope + rch * ROPE_F2_CHUNK * 2, dq_rope_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < ROPE_F2_CHUNK; ++i) {
                    const int fi = rch * ROPE_F2_CHUNK + i;
                    int col = D_V + col_half * (D_ROPE / 2) + fi * 2;
                    sdQ(row_in_cta, col) = bf16(dq_rope_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_rope_chunk[i].y);
                }
            }

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, 0);

            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gdQ = flat_divide(
                    tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx),
                    Tile<Int<B_H/2>>{}
                )(_, cta_idx, _);
                auto thr_tma_dq = tma_params.tma_dQ.get_slice(_0{});
                cute::copy(
                    tma_params.tma_dQ,
                    thr_tma_dq.partition_S(sdQ),
                    thr_tma_dq.partition_D(gdQ)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
        }
    }
    
    // ========================================
    // Warpgroup 1: KV Loading (WG1)
    // Responsibility: Maintain per-block KV tile and kv_peer transfer
    // ========================================
    if (is_wg1) {
        cutlass::arch::warpgroup_reg_dealloc<64>();
        const int local_warp_idx = warp_idx - 4;  // WG1 has global warps [4, 7]
        constexpr int NUM_WARPS = 4;
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 2) / 4 / NUM_WARPS;
        constexpr int KV_PEER_ELEMENTS = (B_TOPK / 2) * D_K;  // 32 * 576 = 18432

        // Use lane0 of each warp (4 warps in WG1) to issue gather4 TMA loads.
        if (elect_one_sync()) {
            bf16* sKV_base = plan.u.q_kv.k_nope.data() + local_warp_idx * 4 * 64;

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;

                if (k_block > 0) {
                    // Keep producer/consumer order between iterations for all producer warps.
                    plan.bar_dq_ready.wait((k_block - 1) & 1);
                }

                int4 indices4[NUM_LOCAL_ROWS_PER_WARP];
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices4[local_row] = __ldg(
                        (const int4*)(gIndices_s + k_block * B_TOPK + cta_idx * (B_TOPK / 2)) +
                        local_row * NUM_WARPS + local_warp_idx
                    );
                }

                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    CUTE_UNROLL
                    for (int local_col = 0; local_col < D_K / 64; ++local_col) {
                        ku::tma_gather4_cta_group_2<true>(
                            &(tma_params.tensor_map_kv),
                            plan.bar_prologue_kv,
                            sKV_base + local_row * (4 * NUM_WARPS) * 64 + local_col * ((B_TOPK / 2) * 64),
                            local_col * 64,
                            indices4[local_row],
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                    }
                }
                // Load kv_peer from peer CTA using cp_async every iteration.
                if (local_warp_idx == 0) {
                    // Wait until WG3 has consumed this K block for P, then copy to kv_peer.
                    plan.bar_p_ready.wait(phase);

                    plan.bar_kv_peer_cp_async.arrive_and_expect_tx(sizeof(bf16) * KV_PEER_ELEMENTS);
                    bf16* peer_kv_peer_ptr = kerutils::get_peer_addr(plan.u.q_kv.kv_peer.data());
                    transac_bar_t* peer_bar_ptr = kerutils::get_peer_addr(&plan.bar_kv_peer_cp_async);
                    kerutils::cp_async_bulk_shared_cta_to_shared_cluster(
                        peer_kv_peer_ptr,
                        plan.u.q_kv.k_nope.data(),
                        sizeof(bf16) * KV_PEER_ELEMENTS,
                        *peer_bar_ptr
                    );
                    fence_view_async_shared();
                    plan.bar_kv_peer_cp_async.wait(phase);
                    if (cta_idx == 0) {
                        plan.bar_kv_peer_ready.arrive(0u);
                    } else {
                        plan.bar_kv_peer_ready.arrive(1u);
                    }
                }
            }
        }
    }
    
    // ========================================
    // Warpgroup 2: dKV Transfer (WG2)
    // Responsibility: Read dKV from TMEM and atomicAdd to global memory
    // ========================================
    if (is_wg2) {
        cutlass::arch::warpgroup_reg_dealloc<96>();
        const int row = idx_in_warpgroup % B_TOPK;   // 0-63: which KV row
        const int half = idx_in_warpgroup / B_TOPK;   // 0 or 1: which column half

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;
            const int row_base = k_block * B_TOPK;
            const int row_global = row_base + row;
            int kv_idx = -1;
            if (row_global < topk_length) {
                kv_idx = __ldg(gIndices_s + row_global);
            }
            const bool row_valid = kv_idx >= 0 && kv_idx <= max_kv_i;

            plan.bar_dkv_part0_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            constexpr int COLS_PER_HALF = 256 / 2;
            constexpr int CHUNK_SIZE = COLS_PER_HALF / 4;
            CUTE_UNROLL
            for (int chunk = 0; chunk < 4; ++chunk) {
                float2 dkv_data[CHUNK_SIZE / 2];
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                if (row_valid) {
                    float* dst = params.dKV + kv_idx * params.stride_dKV_s_kv + half * COLS_PER_HALF + chunk * CHUNK_SIZE;
                    float* src = (float*)dkv_data;
                    atomic_add_32floats_unrolled(dst, src);
                }
            }
            if (cta_idx == 0) {
                plan.bar_dkv_part0_done.arrive(0u);
            } else {
                plan.bar_dkv_part0_done.arrive(1u);
            }

            plan.bar_dkv_part1_ready.wait(phase);
            ku::tcgen05_after_thread_sync();
            CUTE_UNROLL
            for (int chunk = 0; chunk < 4; ++chunk) {
                float2 dkv_data[CHUNK_SIZE / 2];
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                if (row_valid) {
                    float* dst = params.dKV + kv_idx * params.stride_dKV_s_kv + 256 + half * COLS_PER_HALF + chunk * CHUNK_SIZE;
                    float* src = (float*)dkv_data;
                    atomic_add_32floats_unrolled(dst, src);
                }
            }
            if (cta_idx == 0) {
                plan.bar_dkv_part1_done.arrive(0u);
            } else {
                plan.bar_dkv_part1_done.arrive(1u);
            }

            plan.bar_dkv_part2_ready.wait(phase);
            ku::tcgen05_after_thread_sync();
            constexpr int ROPE_COLS_PER_HALF = D_ROPE / 2;
            float2 dkv_rope_data[ROPE_COLS_PER_HALF / 2];
            ku::tmem_ld_32dp32bNx<ROPE_COLS_PER_HALF>(tmem_cols::dKV_RoPE, dkv_rope_data);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            if (row_valid) {
                float* dst = params.dKV + kv_idx * params.stride_dKV_s_kv + D_V + half * ROPE_COLS_PER_HALF;
                float* src = (float*)dkv_rope_data;
                atomic_add_32floats_unrolled(dst, src);
            }
            if (cta_idx == 0) {
                plan.bar_dkv_part2_done.arrive(0u);
            } else {
                plan.bar_dkv_part2_done.arrive(1u);
            }
        }
    }

    // ========================================
    // Warpgroup 3: Matrix Multiplication (WG3)
    // Responsibility: Compute P, dP, and dKV
    // ========================================
    if (is_wg3) {
        cutlass::arch::warpgroup_reg_alloc<168>();
        // Allocate TMEM tensors for P and dP
        TiledMMA_P tiled_mma_P{};
        TiledMMA_dP tiled_mma_dP{};
        Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        tP.data().get() = tmem_cols::P;
        tdP.data().get() = tmem_cols::dP;

        // Allocate TMEM tensors for dKV
        TiledMMA_dKV tiled_mma_dKV{};
        TiledMMA_dKV_RoPE tiled_mma_dKV_RoPE{};
        Tensor tdKV = partition_fragment_C(tiled_mma_dKV, Shape<Int<B_TOPK>, Int<256>>{});
        tdKV.data().get() = tmem_cols::dKV;
        Tensor tdKV_RoPE = partition_fragment_C(tiled_mma_dKV_RoPE, Shape<Int<B_TOPK>, Int<D_ROPE>>{});
        tdKV_RoPE.data().get() = tmem_cols::dKV_RoPE;

        // Extract V from memory: V uses the same layout as K_NoPE
        Tensor sV = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutV{});

        // Loop body is executed by one elected warp in each CTA.
        if (warp_idx == 12 && elect_one_sync()) {
            if (cta_idx == 0) {
                // Q and dO prologue synchronization is consumed in WG3 before any MMA use.
                plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
                plan.bar_prologue_dO.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_nope.wait(0);
                plan.bar_prologue_q_rope.wait(0);
                plan.bar_prologue_dO.wait(0);
                ku::tcgen05_after_thread_sync();
            }

            // S and dS tensors for MMA A operand
            Tensor sS_mma = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
            Tensor sDS_mma = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});

            // dO transposed: full [D_V, B_H/2] = [512, 64], then flat_divide into [256, 64] tiles
            Tensor sdO_t_full = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutQTilesTransposed<D_V/64>{});
            auto sdO_t_div = flat_divide(sdO_t_full, Shape<Int<256>, Int<B_H/2>>{});

            // Q NoPE transposed: full [D_V, B_H/2] = [512, 64], then flat_divide into [256, 64] tiles
            Tensor sQ_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQTilesTransposed<D_V/64>{});
            auto sQ_t_div = flat_divide(sQ_t_full, Shape<Int<256>, Int<B_H/2>>{});

            // Q RoPE transposed: [D_ROPE, B_H/2] = [64, 64]
            Tensor sQ_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPETransposed{});

            // Allocate TMEM tensors for dQ
            TiledMMA_dQ tiled_mma_dQ{};
            TiledMMA_dQ_RoPE tiled_mma_dQ_RoPE{};
            Tensor tdQ_part0 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<256>>{});
            tdQ_part0.data().get() = tmem_cols::dQ;
            Tensor tdQ_part1 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<256>>{});
            tdQ_part1.data().get() = tmem_cols::dQ + 128;
            Tensor tdQ_RoPE = partition_fragment_C(tiled_mma_dQ_RoPE, Shape<Int<B_H/2>, Int<D_ROPE>>{});
            tdQ_RoPE.data().get() = tmem_cols::dQ_RoPE;

            // dS transposed tensor for MMA A operand: [B_H/2, B_TOPK] -> A matrix
            Tensor sDS_t = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});
            auto sDS_t_div = flat_divide(sDS_t, Shape<Int<B_H/2>, Int<B_TOPK/2>>{});

            // K NoPE/RoPE transposed
            Tensor sK_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKVTilesTransposed<D_V/64>{});
            auto sK_nope_t_div = flat_divide(sK_nope_t_full, Shape<Int<256>, Int<B_TOPK/2>>{});
            Tensor sK_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPETransposed{});

            // K_peer transposed
            Tensor sK_peer_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKVTilesTransposed<D_V/64>{});
            auto sK_peer_nope_t_div = flat_divide(sK_peer_nope_t_full, Shape<Int<256>, Int<B_TOPK/2>>{});
            Tensor sK_peer_rope_t = make_tensor(
                make_smem_ptr(plan.u.q_kv.kv_peer.data() + (B_TOPK/2) * D_V),
                SmemLayoutKRoPETransposed{}
            );

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;
                const bool dq_clear = (k_block == 0);

                // Pipeline dependency:
                // Reuse dKV(0:511) TMEM buffer for part0(k) only after part1(k-1) is fully drained by WG2.
                if (k_block > 0) {
                    const int prev_phase = (k_block - 1) & 1;
                    plan.bar_dkv_part1_done.wait(prev_phase);
                    ku::tcgen05_after_thread_sync();
                }

                // CTA0 computes P/dP and notifies WG0.
                if (cta_idx == 0) {
                    // Wait for WG1 KV TMA completion of current block.
                    plan.bar_prologue_kv.arrive_and_expect_tx(B_TOPK * D_K * sizeof(bf16));
                    plan.bar_prologue_kv.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_P, sQNoPE, sKNoPE, tP, true);
                    ku::utcmma_ss(tiled_mma_P, sQRoPE, sKRoPE, tP, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready, 1|2);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready, 1|2);
                    ku::tcgen05_after_thread_sync();
                }

                // dKV part0: dV[0:256] + dK[0:256]
                plan.bar_s_ready.wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sS_mma, sdO_t_div(_, _, _0{}, _0{}), tdKV, true);

                plan.bar_ds_ready.wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sDS_mma, sQ_t_div(_, _, _0{}, _0{}), tdKV, false);
                ku::umma_arrive_noelect(plan.bar_dkv_part0_ready);
                ku::tcgen05_after_thread_sync();

                // dQ: k==0 clear, k>0 accumulate
                if (cta_idx == 0) {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_peer_rope_t, tdQ_RoPE, false);
                } else {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_peer_rope_t, tdQ_RoPE, false);
                }
                ku::umma_arrive_noelect(plan.bar_dq_ready);
                ku::tcgen05_after_thread_sync();

                // dKV part1: dV[256:512] + dK[256:512]
                plan.bar_dkv_part0_done.wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sS_mma, sdO_t_div(_, _, _1{}, _0{}), tdKV, true);
                ku::utcmma_ss(tiled_mma_dKV, sDS_mma, sQ_t_div(_, _, _1{}, _0{}), tdKV, false);
                ku::umma_arrive_noelect(plan.bar_dkv_part1_ready);
                ku::tcgen05_after_thread_sync();

                // dKV part2: dK_rope
                // Reuse dKV_RoPE TMEM buffer for part2(k) only after part2(k-1) is drained by WG2.
                if (k_block > 0) {
                    const int prev_phase = (k_block - 1) & 1;
                    plan.bar_dkv_part2_done.wait(prev_phase);
                    ku::tcgen05_after_thread_sync();
                }
                ku::utcmma_ss(tiled_mma_dKV_RoPE, sDS_mma, sQ_rope_t, tdKV_RoPE, true);
                ku::umma_arrive_noelect(plan.bar_dkv_part2_ready);
                ku::tcgen05_after_thread_sync();
            }

            // Drain outstanding part1/part2 writes of the final k-block before TMEM free.
            if (num_k_blocks > 0) {
                const int final_phase = (num_k_blocks - 1) & 1;
                plan.bar_dkv_part1_done.wait(final_phase);
                ku::tcgen05_after_thread_sync();
                plan.bar_dkv_part2_done.wait(final_phase);
                ku::tcgen05_after_thread_sync();
            }

            (void)sdO_t_full;
        } else if (warp_idx == 13) {
            // KV valid loading warp
            static_assert(B_TOPK == 64);
            if (lane_idx < 8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int32x8_t indices = ldg_256_indices((void*)(gIndices_s + k * B_TOPK + lane_idx * 8));
                    auto is_valid = [&](int rel_idx, int index) -> char {
                        const int topk_idx = k * B_TOPK + lane_idx * 8 + rel_idx;
                        return index >= 0 && index < params.s_kv && index <= max_kv_i && topk_idx < topk_length;
                    };
                    char is_ks_valid_mask =
                        is_valid(7, indices.a7) << 7 |
                        is_valid(6, indices.a6) << 6 |
                        is_valid(5, indices.a5) << 5 |
                        is_valid(4, indices.a4) << 4 |
                        is_valid(3, indices.a3) << 3 |
                        is_valid(2, indices.a2) << 2 |
                        is_valid(1, indices.a1) << 1 |
                        is_valid(0, indices.a0) << 0;

                    plan.bar_k_valid_free.wait(k & 1 ^ 1);
                    plan.is_k_valid[lane_idx] = is_ks_valid_mask;
                    plan.bar_k_valid_ready.arrive();
                }
            }
        }
    }

    // All threads must sync before proceeding
    __syncthreads();
    cluster_sync();
    
    // Free TMEM
    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(plan.tmem_start_addr.data()[0], 512);
    }
#endif
}

static void launch_test_mla_bwd(const SparseAttnBwdParams& params) {
    auto shape_Q_nope = cute::make_shape(B_H, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q),
            cute::make_layout(
                shape_Q_nope,
                cute::make_stride(params.stride_q_h_q, cute::_1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = cute::make_shape(B_H, D_ROPE, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q + D_V),
            cute::make_layout(
                shape_Q_rope,
                cute::make_stride(params.stride_q_h_q, cute::_1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_KV = cute::make_shape(params.s_kv, D_K);
    auto tma_KV = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.kv),
            cute::make_layout(shape_KV, cute::make_stride(params.stride_kv_s_kv, cute::_1{}))
        ),
        SmemLayoutKV{}
    );

    auto shape_dO = cute::make_shape(B_H, D_V, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dO),
            cute::make_layout(
                shape_dO,
                cute::make_stride(params.stride_dO_h_q, cute::_1{}, params.stride_dO_s_q)
            )
        ),
        SmemLayoutdO{}
    );

    auto shape_dQ = cute::make_shape(B_H, D_Q, params.s_q);
    auto tma_dQ = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dQ),
            cute::make_layout(
                shape_dQ,
                cute::make_stride(params.stride_dQ_h_q, cute::_1{}, params.stride_dQ_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    CUtensorMap tensor_map_kv;
    {
        uint64_t size[2] = {(uint64_t)D_K, (unsigned long)params.s_kv};
        uint64_t stride[1] = {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<bf16*>(params.kv),
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    using KernelTmaParams = TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_KV), decltype(tma_KV),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ)
    >;

    KernelTmaParams tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_KV, tma_KV,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        tensor_map_kv
    };

    auto kernel = &test_mla_bwd_kernel<KernelTmaParams>;
    dim3 grid(2 * params.s_q, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = params.stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    KU_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params, tma_params));
}

template<int DQK>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    static_assert(DQK == D_QK);

    KU_ASSERT(params.d_qk == DQK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk > 0 && params.topk % B_TOPK == 0);

    run_bwd_preprocess_delta_kernel<DQK>(params);
    launch_test_mla_bwd(params);
}

}  // namespace sm100::bwd::head128