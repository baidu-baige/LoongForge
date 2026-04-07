#pragma once

#include "../preprocess_delta.cuh"

#include "dq_config.h"

#include <cstring>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

namespace sm100::bwd::head128_2kernels::dq {

using namespace cute;

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

static constexpr int kThreadsPerWarp = 32;
static constexpr int kWarpsPerWarpgroup = 4;
static constexpr int kThreadsPerWarpgroup = kWarpsPerWarpgroup * kThreadsPerWarp;
static constexpr int kNumWarpgroups = 4;
static constexpr int kWg3MmaWarp = 0;
static constexpr int kWg3KvValidWarp = 1;
static constexpr int kWg3StoreSWarp = 2;
static constexpr int kWg3StoreDsWarp = 3;
static constexpr uint16_t kClusterMask2Cta = 0x3;

static_assert(NUM_THREADS == kNumWarpgroups * kThreadsPerWarpgroup, "NUM_THREADS must match the dq warpgroup layout.");

template<typename TmaParamsType>
__global__ __launch_bounds__(NUM_THREADS, 1) void dq_phase_kernel(
    __grid_constant__ const SparseAttnBwdParams params,
    __grid_constant__ const TmaParamsType tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;
    const int s_q_idx = blockIdx.x / 2;
    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = tid % kThreadsPerWarp;
    const int warpgroup_idx = __shfl_sync(0xffffffff, tid / kThreadsPerWarpgroup, 0);
    const int idx_in_warpgroup = tid % kThreadsPerWarpgroup;
    const int local_warp_idx = warp_idx - warpgroup_idx * kWarpsPerWarpgroup;
    if (s_q_idx >= params.s_q) {
        return;
    }
    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int32_t* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;
    const float* lse_s = params.lse + (int64_t)s_q_idx * params.h_q;
    const float* delta_s = params.delta + (int64_t)s_q_idx * params.stride_delta_s_q;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_S.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dS.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_rope));
    }

    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_prologue_q_nope.init(1);
        plan.bar_prologue_q_rope.init(1);
        plan.bar_prologue_utccp.init(1);
        plan.bar_prologue_dO.init(1);
        plan.bar_s_ready.init(kThreadsPerWarpgroup);
        plan.bar_s_store_done.init(1);
        plan.bar_ds_ready.init(kThreadsPerWarpgroup * 2);
        plan.bar_ds_store_ready.init(kThreadsPerWarpgroup);
        plan.bar_ds_store_done.init(1);
        plan.bar_k_valid_ready.init(B_TOPK / 8);
        plan.bar_k_valid_free.init(kThreadsPerWarpgroup);
        plan.bar_k_dq_nope_ready.init(1);
        plan.bar_k_dq_rope_ready.init(1);
        plan.bar_dq_ready.init(1);
        for (int buf = 0; buf < NUM_KV_BUFS; ++buf) {
            plan.bar_prologue_kv[buf].init(1);
            plan.bar_p_ready[buf].init(1);
            plan.bar_dp_ready[buf].init(1);
        }
        fence_barrier_init();
    }

    cluster_sync();

    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.u.q_full.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.u.q_full.data() + (B_H / 2) * D_V), SmemLayoutQRoPE{});
    Tensor sQ = make_tensor(make_smem_ptr(plan.u.q_kv.sq.data()), SmemLayoutQTiles<NUM_sQ_TILES>{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            Tensor gQNoPE = flat_divide(
                tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQNoPE, sQNoPE, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gQRoPE = flat_divide(
                tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_rope, gQRoPE, sQRoPE, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);

            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H / 2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);
        }

        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();
    const uint32_t tmem_base = plan.tmem_start_addr.data()[0];

    if (warpgroup_idx == 0) {
        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
        Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});
        const int idx_in_softmax = idx_in_warpgroup;
        const int global_row_idx = cta_idx * (B_H / 2) + idx_in_softmax % (B_H / 2);
        const float row_lse = __ldg(lse_s + global_row_idx) * 1.44269504f;
        const float neg_delta_val = __ldg(delta_s + global_row_idx);
        const float sm_scale = params.sm_scale;
        const float scale = params.sm_scale_div_log2;

        const float2 neg_lse_f2 = make_float2(-row_lse, -row_lse);
        const float2 scale_f2 = make_float2(scale, scale);
        const float2 neg_delta_f2 = make_float2(neg_delta_val, neg_delta_val);
        const float2 sm_scale_f2 = make_float2(sm_scale, sm_scale);

        const uint32_t tmem_lane = idx_in_softmax % S_DS_ROWS_PER_CTA;
        const uint32_t tmem_p_addr = tmem_base + (tmem_lane << 16) + tmem_cols::P;
        const uint32_t tmem_dp_addr = tmem_base + (tmem_lane << 16) + tmem_cols::dP;
        const int row_in_tile = idx_in_softmax % S_DS_ROWS_PER_CTA;
        const int col_half = idx_in_softmax / S_DS_ROWS_PER_CTA;

        constexpr int SMEM_VEC_F2 = S_DS_VEC_ELEMS / 2;
        constexpr int NUM_SMEM_VEC_STORES = S_DS_COLS_PER_THREAD / S_DS_VEC_ELEMS;
        static_assert(SMEM_VEC_F2 == 4, "Softmax vectorized write path expects 4 float2 per 128-bit store.");
        bf16x8* sS_base = reinterpret_cast<bf16x8*>(plan.s_ds.s.data()) +
            row_in_tile + S_DS_ROWS_PER_CTA * (col_half * NUM_SMEM_VEC_STORES);
        bf16x8* sDS_base = reinterpret_cast<bf16x8*>(plan.s_ds.ds.data()) +
            row_in_tile + S_DS_ROWS_PER_CTA * (col_half * NUM_SMEM_VEC_STORES);

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int kv_buf = k_block % NUM_KV_BUFS;
            const int kv_phase = (k_block / NUM_KV_BUFS) & 1;
            const int round_phase = k_block & 1;

            plan.bar_p_ready[kv_buf].wait(kv_phase);
            ku::tcgen05_after_thread_sync();

            float2 p[(B_TOPK / 2) / 2];
            ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_p_addr, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            plan.bar_k_valid_ready.wait(round_phase);
            const uint32_t is_k_valid_lo =
                *(uint32_t*)(plan.is_k_valid + (idx_in_softmax >= S_DS_ROWS_PER_CTA ? B_TOPK / 8 / 2 : 0));
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK / 2; ++i) {
                if (!(is_k_valid_lo >> i & 1)) {
                    p_float[i] = -CUDART_INF_F;
                }
            }
            plan.bar_k_valid_free.arrive();

            if (k_block > 0) {
                plan.bar_s_store_done.wait((k_block - 1) & 1);
            }

            CUTE_UNROLL
            for (int vec = 0; vec < NUM_SMEM_VEC_STORES; ++vec) {
                const int base_idx = vec * SMEM_VEC_F2;
                bf16x8 s_pack;
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 0], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 0] = p_vec;
                    s_pack.a01 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 1], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 1] = p_vec;
                    s_pack.a23 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 2], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 2] = p_vec;
                    s_pack.a45 = __float22bfloat162_rn(p_vec);
                }
                {
                    float2 p_vec = ku::float2_fma(p[base_idx + 3], scale_f2, neg_lse_f2);
                    p_vec = make_float2(exp2f(p_vec.x), exp2f(p_vec.y));
                    p[base_idx + 3] = p_vec;
                    s_pack.a67 = __float22bfloat162_rn(p_vec);
                }
                sS_base[S_DS_ROWS_PER_CTA * vec] = s_pack;
            }
            fence_view_async_shared();
            __threadfence_block();

            plan.bar_s_ready.arrive(static_cast<uint32_t>(cta_idx));

            plan.bar_dp_ready[kv_buf].wait(kv_phase);
            ku::tcgen05_after_thread_sync();

            constexpr int DP_CHUNK_F2 = SMEM_VEC_F2;
            constexpr int NUM_DP_CHUNKS = (B_TOPK / 2) / 2 / DP_CHUNK_F2;
            if (k_block > 0) {
                plan.bar_ds_store_done.wait((k_block - 1) & 1);
            }
            CUTE_UNROLL
            for (int ch = 0; ch < NUM_DP_CHUNKS; ++ch) {
                float2 dp[DP_CHUNK_F2];
                ku::tmem_ld_32dp32bNx<DP_CHUNK_F2 * 2>(tmem_dp_addr + ch * DP_CHUNK_F2 * 2, dp);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                const int base_idx = ch * DP_CHUNK_F2;
                bf16x8 ds_pack;
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 0], ku::float2_add(dp[0], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a01 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 1], ku::float2_add(dp[1], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a23 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 2], ku::float2_add(dp[2], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a45 = __float22bfloat162_rn(ds_vec);
                }
                {
                    float2 ds_vec = ku::float2_mul(
                        ku::float2_mul(p[base_idx + 3], ku::float2_add(dp[3], neg_delta_f2)),
                        sm_scale_f2
                    );
                    ds_pack.a67 = __float22bfloat162_rn(ds_vec);
                }
                sDS_base[S_DS_ROWS_PER_CTA * ch] = ds_pack;
            }
            fence_view_async_shared();
            __threadfence_block();

            plan.bar_ds_ready.arrive(0u);
            plan.bar_ds_store_ready.arrive(static_cast<uint32_t>(cta_idx));
        }

        const int final_phase = (num_k_blocks - 1) & 1;
        plan.bar_dq_ready.wait(final_phase);
        ku::tcgen05_after_thread_sync();

        {
            constexpr int dQ_ROWS = B_H / 2;
            constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;
            constexpr int NOPE_CHUNKS = 8;
            constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;
            constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;
            constexpr int ROPE_FLOAT2_PER_ROW = D_ROPE / 2 / 2;

            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dq.data()), SmemLayoutQ{});

            const int row_in_cta = idx_in_softmax % dQ_ROWS;
            const int col_half = idx_in_softmax / dQ_ROWS;

            const uint32_t tmem_addr_dq0 = tmem_base + (row_in_cta << 16) + tmem_cols::dQ;
            const uint32_t tmem_addr_dq1 = tmem_base + (row_in_cta << 16) + (tmem_cols::dQ + 128);

            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq0 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = col_half * 256 + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq1 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = 128 + col_half * 256 + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

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
            NamedBarrier::arrive_and_wait(kThreadsPerWarpgroup, 0);

            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gdQ = flat_divide(
                    tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx),
                    Tile<Int<B_H / 2>>{}
                )(_, cta_idx, _);
                auto thr_tma_dq = tma_params.tma_dQ.get_slice(_0{});
                cute::copy(
                    tma_params.tma_dQ,
                    thr_tma_dq.partition_S(sdQ),
                    thr_tma_dq.partition_D(gdQ)
                );
            }
        }

        if (warp_idx == 0) {
            TMEM::Allocator2Sm().free(tmem_base, 512);
        }
    }

    if (warpgroup_idx == 1) {
        constexpr int NUM_WARPS = kWarpsPerWarpgroup;
        static_assert((B_TOPK / 2) % (4 * NUM_WARPS) == 0);
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 2) / 4 / NUM_WARPS;

        if (elect_one_sync()) {
            plan.bar_prologue_utccp.wait(0);

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int buf = k_block % NUM_KV_BUFS;
                const int phase = (k_block / NUM_KV_BUFS) & 1;

                if (k_block >= NUM_KV_BUFS) {
                    // Reuse the ping-pong KV buffer only after the same-phase dQ consumers
                    // have finished the previous round that touched it.
                    plan.bar_dp_ready[buf].wait(phase ^ 1);
                }

                bf16* sKV_base = plan.u.q_kv.kv[buf].data() + local_warp_idx * 4 * 64;
                int4 local_indices4[NUM_LOCAL_ROWS_PER_WARP];
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    local_indices4[local_row] = __ldg(
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
                            plan.bar_prologue_kv[buf],
                            sKV_base + local_row * (4 * NUM_WARPS) * 64 + local_col * ((B_TOPK / 2) * 64),
                            local_col * 64,
                            local_indices4[local_row],
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                    }
                }
            }
        }
    }

    if (warpgroup_idx == 2) {
        constexpr int NUM_WARPS = kWarpsPerWarpgroup;
        static_assert(B_TOPK % (4 * NUM_WARPS) == 0);
        constexpr int NUM_DQ_ROWS_PER_WARP = B_TOPK / 4 / NUM_WARPS;
        bf16* sKDQNoPE_base = plan.u.q_kv.k_dq.data();
        bf16* sKDQRoPE_base = plan.u.q_kv.k_dq.data() + cosize_v<SmemLayoutKDQNoPE>;

        if (elect_one_sync()) {
            plan.bar_prologue_utccp.wait(0);

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;

                if (k_block > 0) {
                    plan.bar_dq_ready.wait((k_block - 1) & 1);
                }

                int4 dq_indices4[NUM_DQ_ROWS_PER_WARP];
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_DQ_ROWS_PER_WARP; ++local_row) {
                    dq_indices4[local_row] = __ldg(
                        (const int4*)(gIndices_s + k_block * B_TOPK) +
                        local_row * NUM_WARPS + local_warp_idx
                    );
                }

                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_DQ_ROWS_PER_WARP; ++local_row) {
                    const int dq_row = local_row * (4 * NUM_WARPS) + local_warp_idx * 4;

                    CUTE_UNROLL
                    for (int local_col = 0; local_col < D_V / 64 / 2; ++local_col) {
                        bf16* k_dq_nope_dst = sKDQNoPE_base + dq_row * 64 + local_col * (B_TOPK * 64);
                        ku::tma_gather4_cta_group_2<true>(
                            &(tma_params.tensor_map_kv_nope),
                            plan.bar_k_dq_nope_ready,
                            k_dq_nope_dst,
                            cta_idx * (D_V / 2) + local_col * 64,
                            dq_indices4[local_row],
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                    }

                    bf16* k_dq_rope_dst = sKDQRoPE_base + dq_row * (D_ROPE / 2);
                    ku::tma_gather4_cta_group_2<true>(
                        &(tma_params.tensor_map_kv_rope),
                        plan.bar_k_dq_rope_ready,
                        k_dq_rope_dst,
                        cta_idx * (D_ROPE / 2),
                        dq_indices4[local_row],
                        (int64_t)TMA::CacheHintSm90::EVICT_LAST
                    );
                }
            }
        }
    }

    if (warpgroup_idx == 3) {
        if (local_warp_idx == kWg3MmaWarp) {
            TiledMMA_P_tQ tiled_mma_P_tQ{};
            TiledMMA_P_sQ tiled_mma_P_sQ{};
            TiledMMA_dP tiled_mma_dP{};
            Tensor tP = partition_fragment_C(tiled_mma_P_tQ, Shape<Int<B_H / 2>, Int<B_TOPK>>{});
            Tensor tQ = tiled_mma_P_tQ.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P_tQ, Shape<Int<B_H / 2>, Int<D_tQ>>{})
            );
            Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H / 2>, Int<B_TOPK>>{});
            tP.data().get() = tmem_cols::P;
            tQ.data().get() = tmem_cols::q;
            tdP.data().get() = tmem_cols::dP;

            if (elect_one_sync()) {
                if (cta_idx == 0) {
                    UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                        make_tensor(
                            make_smem_ptr(plan.u.q_full.data() + (B_H / 2) * D_sQ),
                            tile_to_shape(
                                UMMA::Layout_K_SW128_Atom<bf16>{},
                                Shape<Int<B_H / 2>, Int<64>>{}
                            )
                        )
                    );

                    plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                    plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
                    plan.bar_prologue_dO.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                    plan.bar_prologue_q_nope.wait(0);
                    plan.bar_prologue_q_rope.wait(0);
                    plan.bar_prologue_dO.wait(0);
                    ku::tcgen05_after_thread_sync();

                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < NUM_tQ_TILES; ++tile_idx) {
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
                            SM100_UTCCP_2x64dp128bitlw0213_2cta::copy(
                                sQ_desc + tile_idx * ((B_H / 2) * 128 / 16) + subtile_idx * (16 / 16),
                                tmem_cols::q + tile_idx * 32 + subtile_idx * 4
                            );
                        }
                    }
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_prologue_utccp, kClusterMask2Cta);
                }

                TiledMMA_dQ tiled_mma_dQ{};
                TiledMMA_dQ_RoPE tiled_mma_dQ_RoPE{};
                Tensor tdQ = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H / 2>, Int<D_V>>{});
                tdQ.data().get() = tmem_cols::dQ;
                Tensor tdQ_RoPE = partition_fragment_C(tiled_mma_dQ_RoPE, Shape<Int<B_H / 2>, Int<D_ROPE>>{});
                tdQ_RoPE.data().get() = tmem_cols::dQ_RoPE;

                Tensor sDS_t = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});
                Tensor sK_dq_nope_t = make_tensor(make_smem_ptr(plan.u.q_kv.k_dq.data()), SmemLayoutKDQNoPE_MMA{});
                Tensor sK_dq_rope_t = make_tensor(
                    make_smem_ptr(plan.u.q_kv.k_dq.data() + cosize_v<SmemLayoutKDQNoPE>),
                    SmemLayoutKDQRoPE_MMA{}
                );

                CUTE_NO_UNROLL
                for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                    const int kv_buf = k_block % NUM_KV_BUFS;
                    const int kv_phase = (k_block / NUM_KV_BUFS) & 1;
                    const int round_phase = k_block & 1;
                    const bool dq_clear = (k_block == 0);

                    if (cta_idx == 0) {
                        Tensor sK_sQ = make_tensor(
                            make_smem_ptr(plan.u.q_kv.kv[kv_buf].data()),
                            SmemLayoutKVTiles<NUM_sQ_TILES>{}
                        );
                        Tensor sK_tQ = make_tensor(
                            make_smem_ptr(plan.u.q_kv.kv[kv_buf].data() + (B_TOPK / 2) * D_sQ),
                            SmemLayoutKVTiles<NUM_tQ_TILES>{}
                        );
                        Tensor sV = make_tensor(
                            make_smem_ptr(plan.u.q_kv.kv[kv_buf].data()),
                            SmemLayoutV{}
                        );

                        plan.bar_prologue_kv[kv_buf].arrive_and_expect_tx(B_TOPK * D_K * sizeof(bf16));
                        plan.bar_prologue_kv[kv_buf].wait(kv_phase);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ss(tiled_mma_P_sQ, sQ, sK_sQ, tP, true);
                        ku::utcmma_ts(tiled_mma_P_tQ, tQ, sK_tQ, tP, false);
                        ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready[kv_buf], 1 | 2);
                        ku::tcgen05_after_thread_sync();

                        ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);
                        ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready[kv_buf], 1 | 2);
                        ku::tcgen05_after_thread_sync();

                        plan.bar_ds_ready.wait(round_phase);
                        ku::tcgen05_after_thread_sync();

                        plan.bar_k_dq_nope_ready.arrive_and_expect_tx(B_TOPK * D_V * sizeof(bf16));
                        plan.bar_k_dq_rope_ready.arrive_and_expect_tx(B_TOPK * D_ROPE * sizeof(bf16));
                        plan.bar_k_dq_nope_ready.wait(round_phase);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ss(tiled_mma_dQ, sDS_t, sK_dq_nope_t, tdQ, dq_clear);

                        plan.bar_k_dq_rope_ready.wait(round_phase);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t, sK_dq_rope_t, tdQ_RoPE, dq_clear);
                        ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dq_ready, kClusterMask2Cta);
                        ku::tcgen05_after_thread_sync();
                    }
                }
            }
        }

        if (local_warp_idx == kWg3KvValidWarp) {
            if (lane_idx < B_TOPK / 8) {
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

                    plan.bar_k_valid_free.wait((k & 1) ^ 1);
                    plan.is_k_valid[lane_idx] = is_ks_valid_mask;
                    plan.bar_k_valid_ready.arrive();
                }
            }
        }

        if (local_warp_idx == kWg3StoreSWarp) {
            Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
            auto thr_tma_s = tma_params.tma_S.get_slice(_0{});

            if (elect_one_sync()) {
                CUTE_NO_UNROLL
                for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                    if (k_block > 0) {
                        cute::tma_store_wait<0>();
                        plan.bar_s_store_done.arrive(static_cast<uint32_t>(cta_idx));
                    }
                    plan.bar_s_ready.wait(k_block & 1);
                    Tensor gS = flat_divide(
                        tma_params.tma_S.get_tma_tensor(tma_params.shape_S)(_, _, s_q_idx),
                        Shape<Int<B_H / 2>, Int<B_TOPK>>{}
                    )(_, _, cta_idx, k_block);
                    cute::copy(
                        tma_params.tma_S,
                        thr_tma_s.partition_S(sS),
                        thr_tma_s.partition_D(gS)
                    );
                    cute::tma_store_arrive();
                }
            }
        }

        if (local_warp_idx == kWg3StoreDsWarp) {
            Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});
            auto thr_tma_ds = tma_params.tma_dS.get_slice(_0{});

            if (elect_one_sync()) {
                CUTE_NO_UNROLL
                for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                    if (k_block > 0) {
                        cute::tma_store_wait<0>();
                        plan.bar_ds_store_done.arrive(static_cast<uint32_t>(cta_idx));
                    }
                    plan.bar_ds_store_ready.wait(k_block & 1);
                    Tensor gdS = flat_divide(
                        tma_params.tma_dS.get_tma_tensor(tma_params.shape_dS)(_, _, s_q_idx),
                        Shape<Int<B_H / 2>, Int<B_TOPK>>{}
                    )(_, _, cta_idx, k_block);
                    cute::copy(
                        tma_params.tma_dS,
                        thr_tma_ds.partition_S(sDS),
                        thr_tma_ds.partition_D(gdS)
                    );
                    cute::tma_store_arrive();
                }
            }
        }
    }

#endif
}

static void launch_dq_phase(const SparseAttnBwdParams& params) {
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

    auto shape_S = cute::make_shape(B_H, params.topk, params.s_q);
    auto tma_S = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.s),
            cute::make_layout(
                shape_S,
                cute::make_stride(params.stride_s_h_q, cute::_1{}, params.stride_s_s_q)
            )
        ),
        SmemLayoutS{}
    );

    auto shape_dS = cute::make_shape(B_H, params.topk, params.s_q);
    auto tma_dS = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.ds),
            cute::make_layout(
                shape_dS,
                cute::make_stride(params.stride_ds_h_q, cute::_1{}, params.stride_ds_s_q)
            )
        ),
        SmemLayoutdS{}
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
    CUtensorMap tensor_map_kv_nope = ku::make_tensor_map(
        {D_V, (uint64_t)params.s_kv},
        {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)},
        {64, 1},
        params.kv,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    );
    CUtensorMap tensor_map_kv_rope = ku::make_tensor_map(
        {D_ROPE, (uint64_t)params.s_kv},
        {(uint64_t)params.stride_kv_s_kv * sizeof(bf16)},
        {D_ROPE / 2, 1},
        (bf16*)params.kv + D_V,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    );

    using KernelTmaParams = TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ),
        decltype(shape_S), decltype(tma_S),
        decltype(shape_dS), decltype(tma_dS)
    >;

    KernelTmaParams tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        shape_S, tma_S,
        shape_dS, tma_dS,
        tensor_map_kv,
        tensor_map_kv_nope,
        tensor_map_kv_rope
    };

    auto kernel = &dq_phase_kernel<KernelTmaParams>;
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
void run_bwd_dq_phase_kernel(const SparseAttnBwdParams& params) {
    static_assert(DQK == D_QK);

    KU_ASSERT(params.d_qk == DQK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk > 0 && params.topk % B_TOPK == 0);

    sm100::bwd::head128::run_bwd_preprocess_delta_kernel<DQK>(params);
    launch_dq_phase(params);
}

}  // namespace sm100::bwd::head128_2kernels::dq
