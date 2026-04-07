#pragma once

#include "dkv_config.h"

#include <cstring>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

namespace sm100::bwd::head128_2kernels::dkv {

using namespace cute;

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
void atomic_add_64floats_unrolled(float* dst, const float* src) {
    atomic_add_32floats_unrolled(dst, src);
    atomic_add_32floats_unrolled(dst + 32, src + 32);
}

static constexpr int kThreadsPerWarp = 32;
static constexpr int kWarpsPerWarpgroup = 4;
static constexpr int kThreadsPerWarpgroup = kWarpsPerWarpgroup * kThreadsPerWarp;
static constexpr int kNumWarpgroups = 3;
static constexpr uint16_t kClusterMask2Cta = 0x3;

static_assert(NUM_THREADS == kNumWarpgroups * kThreadsPerWarpgroup, "NUM_THREADS must match the dKV warpgroup layout.");
// WG0/WG1 drain dKV to global memory; WG2 uses warp_idx 8/9/10/11 as MMA, S-TMA, dS-TMA, idle.

template<typename TmaParamsType>
__global__ __launch_bounds__(NUM_THREADS, 1) void dkv_phase_kernel(
    __grid_constant__ const SparseAttnBwdParams params,
    __grid_constant__ const TmaParamsType tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);

    const int cta_idx = blockIdx.x % 2;
    const int s_q_idx = blockIdx.x / 2;
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = tid % kThreadsPerWarp;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int local_warp_idx = warp_idx % kWarpsPerWarpgroup;
    if (s_q_idx >= params.s_q) {
        return;
    }

    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int topk_length = params.topk_length == nullptr ?
        params.topk :
        min(max(__ldg(params.topk_length + s_q_idx), 0), params.topk);
    const int num_k_pairs = max(params.topk / DKV_TILE_M, 1);
    const int* gIndices_s = params.indices + (int64_t)s_q_idx * params.stride_indices_s_q;

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_S.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dS.get_tma_descriptor());
    }

    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_q_nope_ready.init(1);
        plan.bar_q_rope_ready.init(1);
        plan.bar_dO_ready.init(1);
        CUTE_UNROLL
        for (int buf = 0; buf < NUM_S_DS_BUFS; ++buf) {
            plan.bar_s_ready[buf].init(1);
            plan.bar_ds_ready[buf].init(1);
            plan.bar_dkv_nope_ready[buf].init(1);
            plan.bar_dkv_rope_ready[buf].init(1);
        }
        plan.bar_dkv_nope_done.init(4 * kThreadsPerWarpgroup);
        plan.bar_dkv_rope_done.init(2 * kThreadsPerWarpgroup);
        fence_barrier_init();
    }

    cluster_sync();

    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            Tensor gQNoPE = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, cta_idx, s_q_idx);
            ku::launch_tma_copy(
                tma_params.tma_Q_nope,
                gQNoPE,
                sQNoPE,
                plan.bar_q_nope_ready,
                TMA::CacheHintSm90::EVICT_FIRST
            );

            Tensor gQRoPE = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, cta_idx, s_q_idx);
            ku::launch_tma_copy(
                tma_params.tma_Q_rope,
                gQRoPE,
                sQRoPE,
                plan.bar_q_rope_ready,
                TMA::CacheHintSm90::EVICT_FIRST
            );

            Tensor gdO = tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, cta_idx, s_q_idx);
            ku::launch_tma_copy(
                tma_params.tma_dO,
                gdO,
                sdO,
                plan.bar_dO_ready,
                TMA::CacheHintSm90::EVICT_FIRST
            );
        }

        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();

    const uint32_t tmem_base = plan.tmem_start_addr.data()[0];

    if (warp_idx == 9) {
        const bool issue_s_tma = elect_one_sync();
        CUTE_NO_UNROLL
        for (int k_pair = 0; k_pair < num_k_pairs; ++k_pair) {
            if (issue_s_tma) {
                const int buf = k_pair % NUM_S_DS_BUFS;
                const int phase = (k_pair / NUM_S_DS_BUFS) & 1;
                if (k_pair >= NUM_S_DS_BUFS) {
                    plan.bar_dkv_nope_ready[buf].wait(phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                }

                Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s[buf].data()), SmemLayoutS{});
                Tensor gS = tma_params.tma_S.get_tma_tensor(tma_params.shape_S)(_, _, cta_idx, k_pair, s_q_idx);
                ku::launch_tma_copy(
                    tma_params.tma_S,
                    gS,
                    sS,
                    plan.bar_s_ready[buf],
                    TMA::CacheHintSm90::EVICT_FIRST
                );
            }
        }
    } else if (warp_idx == 10) {
        const bool issue_ds_tma = elect_one_sync();
        CUTE_NO_UNROLL
        for (int k_pair = 0; k_pair < num_k_pairs; ++k_pair) {
            if (issue_ds_tma) {
                const int buf = k_pair % NUM_S_DS_BUFS;
                const int phase = (k_pair / NUM_S_DS_BUFS) & 1;
                if (k_pair >= NUM_S_DS_BUFS) {
                    plan.bar_dkv_rope_ready[buf].wait(phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                }

                Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds[buf].data()), SmemLayoutdS{});
                Tensor gdS = tma_params.tma_dS.get_tma_tensor(tma_params.shape_dS)(_, _, cta_idx, k_pair, s_q_idx);
                ku::launch_tma_copy(
                    tma_params.tma_dS,
                    gdS,
                    sDS,
                    plan.bar_ds_ready[buf],
                    TMA::CacheHintSm90::EVICT_FIRST
                );
            }
        }
    } else {
        if (warpgroup_idx < 2) {
            // TMEM ld row/half mapping follows the physical 4-warp lane ordering
            // within each transfer warpgroup.
            const int tmem_lane_128 = local_warp_idx * kThreadsPerWarp + lane_idx;
            const int row = tmem_lane_128 % DKV_ROWS_PER_CTA;
            const int half = (tmem_lane_128 / DKV_ROWS_PER_CTA) & 1;
            constexpr int COLS_PER_HALF = NOPE_COLS_PER_CTA / 2;
            constexpr int NOPE_COLS_PER_CLUSTER_HALF = NOPE_COLS_PER_CTA;
            constexpr int CHUNK_SIZE = 64;
            constexpr int NUM_CHUNKS = COLS_PER_HALF / CHUNK_SIZE;
            constexpr int ROPE_COLS_PER_HALF = D_ROPE / 2;
            static_assert(CHUNK_SIZE == 64);
            static_assert(NUM_CHUNKS == 2);
            static_assert(ROPE_COLS_PER_HALF == 32);

            CUTE_NO_UNROLL
            for (int k_pair = 0; k_pair < num_k_pairs; ++k_pair) {
                const int buf = k_pair % NUM_S_DS_BUFS;
                const int phase = (k_pair / NUM_S_DS_BUFS) & 1;
                const int row_global = (2 * k_pair + cta_idx) * DKV_ROWS_PER_CTA + row;
                int kv_idx = -1;
                if (row_global < topk_length) {
                    kv_idx = __ldg(gIndices_s + row_global);
                }
                const bool row_valid = kv_idx >= 0 && kv_idx < params.s_kv && kv_idx <= max_kv_i;

                plan.bar_dkv_nope_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();

                if (warpgroup_idx == 0) {
                    // WG0 drains the first 256 NoPE columns.
                    CUTE_UNROLL
                    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
                        float2 dkv_data[CHUNK_SIZE / 2];
                        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                        cutlass::arch::fence_view_async_tmem_load();
                        ku::tcgen05_before_thread_sync();

                        if (row_valid) {
                            // TiledMMA_dKV uses the same 2CTA permutation as forward TiledMMA_O:
                            // TMEM [0:128]   -> global [0:128]
                            // TMEM [128:256] -> global [256:384]
                            float* dst = params.dKV + (int64_t)kv_idx * params.stride_dKV_s_kv +
                                half * NOPE_COLS_PER_CLUSTER_HALF + chunk * CHUNK_SIZE;
                            atomic_add_64floats_unrolled(dst, reinterpret_cast<float*>(dkv_data));
                        }
                    }

                    plan.bar_dkv_nope_done.arrive(static_cast<uint32_t>(0));
                } else {
                    // WG1 drains the remaining 256 NoPE columns and the RoPE slice.
                    CUTE_UNROLL
                    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
                        float2 dkv_data[CHUNK_SIZE / 2];
                        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + 128 + chunk * CHUNK_SIZE, dkv_data);
                        cutlass::arch::fence_view_async_tmem_load();
                        ku::tcgen05_before_thread_sync();

                        if (row_valid) {
                            // TMEM [256:384] -> global [128:256]
                            // TMEM [384:512] -> global [384:512]
                            float* dst = params.dKV + (int64_t)kv_idx * params.stride_dKV_s_kv +
                                COLS_PER_HALF + half * NOPE_COLS_PER_CLUSTER_HALF + chunk * CHUNK_SIZE;
                            atomic_add_64floats_unrolled(dst, reinterpret_cast<float*>(dkv_data));
                        }
                    }

                    plan.bar_dkv_nope_done.arrive(static_cast<uint32_t>(0));

                    plan.bar_dkv_rope_ready[buf].wait(phase);
                    ku::tcgen05_after_thread_sync();

                    float2 dkv_rope_data[ROPE_COLS_PER_HALF / 2];
                    ku::tmem_ld_32dp32bNx<ROPE_COLS_PER_HALF>(tmem_cols::dKV_RoPE, dkv_rope_data);
                    cutlass::arch::fence_view_async_tmem_load();
                    ku::tcgen05_before_thread_sync();

                    if (row_valid) {
                        float* dst = params.dKV + (int64_t)kv_idx * params.stride_dKV_s_kv +
                            D_V + half * ROPE_COLS_PER_HALF;
                        atomic_add_32floats_unrolled(dst, reinterpret_cast<float*>(dkv_rope_data));
                    }

                    plan.bar_dkv_rope_done.arrive(static_cast<uint32_t>(0));
                }
            }
        }

        if (cta_idx == 0 && warp_idx == 8 && elect_one_sync()) {
            plan.bar_q_nope_ready.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
            plan.bar_q_rope_ready.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
            plan.bar_dO_ready.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));

            plan.bar_q_nope_ready.wait(0);
            plan.bar_q_rope_ready.wait(0);
            plan.bar_dO_ready.wait(0);
            ku::tcgen05_after_thread_sync();

            TiledMMA_dKV tiled_mma_dKV{};
            TiledMMA_dKV_RoPE tiled_mma_dKV_RoPE{};
            Tensor tdKV = partition_fragment_C(tiled_mma_dKV, Shape<Int<DKV_ROWS_PER_CTA>, Int<D_V>>{});
            Tensor tdKV_RoPE = partition_fragment_C(tiled_mma_dKV_RoPE, Shape<Int<DKV_ROWS_PER_CTA>, Int<D_ROPE>>{});
            tdKV.data().get() = tmem_cols::dKV;
            tdKV_RoPE.data().get() = tmem_cols::dKV_RoPE;

            Tensor sdO_mma_full = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO_MMA{});
            Tensor sQNoPE_mma_full = make_tensor(make_smem_ptr(plan.q_nope.data()), SmemLayoutQNoPE_MMA{});
            Tensor sQRoPE_mma_full = make_tensor(make_smem_ptr(plan.q_rope.data()), SmemLayoutQRoPE_MMA{});

            CUTE_NO_UNROLL
            for (int k_pair = 0; k_pair < num_k_pairs; ++k_pair) {
                const int buf = k_pair % NUM_S_DS_BUFS;
                const int phase = (k_pair / NUM_S_DS_BUFS) & 1;
                const int round_phase = k_pair & 1;

                Tensor sS_mma = make_tensor(make_smem_ptr(plan.s_ds.s[buf].data()), SmemLayoutS_MMA{});
                Tensor sDS_mma = make_tensor(make_smem_ptr(plan.s_ds.ds[buf].data()), SmemLayoutdS_MMA{});
                plan.bar_s_ready[buf].arrive_and_expect_tx(B_H * DKV_TILE_M * sizeof(bf16));

                if (k_pair > 0) {
                    plan.bar_dkv_nope_done.wait(round_phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                }

                plan.bar_s_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sS_mma, sdO_mma_full, tdKV, true);

                plan.bar_ds_ready[buf].arrive_and_expect_tx(B_H * DKV_TILE_M * sizeof(bf16));
                plan.bar_ds_ready[buf].wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sDS_mma, sQNoPE_mma_full, tdKV, false);
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_nope_ready[buf], kClusterMask2Cta);
                ku::tcgen05_after_thread_sync();

                if (k_pair > 0) {
                    plan.bar_dkv_rope_done.wait(round_phase ^ 1);
                    ku::tcgen05_after_thread_sync();
                }

                ku::utcmma_ss(tiled_mma_dKV_RoPE, sDS_mma, sQRoPE_mma_full, tdKV_RoPE, true);
                ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dkv_rope_ready[buf], kClusterMask2Cta);
                ku::tcgen05_after_thread_sync();
            }
        }
    }

    cluster_sync();
    if (warp_idx == 0) {
        TMEM::Allocator2Sm().free(tmem_base, 512);
    }
#endif
}

static void launch_dkv_phase(const SparseAttnBwdParams& params) {
    const int num_k_pairs = max(params.topk / DKV_TILE_M, 1);

    auto shape_Q_nope = cute::make_shape(B_H, NOPE_COLS_PER_CTA, 2, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q),
            cute::make_layout(
                shape_Q_nope,
                cute::make_stride(
                    params.stride_q_h_q,
                    cute::_1{},
                    Int<NOPE_COLS_PER_CTA>{},
                    params.stride_q_s_q
                )
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = cute::make_shape(B_H, ROPE_COLS_PER_CTA, 2, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.q + D_V),
            cute::make_layout(
                shape_Q_rope,
                cute::make_stride(
                    params.stride_q_h_q,
                    cute::_1{},
                    Int<ROPE_COLS_PER_CTA>{},
                    params.stride_q_s_q
                )
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_dO = cute::make_shape(B_H, NOPE_COLS_PER_CTA, 2, params.s_q);
    auto tma_dO = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.dO),
            cute::make_layout(
                shape_dO,
                cute::make_stride(
                    params.stride_dO_h_q,
                    cute::_1{},
                    Int<NOPE_COLS_PER_CTA>{},
                    params.stride_dO_s_q
                )
            )
        ),
        SmemLayoutdO{}
    );

    auto shape_S = cute::make_shape(B_H, DKV_ROWS_PER_CTA, 2, num_k_pairs, params.s_q);
    auto tma_S = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.s),
            cute::make_layout(
                shape_S,
                cute::make_stride(
                    params.stride_s_h_q,
                    cute::_1{},
                    Int<DKV_ROWS_PER_CTA>{},
                    Int<DKV_TILE_M>{},
                    params.stride_s_s_q
                )
            )
        ),
        SmemLayoutS{}
    );

    auto shape_dS = cute::make_shape(B_H, DKV_ROWS_PER_CTA, 2, num_k_pairs, params.s_q);
    auto tma_dS = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)params.ds),
            cute::make_layout(
                shape_dS,
                cute::make_stride(
                    params.stride_ds_h_q,
                    cute::_1{},
                    Int<DKV_ROWS_PER_CTA>{},
                    Int<DKV_TILE_M>{},
                    params.stride_ds_s_q
                )
            )
        ),
        SmemLayoutdS{}
    );

    using KernelTmaParams = TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_S), decltype(tma_S),
        decltype(shape_dS), decltype(tma_dS)
    >;

    KernelTmaParams tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_dO, tma_dO,
        shape_S, tma_S,
        shape_dS, tma_dS
    };

    auto kernel = &dkv_phase_kernel<KernelTmaParams>;
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
void run_bwd_dkv_phase_kernel(const SparseAttnBwdParams& params) {
    static_assert(DQK == D_QK);

    KU_ASSERT(params.d_qk == DQK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk >= DKV_TILE_M, "dKV two-kernel path requires topk >= %d, got %d", DKV_TILE_M, params.topk);
    KU_ASSERT(params.topk % DKV_TILE_M == 0, "dKV two-kernel path requires topk to be a multiple of %d, got %d", DKV_TILE_M, params.topk);
    KU_ASSERT(params.q != nullptr && params.dO != nullptr);
    KU_ASSERT(params.s != nullptr && params.ds != nullptr);
    KU_ASSERT(params.dKV != nullptr);

    launch_dkv_phase(params);
}

}  // namespace sm100::bwd::head128_2kernels::dkv
