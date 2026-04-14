#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"

#include "config.h"
#include "sm100/prefill/sparse/common_subroutine.h"

namespace sm100::fwd::head128 {

using namespace cute;

CUTE_DEVICE int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return val;
}

/*
Pipeline Overview:

| Copy |    MMA    |   Scale & Exp   |

K0
V0
        P0 = QK0^T
K1                  S0 = exp(P0)
                    scale(O) w.r.t P0
        P1 = QK1^T
K2                  S1 = exp(P1)
        O += S0V0
V1                  scale(O) w.r.t P1
        P2 = QK2^T
K3                  S2 = exp(P2)
        O += S1V1
V2                  scale(O) w.r.t P2
        P3 = QK3^T
K4                  S3 = exp(P3)
        O += S2V2
V3                  scale(O) w.r.t P3

...

        O += S(n-3)V(n-3)
V(n-2)              scale(O) w.r.t P(n-2)
        P(n-1) = QK(n-1)^T
                   S(n-1) = exp(P(n-1))
        O += S(n-2)V(n-2)
V(n-1)             scale(O) w.r.t P(n-1)
        O += S(n-1)V(n-1)
*/

template<int D_QK>
template<typename TmaParams>
__device__ void
KernelTemplate<D_QK>::sparse_attn_fwd_kernel_devfunc(const SparseAttnFwdParams &params, const TmaParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int cta_idx = blockIdx.x % 2;
    const int s_q_idx = blockIdx.x / 2;
    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;

    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
        if (params.p_out != nullptr) {
            cute::prefetch_tma_descriptor(tma_params.tma_P.get_tma_descriptor());
        }
    }

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ_full = make_tensor(make_smem_ptr(plan.u.q_full.data()), SmemLayoutQTiles<D_Q/64>{});

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    // Allocate tmem tensors
    TiledMMA tiled_mma_P_tQ = TiledMMA_P_tQ{};
    TiledMMA tiled_mma_P_sQ = TiledMMA_P_sQ{};
    TiledMMA tiled_mma_O = TiledMMA_O{};
    Tensor tP = partition_fragment_C(tiled_mma_P_tQ, Shape<Int<B_H/2>, Int<B_TOPK>>{});
    Tensor tQr = tiled_mma_P_tQ.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P_tQ, Shape<Int<B_H/2>, Int<D_tQ>>{})
    );
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H/2>, Int<D_V>>{});
    tP.data().get() = tmem_cols::p;
    tQr.data().get() = tmem_cols::q;
    tO.data().get() = tmem_cols::o;

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Initialize barriers
            plan.bar_prologue_q.init(1);
            plan.bar_prologue_utccp.init(1);
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_qk_part_done[i].init(1);
                plan.bar_qk_done[i].init(1);
                plan.bar_sv_part_done[i].init(1);
                plan.bar_sv_done[i].init(1);
                plan.bar_k_part0_ready[i].init(1);
                plan.bar_k_part1_ready[i].init(1);
                plan.bar_v_part0_ready[i].init(1);
                plan.bar_v_part1_ready[i].init(1);
                plan.bar_p_free[i].init(128*2);
                plan.bar_so_ready[i].init(128*2);
                plan.bar_k_valid_ready[i].init(16);
                plan.bar_k_valid_free[i].init(128);
            }
            fence_barrier_init();
        }
    }

    cute::cluster_sync();   // We must add a cluster_sync() here, or TMA from CTA1 may launch before barrier initialization in CTA0

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Copy Q
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ_full, plan.bar_prologue_q, TMA::CacheHintSm90::EVICT_FIRST);
        }

        // Initialize TMEM
        cute::TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    }

    __syncthreads();    // Wait for TMEM allocation

    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<144>();
        // Scale & Exp warps

        // The following three numbers are 
        // - mi: max_logits used to scale Pi (i.e. O := exp2(Pi*scale - mi) @ V)
        // - li: sumexp, i.e. li := sum(exp(Pi*scale - mi))
        // - real_mi: real max logits, i.e. real_mi := max(Pi*scale)
        // where Pi is the i-th row of P, P := QK^T
        // mi and real_mi are always consistent within the two threads that
        // controls one row (i.e. thread 0+64, 1+65, 2+66, ...) after every update
        float mi = MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        const float2 scale = float2 {params.sm_scale_div_log2, params.sm_scale_div_log2};
        uint128_t* sS_base = (uint128_t*)plan.s.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8);
        float* sP_base = plan.p + idx_in_warpgroup%64*4 + (idx_in_warpgroup/64)*((B_H/2)*(B_TOPK/2));

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            // Wait for P
            plan.bar_qk_done[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            ku::tcgen05_after_thread_sync();

            // Load P
            float2 p[(B_TOPK/2)/2];
            ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::p, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            plan.bar_p_free[k%NUM_BUFS].arrive(0u);

            // Mask
            plan.bar_k_valid_ready[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            // The following code enables NVCC to use R2P instruction
            // Although we perform 2x LDS.32 instructions here, don't worry, NVCC will
            // convert them to one LDS.64 instruction. However, if we write LDS.64
            // here, NVCC won't use R2P.
            uint32_t is_k_valid_lo = *(uint32_t*)(plan.is_k_valid[k%NUM_BUFS] + (idx_in_warpgroup>=64?B_TOPK/8/2:0));
            uint32_t is_k_valid_hi = *(uint32_t*)(plan.is_k_valid[k%NUM_BUFS] + (idx_in_warpgroup>=64?B_TOPK/8/2:0) + 4);
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                if (!(is_k_valid_lo >> i & 1))
                    p_float[i] = -CUDART_INF_F;
            }
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                if (!(is_k_valid_hi >> i & 1))
                    p_float[i+(B_TOPK/2)/2] = -CUDART_INF_F;
            }

            // Output P matrix (after mask) via shared memory + TMA
            if (params.p_out != nullptr) {
                if (k > 0) {
                    if (warp_idx == 0 && elect_one_sync()) {
                        cute::tma_store_wait<0>();
                    }
                    NamedBarrier::arrive_and_wait(128, 1);
                }

                int smem_h = idx_in_warpgroup % 64;  // head index in shared memory
                int smem_t_base = (idx_in_warpgroup < 64) ? 0 : 64;  // topk offset
                float* sP_row = plan.p + smem_h * B_TOPK + smem_t_base;

                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2); i += 4) {
                    cutlass::Array<float, 4> p_vec;
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        p_vec[j] = p_float[i + j];
                    }
                    // 128-bit store to shared memory
                    *reinterpret_cast<cutlass::Array<float, 4>*>(sP_row + i) = p_vec;
                }

                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, 1);

                if (warp_idx == 0 && elect_one_sync()) {
                    Tensor sP = make_tensor(make_smem_ptr(plan.p), SmemLayoutP{});
                    Tensor tma_gP = flat_divide(
                        tma_params.tma_P.get_tma_tensor(tma_params.shape_P)(_, _, s_q_idx),
                        Shape<Int<B_H/2>, Int<B_TOPK>>{}
                    )(_, _, cta_idx, _);
                    Tensor sP_divided = flat_divide(
                        sP,
                        Shape<Int<B_H/2>, Int<B_TOPK>>{}
                    )(_, _, _0{}, _);
                    auto thr_tma_P = tma_params.tma_P.get_slice(_0{});

                    cute::copy(
                        tma_params.tma_P,
                        thr_tma_P.partition_S(sP_divided(_, _, k)),
                        thr_tma_P.partition_D(tma_gP(_, _, k))
                    );
                    cute::tma_store_arrive();
                }
            }

            // Get rowwise max of Pi
            float cur_pi_max = -CUDART_INF_F;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2); i += 1) {
                cur_pi_max = max(cur_pi_max, p_float[i]);
            }
            cur_pi_max *= params.sm_scale_div_log2;

            plan.bar_k_valid_free[k%NUM_BUFS].arrive();

            NamedBarrier::arrive_and_wait(128, 0);  // Wait for rowwise_max_buf and sP to be ready
            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            NamedBarrier::arrive_and_wait(128, 0);  // TODO Name these barriers
            cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
            real_mi = max(real_mi, cur_pi_max);
            bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
            // By this point:
            // - cur_pi_max, real_mi, and mi is identical within each row (i.e. thread 0+64, 1+65, ...)
            // - should_scale_o is identical among threads 0~31+64~95; and is identical among threads 32~63+96~127


            // Calc scale factor, and scale li
            float new_max, scale_for_old;
            if (!should_scale_o) {
                // Don't scale O
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;   // mi is still identical within each row
            li *= scale_for_old;

            // Calculate S
            __nv_bfloat162 s[(B_TOPK/2)/2];
            float2 neg_new_max = float2 {-new_max, -new_max};
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                float2 d = ku::float2_fma(p[i], scale, neg_new_max);
                d.x = exp2f(d.x);
                d.y = exp2f(d.y);
                li += d.x + d.y;    // NOTE: Theoretically we could use FFMA2 here but actually this is faster...
                s[i] = __float22bfloat162_rn(d);
            }

            // Wait for last SV gemm, write S
            if (k > 0) {
                plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
            }
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK/2/8; i += 1) {
                sS_base[64*i] = *(uint128_t*)(s + i*4);
            }

            // Scale O
            if (k > 0 && should_scale_o) {
                float2 scale_for_old_float2 = float2 {scale_for_old, scale_for_old}; 
                // plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);   // NOTE: We have waited for last SV gemm before
                ku::tcgen05_after_thread_sync();

                static constexpr int CHUNK_SIZE = 32;
                float2 o[CHUNK_SIZE/2];
                CUTE_UNROLL
                for (int chunk_idx = 0; chunk_idx < (D_V/2)/CHUNK_SIZE; ++chunk_idx) {
                    // Load O
                    ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::o + chunk_idx*CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_load();

                    // Mult
                    for (int i = 0; i < CHUNK_SIZE/2; ++i) {
                        o[i] = ku::float2_mul(o[i], scale_for_old_float2);
                    }

                    // Store O
                    ku::tmem_st_32dp32bNx<CHUNK_SIZE>(tmem_cols::o + chunk_idx*CHUNK_SIZE, o);
                    cutlass::arch::fence_view_async_tmem_store();
                }
                ku::tcgen05_before_thread_sync();
            }
            
            fence_view_async_shared();
            plan.bar_so_ready[k%NUM_BUFS].arrive(0u);
        }

        if (params.p_out != nullptr) {
            if (warp_idx == 0 && elect_one_sync()) {
                cute::tma_store_wait<0>();
            }
            NamedBarrier::arrive_and_wait(128, 1);
        }

        // Epilogue

        if (real_mi == -CUDART_INF_F) {
            // real_mi == -CUDART_INF_F <=> No valid TopK indices
            // We set li to 0 to fit the definition that li := exp(x[i] - mi)
            li = 0.0f;
            mi = -CUDART_INF_F;
        }
        
        // Exchange li
        plan.rowwise_li_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, 0);
        li += plan.rowwise_li_buf[idx_in_warpgroup^64];

        // Store mi and li
        if (idx_in_warpgroup < 64) {
            int global_index = s_q_idx*params.h_q + cta_idx*(B_H/2) + idx_in_warpgroup;
            float cur_lse = logf(li) + mi*CUDART_LN2_F;
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }

        // Wait for the last GEMM
        plan.bar_sv_done[(num_k_blocks-1)%NUM_BUFS].wait(((num_k_blocks-1)/NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        // Store O
        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + cta_idx*B_H/2 + (idx_in_warpgroup%64))*CUDART_L2E_F;
        float output_scale = __fdividef(1.0f, li + exp2f(attn_sink - mi));
        Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
        constexpr int B_EPI = 64;
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H/2>, Int<B_EPI>>{}
        )(_, _, cta_idx, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<B_H/2>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        float2 o[B_EPI/2];
        bool have_valid_indices = __any_sync(0xffffffff, li != 0);  // Prevent some threads' li == 0 and some threads' li != 0 which lead to deadlock during ku::tmem_ld
        if (!have_valid_indices) {
            // If there are no valid indices, we set o[i] to 0 and don't load from TMEM
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/2; ++i)
                o[i].x = o[i].y = 0.0f;
            output_scale = 1.0f;
        }

        float2 output_scale_float2 = make_float2(output_scale, output_scale);

        CUTE_UNROLL
        for (int k = 0; k < (D_V/2)/B_EPI; ++k) {
            // Load O from tO
            if (have_valid_indices) {
                ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::o + k*B_EPI, o);
                cutlass::arch::fence_view_async_tmem_load();
            }

            // Convert and store
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/8; ++i) {
                __nv_bfloat162 o_bf16[4];
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j) {
                    float2 d = ku::float2_mul(o[i*4+j], output_scale_float2);
                    o_bf16[j] = __float22bfloat162_rn(d);
                }
                int smem_row = idx_in_warpgroup % 64;
                int smem_col = (idx_in_warpgroup/64)*(D_V/2) + k*B_EPI + i*8;
                *(uint128_t*)(&sO(smem_row, smem_col)) = *(uint128_t*)(o_bf16);
            }

            // Sync
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, 0);
            
            if (warp_idx == 0 && elect_one_sync()) {
                cute::copy(
                    tma_params.tma_O,
                    thr_tma.partition_S(sO_divided(_, _, k)),
                    thr_tma.partition_D(tma_gO(_, _, k))
                );
            }
            if (warp_idx == 1 && elect_one_sync()) {
                int k2 = k + (D_V/B_EPI/2);
                cute::copy(
                    tma_params.tma_O,
                    thr_tma.partition_S(sO_divided(_, _, k2)),
                    thr_tma.partition_D(tma_gO(_, _, k2))
                );
            }
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator2Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // Producer warp for K
        cutlass::arch::warpgroup_reg_dealloc<96>();
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/2)/4/NUM_WARPS;
        if (elect_one_sync()) {
            bf16* sK_base = plan.u.s.k.data() + warp_idx*4*64;

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK + cta_idx*(B_TOPK/2)) + local_row*NUM_WARPS + warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;
                    
                auto load_part_ki = [&](transac_bar_t &bar, int local_col_start, int local_col_end) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = local_col_start; local_col < local_col_end; ++local_col)
                            ku::tma_gather4_cta_group_2<true>(
                                &(tma_params.tensor_map_kv),
                                bar,
                                sK_base + local_row*(4*NUM_WARPS)*64 + local_col*((B_TOPK/2)*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                    }
                };

                int cur_buf = k%NUM_BUFS;
                if (k > 0) {
                    plan.bar_qk_part_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
                }
                if (!should_skip_tma) {
                    load_part_ki(plan.bar_k_part0_ready[cur_buf], 0, D_sQ/64);
                } else {
                    // NOTE: TMA has performance issues when all indices are the same (even if those indices are invalid), so we detect whether all indices in our block are invalid (by inspecting their MIN and MAX, for performance reasons), and skip the copy if all indices are invalid.
                    // NOTE: We can also skip the initial zero-fill procedure (which prevents NaN from appearing in K/V buf if the first TMA copy is skipped) by disabling skipping on the first NUM_BUFS TMAs.
                    // NOTE: We only do this for K to save some checking overhead, since after doing this for K, cases where topk indices are all invalid are faster than the other cases
                    plan.bar_k_part0_ready[cur_buf].complete_transaction(0u, NUM_LOCAL_ROWS_PER_WARP*4*D_sQ*sizeof(bf16), 1u);
                }

                if (k > 0) {
                    plan.bar_qk_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
                }
                if (!should_skip_tma) {
                    load_part_ki(plan.bar_k_part1_ready[cur_buf], D_sQ/64, D_K/64);
                } else {
                    plan.bar_k_part1_ready[cur_buf].complete_transaction(0u, NUM_LOCAL_ROWS_PER_WARP*4*D_tQ*sizeof(bf16), 1u);
                }
            }
        }
    } else if (warpgroup_idx == 2) {
        // Producer warps for V
        cutlass::arch::warpgroup_reg_dealloc<96>();
        int warp_idx = cutlass::canonical_warp_idx_sync() - 8;
        constexpr int NUM_WARPS = 4;

        if (elect_one_sync()) {
            // Wait for UTCCP
            plan.bar_prologue_utccp.wait(0);

            bf16* sV_base = plan.u.s.v.data() + warp_idx*4*64;

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                auto load_part_vi = [&](transac_bar_t &bar, int local_row_start, int local_row_end) {
                    CUTE_UNROLL
                    for (int local_row = local_row_start; local_row < local_row_end; ++local_row) {
                        int4 token_idxs = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx);
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < (D_V/2)/64; ++local_col)
                            ku::tma_gather4_cta_group_2<true>(
                                &(tma_params.tensor_map_kv),
                                bar,
                                sV_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64 + (cta_idx?256:0),
                                token_idxs,
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                    }
                };

                int cur_buf = k%NUM_BUFS;
                if (k > 0) {
                    plan.bar_sv_part_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
                }
                load_part_vi(plan.bar_v_part0_ready[cur_buf], 0, (B_TOPK/2)/4/NUM_WARPS);

                if (k > 0) {
                    plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
                }
                load_part_vi(plan.bar_v_part1_ready[cur_buf], (B_TOPK/2)/4/NUM_WARPS, B_TOPK/4/NUM_WARPS);
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<168>();
        
        // MMA warp
        if (cta_idx == 0 && warp_idx == 12 && elect_one_sync()) {
            // S -> T copy for Q
            UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.u.q_full.data() + (B_H/2)*D_sQ),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_H/2>, Int<64>>{}
                    )
                )
            );
            plan.bar_prologue_q.arrive_and_expect_tx(B_H*D_K*sizeof(bf16));
            plan.bar_prologue_q.wait(0);
            ku::tcgen05_after_thread_sync();
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < NUM_tQ_TILES; ++tile_idx) {
                // A tile is 64 rows * 64 cols (128B)
                CUTE_UNROLL
                for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
                    // A subtile is 64 rows * 8 cols (128b)
                    SM100_UTCCP_2x64dp128bitlw0213_2cta::copy(
                        sQ_desc + tile_idx*((B_H/2)*128/16) + subtile_idx*(16/16),   // Remember that 4 LSBs are not included
                        tmem_cols::q + tile_idx*32 + subtile_idx*4
                    );
                }
            }
            ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_prologue_utccp, 1|2);

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                if (k < num_k_blocks) {
                    // Pi = QKi^T
                    int cur_buf = k%NUM_BUFS;
                    Tensor sQl = make_tensor(make_smem_ptr(plan.u.s.sq.data()), SmemLayoutQTiles<NUM_sQ_TILES>{});
                    Tensor sKl = make_tensor(make_smem_ptr(plan.u.s.k.data()), SmemLayoutKTiles<NUM_sQ_TILES>{});
                    Tensor sKr = make_tensor(make_smem_ptr(plan.u.s.k.data()+64*D_sQ), SmemLayoutKTiles<NUM_tQ_TILES>{});

                    // Wait for K (part0)
                    plan.bar_k_part0_ready[cur_buf].arrive_and_expect_tx(B_TOPK*D_sQ*sizeof(bf16));
                    plan.bar_k_part0_ready[cur_buf].wait((k/NUM_BUFS)&1);
                    if (k > 0) {
                        plan.bar_p_free[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
                    }
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_P_sQ, sQl, sKl, tP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_part_done[cur_buf], 1|2);

                    // Wait for K (part1)
                    plan.bar_k_part1_ready[cur_buf].arrive_and_expect_tx(B_TOPK*(D_K-D_sQ)*sizeof(bf16));
                    plan.bar_k_part1_ready[cur_buf].wait((k/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ts(tiled_mma_P_tQ, tQr, sKr, tP, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_qk_done[cur_buf], 1|2);
                }
                if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutSTiles<2>{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.s.v.data()), SmemLayoutV{});
                    Tensor sS_divided = flat_divide(sS, Tile<Int<B_H/2>, _64>{})(_, _, _0{}, _);    // (B_H/2, 64, 2)
                    Tensor sV_divided = flat_divide(sV, Tile<Int<D_V/2>, _64>{})(_, _, _0{}, _);  // (D_V/2, 64, 2)

                    // Wait for S(i-1) and O to be scaled
                    plan.bar_so_ready[cur_buf].wait(((k-1)/NUM_BUFS)&1);

                    // Wait for V (part0), and issue O += sS @ sV
                    plan.bar_v_part0_ready[cur_buf].arrive_and_expect_tx((B_TOPK/2)*D_V*sizeof(bf16));
                    plan.bar_v_part0_ready[cur_buf].wait(((k-1)/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_O, sS_divided(_, _, _0{}), sV_divided(_, _, _0{}), tO, k == 1);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_sv_part_done[cur_buf], 1|2);

                    // Wait for V (part1), and issue O += sS @ sV
                    plan.bar_v_part1_ready[cur_buf].arrive_and_expect_tx((B_TOPK/2)*D_V*sizeof(bf16));
                    plan.bar_v_part1_ready[cur_buf].wait(((k-1)/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_O, sS_divided(_, _, _1{}), sV_divided(_, _, _1{}), tO, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_sv_done[cur_buf], 1|2);
                }
            }
        } else if (warp_idx == 13) {
            // KV valid loading warp
            static_assert(B_TOPK == 128);
            if (lane_idx < 16) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int cur_buf = k%NUM_BUFS;
                    int32x8_t indices = ldg_256_indices(gIndices + k*B_TOPK + lane_idx*8);
                    auto is_valid = [&](int rel_pos_in_lane, int index) -> char {
                        int abs_pos = k*B_TOPK + lane_idx*8 + rel_pos_in_lane;
                        return index >= 0 && index < params.s_kv && index <= max_kv_i && abs_pos < topk_length;
                    };
                    char is_ks_valid_mask = \
                        is_valid(7, indices.a7) << 7 | 
                        is_valid(6, indices.a6) << 6 | 
                        is_valid(5, indices.a5) << 5 |
                        is_valid(4, indices.a4) << 4 |
                        is_valid(3, indices.a3) << 3 |
                        is_valid(2, indices.a2) << 2 |
                        is_valid(1, indices.a1) << 1 |
                        is_valid(0, indices.a0) << 0;

                    plan.bar_k_valid_free[cur_buf].wait((k/NUM_BUFS)&1^1);
                    plan.is_k_valid[cur_buf][lane_idx] = is_ks_valid_mask;
                    plan.bar_k_valid_ready[cur_buf].arrive();
                }
            }
        }
    }


#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    static_assert(D_QK == 576 || D_QK == 512);
    using Kernel = KernelTemplate<D_QK>;

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % Kernel::B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.h_q == Kernel::B_H);  // To save some calculation
    KU_ASSERT(params.d_qk == D_QK);

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM100_TMA_2SM_LOAD_NOSPLIT{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        (typename Kernel::template SmemLayoutQTiles<D_QK/64>){}
    );

    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.d_v, _1{}, params.h_q*params.d_v)
            )
        ),
        (typename Kernel::template SmemLayoutOTiles<1>){}
    );

    auto shape_P = make_shape(params.h_q, params.topk, params.s_q);
    auto tma_P = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((float*)params.p_out),
            make_layout(
                shape_P,
                make_stride(params.topk, _1{}, params.h_q*params.topk)
            )
        ),
        Layout<Shape<Int<Kernel::B_H/2>, Int<Kernel::B_TOPK>>, Stride<Int<Kernel::B_TOPK>, _1>>{}
    );

    CUtensorMap tensor_map_kv;
    {
        uint64_t size[2] = {D_QK, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            params.kv,
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

    TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_O), decltype(tma_O),
        decltype(shape_P), decltype(tma_P)
    > tma_params = {
        shape_Q, tma_Q,
        shape_O, tma_O,
        shape_P, tma_P,
        tensor_map_kv
    };
    auto kernel = &sparse_attn_fwd_kernel<Kernel, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3(2*params.s_q, 1, 1),
        dim3(Kernel::NUM_THREADS, 1, 1),
        dim3(2, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    ));
}

}

namespace sm100::fwd::head64 {

using namespace cute;

/*
Pipeline Overview:

| Copy |    MMA    |   Scale & Exp   |

KV0
KV1
KV2
        P0 = QK0^T
                    S0 = exp(P0)
                    scale(O) w.r.t P0
        P1 = QK1^T
                    S1 = exp(P1)
        O += S0V0
KV3                 scale(O) w.r.t P1
        P2 = QK2^T
                    S2 = exp(P2)
        O += S1V1
KV4                 scale(O) w.r.t P2
        P3 = QK3^T
                    S3 = exp(P3)
        O += S2V2
KV5                 scale(O) w.r.t P3

...

        O += S(n-3)V(n-3)
                    scale(O) w.r.t P(n-2)
        P(n-1) = QK(n-1)^T
                   S(n-1) = exp(P(n-1))
        O += S(n-2)V(n-2)
                   scale(O) w.r.t P(n-1)
        O += S(n-1)V(n-1)
*/

// using FwdMode = SparseAttnFwdMode;

template<bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 0)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
    // return;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // Grid shape: [s_q, 1, 1]

    const int s_q_idx = blockIdx.x;
    const int max_kv_i = params.q_start_index_s + s_q_idx;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    // Allocate tmem tensors
    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_O = TiledMMA_O{};
    // NOTE These tXXX tensors are only for a forged layout (so that CuTe is able to generate correct address in cute::gemm)
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    Tensor tQ_nope_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );
    Tensor tQ_nope_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );
    Tensor tQ_rope = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<64/2>>{})
    );
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
    tP.data().get() = tmem_cols::P;
    tQ_nope_part0.data().get() = tmem_cols::Q;
    tQ_nope_part1.data().get() = tmem_cols::Q + 64;
    tQ_rope.data().get() = tmem_cols::Q_RoPE;
    tO.data().get() = tmem_cols::O;

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Copy Q
            if constexpr (HAVE_ROPE) {
                cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
            }
            cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());

            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            fence_barrier_init();
            
            if constexpr (HAVE_ROPE) {
                Tensor gQ_rope = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx);
                Tensor sQ_rope = make_tensor(make_smem_ptr(plan.s_q_rope.q_rope.data()), SmemLayoutQRoPE{});
                ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
            }

            Tensor gQ_nope = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx);
            Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data()), SmemLayoutQNoPE{});
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
            
            // Initialize other barriers
            plan.bar_prologue_utccp_rope.init(1);
            plan.bar_prologue_utccp_nope.init(1);
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_qk_nope_done[i].init(1);
                plan.bar_sv_done[i].init(1);
                plan.bar_kv_nope_ready[i][0].init(1);
                plan.bar_kv_nope_ready[i][1].init(1);
                plan.bar_k_valid_ready[i].init(B_TOPK/8);
                plan.bar_k_valid_free[i].init(128);
                plan.bar_p_free[i].init(128);
            }
            plan.bar_so_ready.init(128);
            plan.bar_qk_rope_done.init(1);
            plan.bar_kv_rope_ready.init(64);
            fence_barrier_init();
        }

        // Initialize TMEM
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    if (warpgroup_idx == 0) {
        // Scale & Exp warps

        // The following three numbers are 
        // - mi: max_logits used to scale Pi (i.e. O := exp2(Pi*scale - mi) @ V)
        // - li: sumexp, i.e. li := sum(exp(Pi*scale - mi))
        // - real_mi: real max logits, i.e. real_mi := max(Pi*scale)
        // where Pi is the i-th row of P, P := QK^T
        // mi and real_mi are always consistent within the two threads that
        // controls one row (i.e. thread 0+64, 1+65, 2+66, ...) after every update
        float mi = MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        bf16* sS_base = plan.s_q_rope.s + lane_idx*8 + (warp_idx&1)*(B_H/2)*8 + (warp_idx/2)*B_H*(B_TOPK/2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            // Wait for P
            NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync+(warp_idx&1));
            plan.bar_qk_nope_done[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            plan.bar_k_valid_ready[k%NUM_BUFS].wait((k/NUM_BUFS)&1);    // Put the barrier wait here for more code reordering space
            ku::tcgen05_after_thread_sync();
            
            // Load P
            float p[NUM_ELEMS_PER_THREAD];
            retrieve_mask_and_reduce_p<
                NUM_ELEMS_PER_THREAD,
                tmem_cols::P,
                NamedBarriers::wg0_warp02_sync,
                NamedBarriers::wg0_warp13_sync,
                false
            >(
                plan.is_k_valid[k%NUM_BUFS],
                warp_idx, lane_idx,
                [&]() {plan.bar_p_free[k%NUM_BUFS].arrive();},
                plan.p_exchange_buf,
                p
            );

            plan.bar_k_valid_free[k%NUM_BUFS].arrive();

            // Output P matrix (after mask) - write directly to global memory
            if (params.p_out != nullptr) {
                // Thread mapping: threads 0-63 and 64-127 form pairs handling the same row
                // Each pair writes 64 elements total (32 from each thread)
                int row = idx_in_warpgroup % 64;   // row ∈ [0, 63]，64 个线程一组处理 64 个 row
                int col_base = (idx_in_warpgroup < 64) ? 0 : 32;  // 前 64 线程写前 32 列，后 64 线程写后 32 列
		
                float* gP_base = reinterpret_cast<float*>(params.p_out) +
                                 s_q_idx * B_H * params.topk +
                                 row * params.topk + k * B_TOPK + col_base;

                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD / 4; ++i) {
                    float4* src = reinterpret_cast<float4*>(p + i * 4);
                    float4* dst = reinterpret_cast<float4*>(gP_base + i * 4);
                    *dst = *src;
                }
            }

            // Get rowwise max of Pi
            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
            cur_pi_max *= params.sm_scale_div_log2;

            plan.rowwise_bufs.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            cur_pi_max = max(cur_pi_max, plan.rowwise_bufs.rowwise_max_buf[idx_in_warpgroup^64]);
            real_mi = max(real_mi, cur_pi_max);
            bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);
            // By this point:
            // - cur_pi_max, real_mi, and mi is identical within each row (i.e. thread 0+64, 1+65, ...)
            // - should_scale_o is identical among every warp, and is identical among threads that controls the same row (i.e. among threads 0~31+64~95; and is identical among threads 32~63+96~127)


            // Calc scale factor, and scale li
            float new_max, scale_for_old;
            if (!should_scale_o) {
                // Don't scale O
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;   // mi is still identical within each row

            // Calculate S
            nv_bfloat162 s[NUM_ELEMS_PER_THREAD/2];
            float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(s, p, params.sm_scale_div_log2, new_max);
            li = fma(li, scale_for_old, cur_sum);

            // Wait for last SV gemm, write S
            if (k > 0) {
                plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; i += 1) {
                *(uint128_t*)(sS_base + B_H*8*i) = *(uint128_t*)(s + i*4);
            }

            // Scale O
            if (k > 0 && should_scale_o) {
                // plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);   // NOTE We have waited for last SV gemm before
                ku::tcgen05_after_thread_sync();
                rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);
                ku::tcgen05_before_thread_sync();
            }
            
            fence_view_async_shared();
            plan.bar_so_ready.arrive();
        }

        // Epilogue

        if (real_mi == -CUDART_INF_F) {
            // real_mi == -CUDART_INF_F <=> No valid TopK indices
            // We set li to 0 to fit the definition that li := exp(x[i] - mi)
            li = 0.0f;
            mi = -CUDART_INF_F;
        }
        
        // Exchange li
        plan.rowwise_bufs.rowwise_li_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        li += plan.rowwise_bufs.rowwise_li_buf[idx_in_warpgroup^64];

        // Store mi and li
        if (idx_in_warpgroup < 64) {
            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }

        // Wait for the last GEMM
        plan.bar_sv_done[(num_k_blocks-1)%NUM_BUFS].wait(((num_k_blocks-1)/NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        // Fetch dO if necessary

        // Store O
        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + (idx_in_warpgroup%64))*CUDART_L2E_F;
        float output_scale = __fdividef(1.0f, li + exp2f(attn_sink - mi));
        Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), SmemLayoutO{});
        constexpr int B_EPI = 64;
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        float2 o[B_EPI/2];
        bool have_valid_indices = __any_sync(0xffffffff, li != 0);  // Prevent some threads' li == 0 and some threads' li != 0 which lead to deadlock during ku::tmem_ld
        if (!have_valid_indices) {
            // If there are no valid indices, we set o[i] to 0 and don't load from TMEM
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/2; ++i)
                o[i].x = o[i].y = 0.0f;
            output_scale = 1.0f;
        }

        float2 output_scale_float2 = make_float2(output_scale, output_scale);

        bf16* sO_addrs[8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            sO_addrs[i] = &sO(idx_in_warpgroup%64, i*8);
        }

        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            // Each tile: 64 x 256
            CUTE_UNROLL
            for (int k = 0; k < (D_V/4)/B_EPI; ++k) {
                // Load O from tO
                if (have_valid_indices) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c*128 + k*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                }

                // Convert and store
                CUTE_UNROLL
                for (int i = 0; i < B_EPI/8; ++i) {
                    nv_bfloat162 o_bf16[4];
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                    }
                    *(uint128_t*)(sO_addrs[i] + (c*(D_V/2) + (idx_in_warpgroup/64)*(D_V/4) + k*B_EPI)*64) = *(uint128_t*)(o_bf16);
                }

                // Sync
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                
                if (warp_idx == 0 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
                if (warp_idx == 1 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + (D_V/B_EPI/4) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
            }
        }


        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // Producer warp for KV
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;
        if (elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                if (k == 2) {
                    plan.bar_prologue_utccp_nope.wait(0);   // Since q_nope coincidences with k[2]
                }

                // Copy NoPE
                int cur_buf = k%NUM_BUFS;
                plan.bar_sv_done[cur_buf].wait((k/NUM_BUFS)&1^1);
                bf16* sK_nope_base = plan.u.k.k_nope[cur_buf].data() + warp_idx*4*64;

                auto load_kv_nope_part = [&](int part_idx) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = part_idx*(D_V/2/64); local_col < (part_idx+1)*(D_V/2/64); ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_nope),
                                plan.bar_kv_nope_ready[cur_buf][part_idx],
                                sK_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                if (!should_skip_tma) {
                    load_kv_nope_part(0);
                    load_kv_nope_part(1);
                } else {
                    // NOTE See head128/phase1.cuh for this TMA skipping technique
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx)
                        plan.bar_kv_nope_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(bf16));
                }
            }
        }
    } else {
        // MMA warp
        if (warp_idx == 8 && elect_one_sync()) {
            // S -> T copy for Q
            UMMA::SmemDescriptor sQ_nope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.u.q_full.q_nope.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<64>>{}    // We use this shape for dual gemm (TODO Link)
                    )
                )
            );
            UMMA::SmemDescriptor sQ_rope_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.s_q_rope.q_rope.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW64_Atom<bf16>{},
                        Shape<Int<B_H*2>, Int<32>>{}
                    )
                )
            );
            
            if constexpr (HAVE_ROPE) {
                // Copy the RoPE tile: 128 rows * 32 cols (64B) (in UTCCP's view), or 64 rows * 64 cols (in our view)
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*(D_Q-D_V)*sizeof(bf16));
                plan.bar_prologue_q_rope.wait(0);
                ku::tcgen05_after_thread_sync();
                CUTE_UNROLL
                for (int subtile_idx = 0; subtile_idx < 2; ++subtile_idx) {
                    // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or 64 rows * 16 cols * 2 (in our view)
                    SM100_UTCCP_128dp256bit_1cta::copy(
                        sQ_rope_desc + (subtile_idx*32) / 16,
                        tmem_cols::Q_RoPE + subtile_idx*8
                    );
                }
                ku::umma_arrive_noelect(plan.bar_prologue_utccp_rope);
            }

            plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(bf16));
            plan.bar_prologue_q_nope.wait(0);
            ku::tcgen05_after_thread_sync();
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < D_V/64/2; ++tile_idx) {
                // A tile is 128 rows * 64 cols (128B) (in UTCCP's view), or 64 rows * 128 cols (in our view)
                CUTE_UNROLL
                for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                    // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or 64 rows * 16 cols * 2 (in our view)
                    SM100_UTCCP_128dp256bit_1cta::copy(
                        sQ_nope_desc + (tile_idx*(B_H*128*2) + subtile_idx*32) / 16,   // Remember that 4 LSBs are not included
                        tmem_cols::Q + tile_idx*32 + subtile_idx*8
                    );
                }
            }
            ku::umma_arrive_noelect(plan.bar_prologue_utccp_nope);

            if constexpr (HAVE_ROPE) {
                plan.bar_prologue_utccp_rope.wait(0);
            }

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                if (k < num_k_blocks) {
                    // Pi = QKi^T
                    int cur_buf = k%NUM_BUFS;
                    Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutKNoPE_TiledMMA{});
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE_TiledMMA{});

                    plan.bar_p_free[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();
                    
                    // Wait for K (RoPE)
                    // P = Q(rope) @ K(rope)^T
                    if constexpr (HAVE_ROPE) {
                        plan.bar_kv_rope_ready.wait(k&1);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ts(tiled_mma_P, tQ_rope, sK_rope, tP, true);
                        ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                    }

                    // Wait for K (NoPE)
                    if (k == 0) {
                        plan.bar_prologue_utccp_nope.wait(0);
                    }
                    Tensor sK_nope_divided = flat_divide(sK_nope, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
                    CUTE_UNROLL
                    for (int kv_nope_part_idx = 0; kv_nope_part_idx < 2; ++kv_nope_part_idx) {
                        plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(bf16));
                        plan.bar_kv_nope_ready[cur_buf][kv_nope_part_idx].wait((k/NUM_BUFS)&1);
                        ku::tcgen05_after_thread_sync();

                        // P += Q(nope) @ K(nope)^T
                        bool clear_accum = (!HAVE_ROPE) && kv_nope_part_idx == 0;
                        ku::utcmma_ts(tiled_mma_P, kv_nope_part_idx ? tQ_nope_part1 : tQ_nope_part0, sK_nope_divided(_, _, kv_nope_part_idx), tP, clear_accum);
                    }
                    ku::umma_arrive_noelect(plan.bar_qk_nope_done[cur_buf]);
                }
                if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s_q_rope.s), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutV{});

                    // Wait for S(i-1) and O to be scaled
                    plan.bar_so_ready.wait((k-1)&1);
                    ku::tcgen05_after_thread_sync();

                    // O += sS @ sV
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1);
                    ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                }
            }
        } else if (warp_idx == 9) {
            // KV valid loading warp
            if (lane_idx < B_TOPK/8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    char k_validness_mask = load_indices_and_generate_mask(
                        lane_idx,
                        gIndices + k*B_TOPK,
                        params.s_kv,
                        k*B_TOPK,
                        topk_length
                    );

                    int cur_buf = k%NUM_BUFS;
                    plan.bar_k_valid_free[cur_buf].wait((k/NUM_BUFS)&1^1);
                    plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                    plan.bar_k_valid_ready[cur_buf].arrive();
                }
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            if constexpr (HAVE_ROPE) {
                int thread_idx = threadIdx.x - 10*32;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 64/GROUP_SIZE, ROWS_PER_THREAD = B_TOPK/NUM_GROUPS;
                int group_idx = thread_idx / GROUP_SIZE, idx_in_group = thread_idx % GROUP_SIZE;
                Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE{});
                bf16* sK_rope_base = &sK_rope(group_idx, idx_in_group*8);
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int indices[ROWS_PER_THREAD];
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row)
                        indices[local_row] = __ldg(gIndices + k*B_TOPK + group_idx + local_row*NUM_GROUPS);
                    plan.bar_qk_rope_done.wait(k&1^1);
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row) {
                        int index = indices[local_row];
                        ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                            params.kv + (int64_t)index*params.stride_kv_s_kv + 512 + idx_in_group*8,
                            sK_rope_base + local_row*NUM_GROUPS*32,
                            index >= 0 && index < params.s_kv && index <= max_kv_i
                        );  // NOTE Using cp.async instead of TMA is faster here
                        // NOTE Here we only consider the range of `index` instead of also checking against topk_length, as it's noted that under this scenario (i.e. there exists a valid index among indices[topk_length: ] that points to a token who has NaN inside)
                    }
                    cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)&(plan.bar_kv_rope_ready));
                }
            }
        }
    }


#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<int D_QK>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.h_q == B_H);  // To save some calculation
    KU_ASSERT(params.d_qk == D_QK);
    static_assert(D_QK == 576 || D_QK == 512);

    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q_nope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = make_shape(params.h_q, D_Q-D_V, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q + D_V),
            make_layout(
                shape_Q_rope,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.d_v, _1{}, params.h_q*params.d_v)
            )
        ),
        SmemLayoutOTiles<1>{}
    );

    auto shape_P = make_shape(params.h_q, params.topk, params.s_q);
    auto tma_P = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((float*)params.p_out),
            make_layout(
                shape_P,
                make_stride(params.topk, _1{}, params.h_q*params.topk)
            )
        ),
        SmemLayoutP{}
    );


    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {D_V, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            params.kv,
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

    TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_O), decltype(tma_O),
        decltype(shape_P), decltype(tma_P)
    > tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_O, tma_O,
        shape_P, tma_P,
        tensor_map_kv_nope
    };
    auto kernel = &sparse_attn_fwd_kernel<D_QK == 576, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

}

