#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;
using namespace deep_gemm::sm100;


template <
    uint32_t kNumHeads,                                     // 64  heads of DSA indexer
    uint32_t kHeadDim,                                      // 128 dim   of DSA indexer
    bool kIsCompressedLogits,                               // false
    uint32_t BLOCK_Q,                                       // 1   dim per q_block
    uint32_t BLOCK_KV,                                      // 128 dim per kv_block
    uint32_t kNumQStages,                                   // 2 stages for q_block  barriers
    uint32_t kNumKVStages,                                  // 2 stages for kv_block barriers
    uint32_t kNumSpecializedThreads,                        // 128 threads for umma scheduler
    uint32_t kNumMathThreads,                               // 128 threads for grad epilogue
    uint32_t kNumSparseLoadThreads,                         // 128 threads for sparse kv/kv_scales/d_output loading, 1 thread for q/weights TMA loading
    uint32_t kNumMathWarpGroups = kNumMathThreads / 128>    // 1 warp-group for umma loop
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads + kNumSparseLoadThreads, 1)
void smxx_fp8_mqa_logits_bwd(
    // Parameters
    const uint32_t seq_len,
    const uint32_t seq_len_kv,
    const uint32_t max_seqlen_k,
    const uint32_t topk,
    const uint64_t stride_logits,
    // Inputs
    cutlass::float_e4m3_t* kv,
    float* kv_scales,
    float* grad_logits,
    uint32_t* cu_seq_len_k_start,
    uint32_t* cu_seq_len_k_end,
    uint32_t* topk_indices,
    // Results
    float* grad_q,
    float* grad_kv,
    float* grad_weights,
    // TMA descs
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights
)
{
    DG_STATIC_ASSERT(BLOCK_Q == 1, "Indexer bwd only support BLOCK_Q = 1");
    const auto& num_q_blocks = seq_len;

    // UMMA settings
    // Construct fp8 instruction with layout D, 16B_base
    DG_STATIC_ASSERT(BLOCK_KV == 128, "Indexer bwd only support BLOCK_KV = 128");
    constexpr uint32_t UMMA_M_FP8 = 128;
    constexpr uint32_t UMMA_K_FP8 = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N_FP8 = BLOCK_Q * kNumHeads;
    // Construct bf16 instruction with layout D, 32B_base
    DG_STATIC_ASSERT(kHeadDim == 128, "Indexer bwd only support kHeadDim = 128");
    constexpr uint32_t UMMA_M_BF16 = 128;
    constexpr uint32_t UMMA_K_BF16 = 32 / sizeof(cutlass::bfloat16_t);
    constexpr uint32_t UMMA_N_BF16_GRAD_Q  = BLOCK_Q * kNumHeads;
    constexpr uint32_t UMMA_N_BF16_GRAD_KV = BLOCK_KV;

    // Types
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using FP8_SWIZZLE_128B  = cute::Swizzle<3, 4, 3>;
    using BF16_SWIZZLE_128B = cute::Swizzle<2, 4, 2>;  // inner dim = 64 bf16 = 128B, required for BF16 UMMA
    using UMMA_LOGIT = cute::SM100_MMA_F8F6F4_SS;
    using UMMA_GRAD_Q = cute::SM100_MMA_F16BF16_SS<cutlass::bfloat16_t, cutlass::bfloat16_t, float, UMMA_M_BF16, UMMA_N_BF16_GRAD_Q, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using UMMA_GRAD_KV = cute::SM100_MMA_F16BF16_SS<cutlass::bfloat16_t, cutlass::bfloat16_t, float, UMMA_M_BF16, UMMA_N_BF16_GRAD_KV, cute::UMMA::Major::K, cute::UMMA::Major::K>;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = cutlass::canonical_warp_idx_sync();
    const auto& warp_in_group_idx = warp_idx % 4;
    const auto& warpgroup_idx = cutlass::canonical_warp_group_idx();
    const auto& lane_idx = get_lane_idx();

    // Block   | kNumMathThreads | kNumSpecializedThreads | kNumSparseLoadThreads |
    // Threads |     0 ~ 127     |        128 ~ 255       |       256 ~ 383       |
    //         |  grad epilogue  |  umma & tma scheduler  |      sparse load      |

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads == 128 and kNumSparseLoadThreads == 128, "Invalid threads");
    if ( warp_idx == (kNumMathThreads + kNumSpecializedThreads) / 32 and cute::elect_one_sync() )
    {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // 1. q            [BLOCK_Q, kNumHeads, kHeadDim]
    // 2. kv           [BLOCK_KV, kHeadDim]
    // 3. weights      [BLOCK_Q, kNumHeads]
    // 4. kv_scale     [BLOCK_KV]
    // 5. topk_indices [BLOCK_Q, BLOCK_KV]
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE            = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    // smem_q_bf16 and smem_reshape_d_logit_bf16 inner dim (K for UMMA) must be >= 64 bf16 (128B)
    // for SWIZZLE_128B_BASE32B. Pad to 64 when kNumHeads < 64.
    static constexpr uint32_t kNumHeadsPadded                  = kNumHeads < 64 ? 64 : kNumHeads;
    static constexpr uint32_t SMEM_Q_BF16_SIZE_PER_STAGE       = BLOCK_Q * kNumHeadsPadded * kHeadDim * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE           = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_BF16_SIZE_PER_STAGE      = BLOCK_KV * kHeadDim * sizeof(cutlass::bfloat16_t);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE       = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE     = BLOCK_KV * sizeof(float);
    static constexpr uint32_t SMEM_TOPK_INDICES_SIZE_PER_STAGE = BLOCK_Q * BLOCK_KV * sizeof(uint32_t);
    static constexpr uint32_t SMEM_D_LOGIT_SIZE_PER_STAGE      = BLOCK_Q * kNumHeadsPadded * BLOCK_KV * sizeof(cutlass::bfloat16_t);


    // TODO: check if 1024 bytes for swizzle-128B???
    // Align to 512 bytes for swizzle-64B
    extern __shared__ __align__(512) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    // Tmem configs
    constexpr uint32_t kNumTmemColsLogits = BLOCK_Q * kNumHeads; // recompute `logits = kv @ q^T` :     [BLOCK_KV, BLOCK_Q * kNumHeads], M=128, N=64, layoutD
    constexpr uint32_t kNumTmemColsGradQ  = BLOCK_Q * kNumHeads; // `d_q   = kv @ d_logits^T`:          [kHeadDim, BLOCK_Q * kNumHeads], M=128, N=64, layoutD
    constexpr uint32_t kNumTmemColsGradKV = BLOCK_KV;            // `d_kv  = reshape_d_logits @ q^T` :  [kHeadDim, BLOCK_KV], M=128, N=128, layoutD
    // UMMA requires TMEM column offset aligned to N; d_kv uses N=BLOCK_KV=128, so round up.
    // kNumHeads=64: (64+64+128-1)/128*128=128; kNumHeads=32: (32+32+128-1)/128*128=128
    constexpr uint32_t kTmemStartGradKV = ((kNumTmemColsLogits + kNumTmemColsGradQ + kNumTmemColsGradKV - 1) / kNumTmemColsGradKV) * kNumTmemColsGradKV;
    constexpr uint32_t kNumTmemCols = kTmemStartGradKV + kNumTmemColsGradKV; // kNumMathWarpGroups = 1
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer +
            kNumQStages * SMEM_Q_SIZE_PER_STAGE + 
            SMEM_KV_SIZE_PER_STAGE * i);
    });

    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            kNumQStages  * SMEM_Q_SIZE_PER_STAGE  +
            kNumKVStages * SMEM_KV_SIZE_PER_STAGE +
            SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            kNumQStages  * (SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE) +
            kNumKVStages * SMEM_KV_SIZE_PER_STAGE +
            SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto smem_topk_indices = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer +
            kNumQStages  * (SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE) +
            kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE) +
            SMEM_TOPK_INDICES_SIZE_PER_STAGE * i);
    });

    // smem for bf16 gemm
    auto smem_d_logit_bf16 = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer +
            kNumQStages  * (SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE) +
            kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE + SMEM_TOPK_INDICES_SIZE_PER_STAGE) +
            SMEM_D_LOGIT_SIZE_PER_STAGE * i);
    });
    auto smem_reshape_d_logit_bf16 = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer +
            kNumQStages  * (SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE) +
            kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE + SMEM_TOPK_INDICES_SIZE_PER_STAGE + SMEM_D_LOGIT_SIZE_PER_STAGE) +
            SMEM_D_LOGIT_SIZE_PER_STAGE * i);
    });
    auto smem_q_bf16 = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer +
            kNumQStages  * (SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE) +
            kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE + SMEM_TOPK_INDICES_SIZE_PER_STAGE + SMEM_D_LOGIT_SIZE_PER_STAGE * 2) +
            SMEM_Q_BF16_SIZE_PER_STAGE * i);
    });
    auto smem_kv_bf16 = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer +
            kNumQStages  * (SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE + SMEM_Q_BF16_SIZE_PER_STAGE) +
            kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE + SMEM_TOPK_INDICES_SIZE_PER_STAGE + SMEM_D_LOGIT_SIZE_PER_STAGE * 2) +
            SMEM_KV_BF16_SIZE_PER_STAGE * i);
    });

    // Barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_bf16[kNumKVStages]);
    auto full_q_barriers          = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers         = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers         = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers        = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_fp8_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_fp8_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i); });
    auto full_umma_bf16_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2 + i); });
    auto empty_umma_bf16_barriers = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2 + kNumKVStages + i); });

    // Tmem ptr
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2 + kNumKVStages * 2);

    // Initialize barriers
    DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");
    const bool& is_tma_load_warp  = (warp_idx == (kNumMathThreads / 32));
    const bool& is_umma_fp8_warp  = (warp_idx == (kNumMathThreads / 32 + 1));
    const bool& is_umma_bf16_warp = (warp_idx == (kNumMathThreads / 32 + 2));
    if ( is_tma_load_warp and cute::elect_one_sync() )
    {
        #pragma unroll
        for ( uint32_t i = 0; i < kNumQStages; ++ i )
        {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + kNumSparseLoadThreads);
        }
        #pragma unroll
        for ( uint32_t i = 0; i < kNumKVStages; ++ i )
        {
            full_kv_barriers[i]->init(kNumSparseLoadThreads);
            empty_kv_barriers[i]->init(kNumMathThreads + kNumSparseLoadThreads);
        }
        #pragma unroll
        for ( uint32_t i = 0; i < kNumMathWarpGroups; ++ i )
        {
            full_umma_fp8_barriers[i]->init(1);
            empty_umma_fp8_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for ( uint32_t i = 0; i < kNumKVStages; ++ i )
        {
            full_umma_bf16_barriers[i]->init(1);
            empty_umma_bf16_barriers[i]->init(kNumMathThreads + kNumSparseLoadThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    else if ( is_umma_fp8_warp )
    {
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }

    // Initialize smem
    for ( uint32_t q_stage = 0; q_stage < kNumQStages; q_stage++ )
    {
        for ( uint32_t i = threadIdx.x; i < BLOCK_Q * kNumHeads * kHeadDim; i += blockDim.x )
        {
            smem_q[q_stage][i] = (cutlass::float_e4m3_t)0;
            smem_q_bf16[q_stage][i] = (cutlass::bfloat16_t)0;
        }
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 104;
    constexpr uint32_t kSparseLoadRegisters = 192;
    constexpr uint32_t kNumMathRegisters = 208;

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t>
    {
        return {block_q_idx + gridDim.x, q_iter_idx + 1};
    };
    const auto& load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
    {
        const auto& q_idx = min(block_q_idx, seq_len - 1);
        uint32_t seq_k_start = __ldg(cu_seq_len_k_start + q_idx);
        uint32_t start = min(cute::numeric_limits<uint32_t>::max(), min(seq_k_start, seq_len_kv));
        uint32_t end = max(cute::numeric_limits<uint32_t>::min(), min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv));

        // start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start,
                min(ceil_div(end - start, BLOCK_KV), ceil_div(topk, BLOCK_KV))};
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto& get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t>
    {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    if ( threadIdx.x >= (kNumMathThreads + kNumSpecializedThreads) )
    {
        cutlass::arch::warpgroup_reg_alloc<kSparseLoadRegisters>();

        constexpr auto q_bf16_layout = cute::make_layout(
            cute::make_shape(cute::Int<kNumHeadsPadded>{}, cute::Int<kHeadDim>{}),
            cute::make_stride(cute::Int<1>{}, cute::Int<kNumHeadsPadded>{})
        );
        // inner dim = kNumHeadsPadded (>=64) bf16 = 128B → always BF16_SWIZZLE_128B
        constexpr auto swizzled_q_bf16_layout = cute::composition(q_bf16_layout, BF16_SWIZZLE_128B{});

        constexpr auto q_fp8_layout = cute::make_layout(
            cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kNumHeads>{}),
            cute::make_stride(cute::Int<1>{}, cute::Int<kHeadDim>{})
        );
        constexpr auto swizzled_q_fp8_layout = cute::composition(q_fp8_layout, FP8_SWIZZLE_128B{});

        constexpr auto kv_bf16_layout = cute::make_layout(
            cute::make_shape(cute::Int<BLOCK_KV/2>{}, cute::Int<kHeadDim>{}),
            cute::make_stride(cute::Int<1>{}, cute::Int<BLOCK_KV/2>{})
        );
        constexpr auto swizzled_kv_bf16_layout = cute::composition(kv_bf16_layout, BF16_SWIZZLE_128B{});

        constexpr auto smem_kv_layout = cute::make_layout(
            cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<BLOCK_KV>{}), 
            cute::make_stride(cute::Int<1>{}, cute::Int<kHeadDim>{})
        );
        constexpr auto swizzled_smem_kv_layout = cute::composition(smem_kv_layout, FP8_SWIZZLE_128B{});

        while ( block_q_idx < num_q_blocks )
        {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            uint32_t ke = __ldg(cu_seq_len_k_end + block_q_idx);

            // cast q fp8 -> bf16
            auto q_bf16_tensor = cute::make_tensor(cute::make_smem_ptr(smem_q_bf16[q_stage_idx]), swizzled_q_bf16_layout);
            auto q_fp8_tensor = cute::make_tensor(cute::make_smem_ptr(smem_q[q_stage_idx]), swizzled_q_fp8_layout);
            #pragma unroll
            for ( uint32_t head_idx = 0; head_idx < kNumHeads; head_idx++ )
            {
                q_bf16_tensor(head_idx, threadIdx.x % 128) = cutlass::bfloat16_t(static_cast<float>(q_fp8_tensor(threadIdx.x % 128, head_idx)));
            }

            // kv_idx is constant per thread across all kv_blocks.
            const uint32_t kv_idx = threadIdx.x % 128;

            for ( uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx )
            {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);

                // Load topk-index
                uint32_t topk_index = *reinterpret_cast<volatile uint32_t*>(
                    topk_indices + block_q_idx * topk + kv_block_idx * BLOCK_KV + kv_idx);

                // Wait KV consumer release
                empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                uint32_t global_kv_idx = topk_index < ke ? topk_index : ke - 1;

                // Load sparse KV via cp.async
                #pragma unroll 1
                for ( int i = 0; i < 8; i += 4 )
                {
                    uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv[kv_stage_idx] + swizzled_smem_kv_layout((i+0) * 16, kv_idx)));
                    uint32_t s1 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv[kv_stage_idx] + swizzled_smem_kv_layout((i+1) * 16, kv_idx)));
                    uint32_t s2 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv[kv_stage_idx] + swizzled_smem_kv_layout((i+2) * 16, kv_idx)));
                    uint32_t s3 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv[kv_stage_idx] + swizzled_smem_kv_layout((i+3) * 16, kv_idx)));
                    const void* g0 = kv + global_kv_idx * kHeadDim + (i+0) * 16;
                    const void* g1 = kv + global_kv_idx * kHeadDim + (i+1) * 16;
                    const void* g2 = kv + global_kv_idx * kHeadDim + (i+2) * 16;
                    const void* g3 = kv + global_kv_idx * kHeadDim + (i+3) * 16;
                    asm volatile(
                        "cp.async.cg.shared.global [%0], [%1], 16;\n"
                        "cp.async.cg.shared.global [%2], [%3], 16;\n"
                        "cp.async.cg.shared.global [%4], [%5], 16;\n"
                        "cp.async.cg.shared.global [%6], [%7], 16;\n"
                        : : "r"(s0), "l"(g0), "r"(s1), "l"(g1), "r"(s2), "l"(g2), "r"(s3), "l"(g3)
                    );
                }

                // Issue kv_scale load before the fence so its ~200-cycle L2 latency
                // overlaps with the remaining cp.async wait time.
                float kv_scale_val = *reinterpret_cast<volatile float*>(kv_scales + global_kv_idx);

                cute::cp_async_fence();
                asm volatile("cp.async.wait_group %0;" :: "n"(0));
                cutlass::arch::fence_view_async_shared();

                smem_kv_scales[kv_stage_idx][kv_idx]    = kv_scale_val;
                smem_topk_indices[kv_stage_idx][kv_idx] = topk_index;

                full_kv_barriers[kv_stage_idx]->arrive();
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                asm volatile("bar.sync %0, %1;" : : "r"(warpgroup_idx), "r"(128) : "memory");

                // cast kv fp8 -> bf16
                auto smem_kv_bf16_ptr = kv_idx < BLOCK_KV/2 ? smem_kv_bf16[kv_stage_idx] : smem_kv_bf16[kv_stage_idx] + kHeadDim * BLOCK_KV/2;
                auto kv_bf16_tensor = cute::make_tensor(cute::make_smem_ptr(smem_kv_bf16_ptr), swizzled_kv_bf16_layout);
                auto kv_fp8_tensor  = cute::make_tensor(cute::make_smem_ptr(smem_kv[kv_stage_idx]), swizzled_smem_kv_layout);
                auto tensor_kv_idx  = kv_idx < BLOCK_KV/2 ? kv_idx : kv_idx - BLOCK_KV/2;
                #pragma unroll
                for ( uint32_t dim_idx = 0; dim_idx < kHeadDim; dim_idx++ )
                {
                    kv_bf16_tensor(tensor_kv_idx, dim_idx) = cutlass::bfloat16_t(static_cast<float>(kv_fp8_tensor(dim_idx, kv_idx)));
                }

                asm volatile("bar.sync %0, %1;" : : "r"(warpgroup_idx), "r"(128) : "memory");
                tcgen05_before_thread_sync();
                empty_kv_barriers[kv_stage_idx]->arrive();
                empty_umma_bf16_barriers[kv_stage_idx]->arrive();
            }
            CUTE_TIE_DECL(get_kv_pipeline(num_kv_blocks - 1), last_kv_stage_idx, last_kv_phase);
            full_umma_bf16_barriers[last_kv_stage_idx]->wait(last_kv_phase);
            tcgen05_after_thread_sync();

            const auto& tmem_start_d_q = __shfl_sync(0xffffffff, kNumTmemColsLogits, 0);
            // d_q: Part 0
            {
                // uint32_t shifted_d_q[32] = {};
                uint32_t shifted_d_q[32];
                #pragma unroll
                for ( uint32_t i = 0; i < 32; i++ )
                {
                    shifted_d_q[i] = 0;
                }
                float* d_q = reinterpret_cast<float*>(shifted_d_q);
                auto tmem_load_d_q = [&](auto... Is) { cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start_d_q, shifted_d_q[Is]...); };
                [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load_d_q(Is...); }(cute::make_index_sequence<32>{});
                cutlass::arch::fence_view_async_tmem_load();

                #pragma unroll
                for ( uint32_t head_idx = 0; head_idx < 32; head_idx++ )
                {
                    uint32_t grad_q_offset = block_q_idx * kNumHeads * kHeadDim + head_idx * kHeadDim;
                    grad_q[grad_q_offset + (threadIdx.x % 128)] = d_q[head_idx];
                }
            }
            // d_q: Part 1 (head 32~63; only exists for kNumHeads > 32)
            if constexpr (kNumHeads > 32) {
                // uint32_t shifted_d_q[32] = {};
                uint32_t shifted_d_q[32];
                #pragma unroll
                for ( uint32_t i = 0; i < 32; i++ )
                {
                    shifted_d_q[i] = 0;
                }
                float* d_q = reinterpret_cast<float*>(shifted_d_q);
                auto tmem_load_d_q = [&](auto... Is) { cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start_d_q + 32, shifted_d_q[Is]...); };
                [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load_d_q(Is...); }(cute::make_index_sequence<32>{});
                cutlass::arch::fence_view_async_tmem_load();

                #pragma unroll
                for ( uint32_t head_idx = 0; head_idx < 32; head_idx++ )
                {
                    uint32_t grad_q_offset = block_q_idx * kNumHeads * kHeadDim + (head_idx + 32) * kHeadDim;
                    grad_q[grad_q_offset + (threadIdx.x % 128)] = d_q[head_idx];
                }
            }

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            num_total_kv_blocks += num_kv_blocks;

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
    else if ( is_tma_load_warp )
    {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        if ( cute::elect_one_sync() )
        {
            const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx)
            {
                tma_copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
                tma_copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            };

            while ( block_q_idx < num_q_blocks )
            {
                CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, block_q_idx);

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    }
    else if ( is_umma_fp8_warp )
    {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // Make FP8 UMMA desc for recompute `logits = k @ q^T`
        // [BLOCK_KV, BLOCK_Q * kNumHeads] = [BLOCK_KV, kHeadDim] @ [BLOCK_Q * kNumHeads, kHeadDim]^T
        auto instr_desc_logits = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                             UMMA_M_FP8, UMMA_N_FP8, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto umma_desc_logits = cute::UMMA::make_runtime_instr_desc(instr_desc_logits);

        while ( block_q_idx < num_q_blocks )
        {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Compute over KV blocks
            #pragma unroll
            for ( uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++ )
            {
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Issue FP8 UMMA for logits
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % UMMA_K_FP8 == 0, "Invalid head dim");
                DG_STATIC_ASSERT(kNumMathWarpGroups == 1, "Invalid math warp-group");
                empty_umma_fp8_barriers[0]->wait(((num_total_kv_blocks + kv_block_idx) & 1) ^ 1);
                tcgen05_after_thread_sync();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / UMMA_K_FP8; ++ k) {
                    auto a_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(smem_kv[kv_stage_idx], 0, k * UMMA_K_FP8);
                    auto b_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(smem_q[q_stage_idx],   0, k * UMMA_K_FP8);
                    UMMA_LOGIT::fma(a_desc, b_desc, 0, k, umma_desc_logits);
                }

                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_fp8_barriers[0]));
            }
            num_total_kv_blocks += num_kv_blocks;

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
    else if ( is_umma_bf16_warp )
    {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // Make BF16 UMMA desc for `d_q^T = k^T @ d_logits`: 
        // [kHeadDim, BLOCK_Q * kNumHeads] = [kHeadDim, BLOCK_KV/2] @ [BLOCK_Q * kNumHeads, BLOCK_KV/2]^T, loop 2
        auto instr_desc_d_q = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                          UMMA_M_BF16, UMMA_N_BF16_GRAD_Q, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto umma_desc_d_q = cute::UMMA::make_runtime_instr_desc(instr_desc_d_q);

        // Make BF16 UMMA desc for `d_kv = q @ d_logits^T` of better bandwidth
        // [kHeadDim, BLOCK_KV] = [kHeadDim, BLOCK_Q * kNumHeads] @ [BLOCK_KV, BLOCK_Q * kNumHeads]^T
        // K = BLOCK_Q * kNumHeads BF16 (128B for kNumHeads=64, 64B for kNumHeads=32)
        auto instr_desc_d_kv = cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                           UMMA_M_BF16, UMMA_N_BF16_GRAD_KV, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto umma_desc_d_kv = cute::UMMA::make_runtime_instr_desc(instr_desc_d_kv);

        while ( block_q_idx < num_q_blocks )
        {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Compute over KV blocks
            #pragma unroll
            for ( uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++ )
            {
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                empty_umma_bf16_barriers[kv_stage_idx]->wait(kv_phase);
                tcgen05_after_thread_sync();

                // Issue BF16 UMMA for d_q
                #pragma unroll 1
                for ( uint32_t k = 0; k < (BLOCK_KV / 2) / UMMA_K_BF16; ++k )
                {
                    auto a_desc = make_umma_desc<cute::UMMA::Major::K, UMMA_M_BF16,        BLOCK_KV/2, 128, true>(smem_kv_bf16[kv_stage_idx],      0, k * UMMA_K_BF16);
                    auto b_desc = make_umma_desc<cute::UMMA::Major::K, UMMA_N_BF16_GRAD_Q, BLOCK_KV/2, 128, true>(smem_d_logit_bf16[kv_stage_idx], 0, k * UMMA_K_BF16);
                    UMMA_GRAD_Q::fma(a_desc, b_desc, kNumTmemColsLogits, k + kv_block_idx, umma_desc_d_q);
                }
                #pragma unroll 1
                for ( uint32_t k = 0; k < (BLOCK_KV / 2) / UMMA_K_BF16; ++k )
                {
                    auto a_desc = make_umma_desc<cute::UMMA::Major::K, UMMA_M_BF16,        BLOCK_KV/2, 128, true>(smem_kv_bf16[kv_stage_idx] + kHeadDim * BLOCK_KV/2,        0, k * UMMA_K_BF16);
                    auto b_desc = make_umma_desc<cute::UMMA::Major::K, UMMA_N_BF16_GRAD_Q, BLOCK_KV/2, 128, true>(smem_d_logit_bf16[kv_stage_idx] + kNumHeads * BLOCK_KV/ 2, 0, k * UMMA_K_BF16);
                    UMMA_GRAD_Q::fma(a_desc, b_desc, kNumTmemColsLogits, 1, umma_desc_d_q);
                }

                // Issue BF16 UMMA for d_kv
                // smem_q_bf16 and smem_reshape_d_logit_bf16 inner dim is kNumHeadsPadded (>=64) = 128B → SWIZZLE_128B_BASE32B (kUseBase32=true)
                #pragma unroll 1
                for ( uint32_t k = 0; k < (BLOCK_Q * kNumHeadsPadded) / UMMA_K_BF16; ++k )
                {
                    // leading dim = BLOCK_Q * kNumHeadsPadded * sizeof(bf16) = 128B for both kNumHeads=32 and 64
                    auto a_desc = make_umma_desc<cute::UMMA::Major::K, UMMA_M_BF16,         BLOCK_Q * kNumHeadsPadded, BLOCK_Q * kNumHeadsPadded * 2, true>(smem_q_bf16[q_stage_idx],                0, k * UMMA_K_BF16);
                    auto b_desc = make_umma_desc<cute::UMMA::Major::K, UMMA_N_BF16_GRAD_KV, BLOCK_Q * kNumHeadsPadded, BLOCK_Q * kNumHeadsPadded * 2, true>(smem_reshape_d_logit_bf16[kv_stage_idx], 0, k * UMMA_K_BF16);
                    UMMA_GRAD_KV::fma(a_desc, b_desc, kTmemStartGradKV, k, umma_desc_d_kv);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_bf16_barriers[kv_stage_idx]));
            }
            num_total_kv_blocks += num_kv_blocks;

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
    else if ( threadIdx.x >=kNumMathThreads and threadIdx.x < (kNumMathThreads + kNumSpecializedThreads) )
    {
        // do nothing
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    }
    else if ( threadIdx.x < kNumMathThreads )
    {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // Offsets
        const auto& tmem_start_logits = __shfl_sync(0xffffffff, 0, 0);
        const auto& tmem_start_d_q    = __shfl_sync(0xffffffff, kNumTmemColsLogits, 0);
        const auto& tmem_start_d_kv   = __shfl_sync(0xffffffff, kTmemStartGradKV, 0);
        const auto& warp_offset = warp_idx * 32;

        const auto& m_v = threadIdx.x % 128;
        const auto& m_dim = threadIdx.x % 128;
        const auto tensor_m_v = m_v < BLOCK_KV/2 ? m_v : m_v - BLOCK_KV/2;

        // Smem tensor layout
        constexpr auto d_logit_bf16_layout = cute::make_layout(
            cute::make_shape(cute::Int<BLOCK_KV/2>{}, cute::Int<kNumHeads>{}),
            cute::make_stride(cute::Int<1>{}, cute::Int<BLOCK_KV/2>{})
        );
        constexpr auto swizzled_d_logit_bf16_layout = cute::composition(d_logit_bf16_layout, BF16_SWIZZLE_128B{});

        constexpr auto reshape_d_logit_bf16_layout = cute::make_layout(
            cute::make_shape(cute::Int<kNumHeadsPadded>{}, cute::Int<BLOCK_KV>{}),
            cute::make_stride(cute::Int<1>{}, cute::Int<kNumHeadsPadded>{})
        );
        // inner dim = kNumHeadsPadded (>=64) bf16 = 128B → always BF16_SWIZZLE_128B
        constexpr auto swizzled_reshape_d_logit_bf16_layout = cute::composition(reshape_d_logit_bf16_layout, BF16_SWIZZLE_128B{});

        while ( block_q_idx < num_q_blocks )
        {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Accumulator for d_weights over kv_block
            float d_weights_accum[kNumHeads] = {};

            // k_end for block_q_idx
            uint32_t ke = __ldg(cu_seq_len_k_end + block_q_idx);

            // Compute over KV blocks
            for ( uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx )
            {
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv = ld_shared(smem_kv_scales[kv_stage_idx] + m_v);

                uint32_t d_logits_offset = block_q_idx * topk + kv_block_idx * BLOCK_KV + m_v;
                prefetch_l1(grad_logits + d_logits_offset);

                // Wait UMMA arrival
                full_umma_fp8_barriers[warpgroup_idx]->wait((num_total_kv_blocks + kv_block_idx) & 1);
                tcgen05_after_thread_sync();

                // Load d_logits
                float d_logit = smem_topk_indices[kv_stage_idx][m_v] < ke ? __ldg(grad_logits + d_logits_offset) : 0.f;

                // d_weights[h] += ReLU(S[h, k]) * d_logits[k] * scale_kv[k]
                float d_logit_sum = d_logit * scale_kv;

                uint32_t shifted_accum_logits[kNumHeads];
                float* accum_logits = reinterpret_cast<float*>(shifted_accum_logits);
                // kNumHeads=64: kNumTmemColsLogits=64 → use 64x; kNumHeads=32: use 32x
                if constexpr (kNumHeads == 64) {
                    auto tmem_load_logits = [&](auto... Is) { cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start_logits, shifted_accum_logits[Is]...); };
                    [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load_logits(Is...); }(cute::make_index_sequence<kNumHeads>{});
                } else {
                    auto tmem_load_logits = [&](auto... Is) { cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start_logits, shifted_accum_logits[Is]...); };
                    [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load_logits(Is...); }(cute::make_index_sequence<kNumHeads>{});
                }
                cutlass::arch::fence_view_async_tmem_load();

                tcgen05_before_thread_sync();
                empty_umma_fp8_barriers[warpgroup_idx]->arrive();

                // d_weight
                #pragma unroll
                for ( uint32_t head_idx = 0; head_idx < kNumHeads; head_idx++ )
                {
                    d_weights_accum[head_idx] += fmaxf(accum_logits[head_idx], 0) * d_logit_sum;
                }

                // d_logit_bf16
                auto smem_d_logit_bf16_ptr = m_v < BLOCK_KV/2 ? smem_d_logit_bf16[kv_stage_idx] : smem_d_logit_bf16[kv_stage_idx] + kNumHeads * BLOCK_KV/ 2;
                auto d_logit_bf16_tensor = cute::make_tensor(cute::make_smem_ptr(smem_d_logit_bf16_ptr), swizzled_d_logit_bf16_layout);

                // reshape d_logit_bf16
                auto reshape_d_logit_bf16_tensor = cute::make_tensor(cute::make_smem_ptr(smem_reshape_d_logit_bf16[kv_stage_idx]), swizzled_reshape_d_logit_bf16_layout);
                #pragma unroll
                for ( uint32_t head_idx = 0; head_idx < kNumHeads; head_idx++ )
                {
                    auto d_logit_bf16 = accum_logits[head_idx] > 0 ? cutlass::bfloat16_t(d_logit_sum * ld_shared(smem_weights[q_stage_idx] + head_idx)) : cutlass::bfloat16_t(0.f); // TODO: preload weight
                    d_logit_bf16_tensor(tensor_m_v, head_idx) = d_logit_bf16;
                    reshape_d_logit_bf16_tensor(head_idx, m_v) = d_logit_bf16;
                }

                tcgen05_before_thread_sync();
                empty_umma_bf16_barriers[kv_stage_idx]->arrive();
                full_umma_bf16_barriers[kv_stage_idx]->wait(kv_phase);
                tcgen05_after_thread_sync();

                // store d_kv
                uint32_t shifted_d_kv[128];
                float* d_kv = reinterpret_cast<float*>(shifted_d_kv);
                auto tmem_load_d_kv = [&](auto... Is) { cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start_d_kv, shifted_d_kv[Is]...); };
                [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load_d_kv(Is...); }(cute::make_index_sequence<128>{});
                cutlass::arch::fence_view_async_tmem_load();

                #pragma unroll
                for ( uint32_t kv_idx = 0; kv_idx < 128; kv_idx++ )
                {
                    if ( smem_topk_indices[kv_stage_idx][kv_idx] < ke )
                        atomicAdd(grad_kv + smem_topk_indices[kv_stage_idx][kv_idx] * kHeadDim + m_dim, d_kv[kv_idx]);
                }

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;

            #pragma unroll
            for ( uint32_t head_idx = 0; head_idx < kNumHeads; head_idx++ )
            {
                #pragma unroll
                for ( uint32_t i = 0; i < 5; i++ )
                {
                    const auto& offset = static_cast<int>(1u << i);
                    d_weights_accum[head_idx] += __shfl_xor_sync(0xffffffffu, d_weights_accum[head_idx], offset);
                }
                __syncwarp();
                if ( cute::elect_one_sync() )
                {
                    atomicAdd(grad_weights + block_q_idx * kNumHeads + head_idx, d_weights_accum[head_idx]);
                }
                __syncwarp();
            }

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }

    // Free tensor memory
    __syncthreads();
    if (is_umma_fp8_warp)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
}

} // namespace deep_gemm
