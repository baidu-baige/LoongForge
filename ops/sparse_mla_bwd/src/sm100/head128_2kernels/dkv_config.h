#pragma once

#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"

namespace sm100::bwd::head128_2kernels::dkv {

using namespace cute;

template<
    typename Shape_QNoPE, typename TMA_QNoPE,
    typename Shape_QRoPE, typename TMA_QRoPE,
    typename Shape_dO, typename TMA_dO,
    typename Shape_S, typename TMA_S,
    typename Shape_dS, typename TMA_dS
>
struct TmaParams {
    Shape_QNoPE shape_Q_nope;
    TMA_QNoPE tma_Q_nope;
    Shape_QRoPE shape_Q_rope;
    TMA_QRoPE tma_Q_rope;
    Shape_dO shape_dO;
    TMA_dO tma_dO;
    Shape_S shape_S;
    TMA_S tma_S;
    Shape_dS shape_dS;
    TMA_dS tma_dS;
};

static constexpr int D_QK = 576;
static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr int D_ROPE = D_Q - D_V;
static constexpr int B_H = 128;
static constexpr int TOPK_GRANULARITY = 128;
static constexpr int DKV_TILE_M = TOPK_GRANULARITY;
static constexpr int DKV_ROWS_PER_CTA = DKV_TILE_M / 2;
static constexpr int NOPE_COLS_PER_CTA = 256;
static constexpr int ROPE_COLS_PER_CTA = D_ROPE / 2;
static constexpr int NUM_S_DS_BUFS = 2;
static constexpr int NUM_THREADS = 12 * 32;

static_assert(DKV_TILE_M == B_H, "dKV paired tile expects 128-row MMA tiles.");
static_assert(DKV_ROWS_PER_CTA == 64, "Each CTA in the dKV kernel owns 64 rows.");
static_assert(TOPK_GRANULARITY == 2 * DKV_ROWS_PER_CTA, "The paired dKV tile must be split evenly across two CTAs.");
static_assert(NOPE_COLS_PER_CTA * 2 == D_V, "NoPE staging must cover the full 512-dim latent width across the cluster.");
static_assert(ROPE_COLS_PER_CTA * 2 == D_ROPE, "RoPE staging must cover the full rope width across the cluster.");
static_assert(NUM_S_DS_BUFS == 2, "dKV kernel currently expects ping-pong shared-memory buffering for S/dS tiles.");

using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<NOPE_COLS_PER_CTA>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdO = SmemLayoutQNoPE;

using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<ROPE_COLS_PER_CTA>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQNoPE_MMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<NOPE_COLS_PER_CTA>, Int<B_H>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutdO_MMA = SmemLayoutQNoPE_MMA;

using SmemLayoutQRoPE_MMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW64_Atom<bf16>{},
    Shape<Int<ROPE_COLS_PER_CTA>, Int<B_H>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<DKV_ROWS_PER_CTA>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdS = SmemLayoutS;

using SmemLayoutS_MMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_INTER_Atom<bf16>{},
    Shape<Int<DKV_ROWS_PER_CTA>, Int<B_H>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutdS_MMA = SmemLayoutS_MMA;

static_assert(cosize_v<SmemLayoutQNoPE> == cosize_v<SmemLayoutQNoPE_MMA>);
static_assert(cosize_v<SmemLayoutQRoPE> == cosize_v<SmemLayoutQRoPE_MMA>);
static_assert(cosize_v<SmemLayoutS> == cosize_v<SmemLayoutS_MMA>);

using TiledMMA_dKV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, DKV_TILE_M, NOPE_COLS_PER_CTA, UMMA::Major::MN, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}
));

using TiledMMA_dKV_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, DKV_TILE_M, D_ROPE, UMMA::Major::MN, UMMA::Major::MN>{}
));

struct tmem_cols {
    static constexpr int dKV = 0;
    static constexpr int dKV_RoPE = 256;
    static constexpr int kNumUsedCols = dKV_RoPE + D_ROPE / 2;
};

static_assert(tmem_cols::kNumUsedCols == 288, "dKV kernel uses 288 logical TMEM columns.");

struct alignas(128) SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
    array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutS>> s[NUM_S_DS_BUFS];
        array_aligned<bf16, cosize_v<SmemLayoutdS>> ds[NUM_S_DS_BUFS];
    } s_ds;

    transac_bar_t bar_q_nope_ready;
    transac_bar_t bar_q_rope_ready;
    transac_bar_t bar_dO_ready;
    transac_bar_t bar_s_ready[NUM_S_DS_BUFS];
    transac_bar_t bar_ds_ready[NUM_S_DS_BUFS];
    transac_bar_t bar_dkv_nope_ready[NUM_S_DS_BUFS];
    transac_bar_t bar_dkv_rope_ready[NUM_S_DS_BUFS];
    transac_bar_t bar_dkv_nope_done;
    transac_bar_t bar_dkv_rope_done;

    array_aligned<uint32_t, 1> tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

}  // namespace sm100::bwd::head128_2kernels::dkv
