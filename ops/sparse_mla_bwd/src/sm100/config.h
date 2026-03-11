#pragma once

#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::bwd::head128 {

using namespace cute;

template<
    typename Shape_QNoPE, typename TMA_QNoPE,
    typename Shape_QRoPE, typename TMA_QRoPE,
    typename Shape_KV, typename TMA_KV,
    typename Shape_dO, typename TMA_dO,
    typename Shape_dQ, typename TMA_dQ
>
struct TmaParams {
    Shape_QNoPE shape_Q_nope;
    TMA_QNoPE tma_Q_nope;
    Shape_QRoPE shape_Q_rope;
    TMA_QRoPE tma_Q_rope;
    Shape_KV shape_KV;
    TMA_KV tma_KV;
    Shape_dO shape_dO;
    TMA_dO tma_dO;
    Shape_dQ shape_dQ;
    TMA_dQ tma_dQ;
    CUtensorMap tensor_map_kv;
};

static constexpr int D_QK = 576;
static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr int D_ROPE = D_Q - D_V;
static constexpr int B_H = 128;
static constexpr int B_TOPK = 64;
static constexpr int NUM_THREADS = 4 * 128;  // 4 warp-groups

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQNoPE = SmemLayoutQTiles<8>;
using SmemLayoutQRoPE = SmemLayoutQTiles<1>;
using SmemLayoutQ = SmemLayoutQTiles<9>;
using SmemLayoutdO = SmemLayoutQTiles<D_V / 64>;

template<int NUM_TILES>
using SmemLayoutQTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64 * NUM_TILES>, Int<B_H / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutQNoPETransposed = SmemLayoutQTilesTransposed<4>;
using SmemLayoutQRoPETransposed = SmemLayoutQTilesTransposed<1>;
using SmemLayoutdOTransposed = SmemLayoutQTilesTransposed<4>;

template<int NUM_TILES>
using SmemLayoutKVTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK / 2>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE = SmemLayoutKVTiles<8>;
using SmemLayoutKRoPE = SmemLayoutKVTiles<1>;
using SmemLayoutKV = SmemLayoutKVTiles<9>;
using SmemLayoutV = SmemLayoutKNoPE;

template<int NUM_TILES>
using SmemLayoutKVTilesTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<64 * NUM_TILES>, Int<B_TOPK / 2>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPETransposed = SmemLayoutKVTilesTransposed<4>;
using SmemLayoutKRoPETransposed = SmemLayoutKVTilesTransposed<1>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<B_H / 2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutdS = SmemLayoutS;

using SmemLayoutdSTransposed = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_INTER_Atom<bf16>{},
    Shape<Int<B_H / 2>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_dP = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_dQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H / 2, 256, UMMA::Major::MN, UMMA::Major::MN>{}
));

using TiledMMA_dQ_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H / 2, D_ROPE, UMMA::Major::MN, UMMA::Major::MN>{}
));

using TiledMMA_dKV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

using TiledMMA_dKV_RoPE = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_TOPK, D_ROPE, UMMA::Major::K, UMMA::Major::MN>{}
));

struct tmem_cols {
    static constexpr int dQ = 0;
    static constexpr int dQ_RoPE = 256;
    static constexpr int dKV = 288;
    static constexpr int dKV_RoPE = 416;
    static constexpr int P = 448;
    static constexpr int dP = 480;
};

struct alignas(128) SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope;
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
            array_aligned<bf16, cosize_v<SmemLayoutKV>> kv_peer;
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
            array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
        } q_kv;
        array_aligned<bf16, cosize_v<SmemLayoutQ>> dq;
    } u;

    array_aligned<bf16, cosize_v<SmemLayoutdO>> dO;
    struct {
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
        array_aligned<bf16, cosize_v<SmemLayoutS>> ds;
    } s_ds;
    char is_k_valid[B_TOPK / 8];

    transac_bar_t bar_prologue_q_nope;
    transac_bar_t bar_prologue_q_rope;
    transac_bar_t bar_prologue_kv;
    transac_bar_t bar_prologue_dO;
    transac_bar_t bar_p_ready;
    transac_bar_t bar_dp_ready;
    transac_bar_t bar_s_ready;
    transac_bar_t bar_ds_ready;
    transac_bar_t bar_k_valid_free;
    transac_bar_t bar_k_valid_ready;
    transac_bar_t bar_dkv_part0_ready;
    transac_bar_t bar_dkv_part1_ready;
    transac_bar_t bar_dkv_part2_ready;
    transac_bar_t bar_dkv_part0_done;
    transac_bar_t bar_dkv_part1_done;
    transac_bar_t bar_dkv_part2_done;
    transac_bar_t bar_kv_peer_cp_async;
    transac_bar_t bar_kv_peer_ready;
    transac_bar_t bar_dq_ready;

    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128];
    float rowwise_li_buf[128];
    float rowwise_delta_buf[128];
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemoryPlan);

}  // namespace sm100::bwd::head128
