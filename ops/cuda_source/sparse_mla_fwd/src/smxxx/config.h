#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "params.h"
#include "defines.h"

namespace sm100::fwd::head128 {

using namespace cute;

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_O, typename TMA_O,
    typename Shape_P = void*, typename TMA_P = void*
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
    Shape_P shape_P; TMA_P tma_P;
    CUtensorMap tensor_map_kv;
};

struct float2x2 {
    float2 lo, hi;
};

template<int D_QK>
struct KernelTemplate {

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

static constexpr int B_H = 128;    // For 2 CTAs
static constexpr int B_TOPK = 128; // For 2 CTAs
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_THREADS = 256 + 128 + 128; // 128 scale & exp threads, 128x2 TMA threads, 32 UTCMMA threads


static constexpr int D_tQ = 384, NUM_tQ_TILES = D_tQ / 64;
static constexpr int D_sQ = D_QK-D_tQ, NUM_sQ_TILES = D_sQ / 64;
static_assert(D_sQ%64 == 0 && D_tQ%64 == 0 && D_sQ + D_tQ == D_Q);

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 320: P
    // 320 ~ 512: Q[D_QK-D_tQ:]
    static constexpr int o = 0;
    static constexpr int p = 256;
    static constexpr int q = 512 - D_tQ/2;
    static_assert(p+64 <= q);
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<8>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK/2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutV = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<256>, Int<B_TOPK>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutSTiles = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_INTER_Atom<bf16>{},
	Shape<Int<B_H/2>, Int<64*NUM_TILES>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutP = Layout<Shape<Int<B_H/2>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>;

struct SharedMemoryPlan {
    union {
        array_aligned<bf16, cosize_v<SmemLayoutQTiles<D_Q/64>>> q_full;
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQTiles<NUM_sQ_TILES>>> sq;
            array_aligned<bf16, cosize_v<SmemLayoutV>> v;
            // NOTE K is not overlapped with q_full, so we can do k copy-in while performing S->T copy for q
            static_assert(cosize_v<SmemLayoutQTiles<D_Q/64>> <= cosize_v<SmemLayoutQTiles<NUM_sQ_TILES>> + cosize_v<SmemLayoutV>);
            array_aligned<bf16, cosize_v<SmemLayoutKTiles<D_K/64>>> k;
        } s;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } u;
    array_aligned<bf16, cosize_v<SmemLayoutSTiles<2>>> s;
    float p[(B_H/2)*B_TOPK];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue_q, bar_prologue_utccp;
    transac_bar_t bar_qk_part_done[NUM_BUFS], bar_qk_done[NUM_BUFS];    // Pi = QKi^T done (i.e. Ki free)
    transac_bar_t bar_sv_part_done[NUM_BUFS], bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. Vi free)
    transac_bar_t bar_k_part0_ready[NUM_BUFS], bar_k_part1_ready[NUM_BUFS];
    transac_bar_t bar_v_part0_ready[NUM_BUFS], bar_v_part1_ready[NUM_BUFS];    // Vi is ready
    transac_bar_t bar_p_free[NUM_BUFS];
    transac_bar_t bar_so_ready[NUM_BUFS];   // S and O are ready
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
};

using TiledMMA_P_tQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_P_sQ = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}  // We use this permutation layout to let CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]
));

template<typename TmaParams>
static __device__ void
sparse_attn_fwd_kernel_devfunc(const SparseAttnFwdParams &params, const TmaParams &tma_params);

};

}

namespace sm100::fwd::head64 {

using namespace cute;

template<
    typename Shape_Q_NoPE, typename TMA_Q_NoPE,
    typename Shape_Q_RoPE, typename TMA_Q_RoPE,
    typename Shape_O, typename TMA_O,
    typename Shape_P = void*, typename TMA_P = void*
>
struct TmaParams {
    Shape_Q_NoPE shape_Q_nope; TMA_Q_NoPE tma_Q_nope;
    Shape_Q_RoPE shape_Q_rope; TMA_Q_RoPE tma_Q_rope;
    Shape_O shape_O; TMA_O tma_O;
    Shape_P shape_P; TMA_P tma_P;
    CUtensorMap tensor_map_kv_nope;
};

struct float2x2 {
    float2 lo, hi;
};

constexpr int D_Q = 576;
constexpr int D_K = 576;
constexpr int D_V = 512;
constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int NUM_BUFS = 3;
constexpr int NUM_THREADS = 128 + 128 + 128; // 128 scale & exp threads, 128 TMA threads, 32 UTCMMA threads


// Tensor memory columns
namespace tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 400: Q
    // 400 ~ 464: P
    constexpr int O = 0;
    constexpr int Q = 256;
    constexpr int Q_RoPE = 256 + 128;
    constexpr int P = 400;
}

using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_Q-D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<8>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE = SmemLayoutKTiles<8>;
using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutKNoPE{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutKRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<D_V/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));   // Re-view K-NoPE as B_TOPK*2 x D_V/2 for dual gemm

using SmemLayoutKRoPE_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_TOPK*2>, Int<64/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_INTER_Atom<bf16>{},
	Shape<Int<B_H>, Int<B_TOPK>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutP = Layout<Shape<Int<B_H>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>;

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> _k_rope_pad;
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> _k_pad[2];   // So that q_nope covers k[2]
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
        } q_full;
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope[NUM_BUFS];
        } k;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } u;
    float p_exchange_buf[4][32 * (B_TOPK/2)];
    union {
        struct {
            float rowwise_max_buf[128];
            float rowwise_li_buf[128];
        } rowwise_bufs;
    };
    union {
        bf16 s[B_H*B_TOPK];  // Used as s in main loop, and as partial P storage (64x64) for TMA output
        array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    } s_q_rope;
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue_q_nope, bar_prologue_q_rope, bar_prologue_utccp_nope, bar_prologue_utccp_rope;
    transac_bar_t bar_qk_nope_done[NUM_BUFS], bar_qk_rope_done;    // Pi = QKi^T (the nope part) done
    transac_bar_t bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. O, Si and Vi are free)
    transac_bar_t bar_kv_nope_ready[NUM_BUFS][2], bar_kv_rope_ready;
    transac_bar_t bar_p_free[NUM_BUFS];
    transac_bar_t bar_so_ready;   // S and O are ready
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, 128, UMMA::Major::K, UMMA::Major::K>{}  // Here we use N = 128 = 2*B_TOPK since we're going to use implicit dual gemm: <TODO Fill link here>
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    pepi_sync = 3,
};

}
