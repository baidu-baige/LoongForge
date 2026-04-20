# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tilelang kernels for Sparse MLA backward pass."""

# ruff: noqa
import tilelang
from tilelang import language as T
import torch


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """
    cu_seqlens is a 1D tensor of shape [B+1], where B is the batch size, 
    and cu_seqlens[i] is the cumulative sequence length of the first i sequences
    in the batch. This function returns a 1D tensor of shape [B], 
    where the i-th element is the sequence length of the i-th sequence in the batch.
    """
    return torch.diff(cu_seqlens)


def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """
    This function takes the cumulative sequence lengths and returns a 1D 
    tensor of position ids for each token in the batch.
    """
    lens = prepare_lens(cu_seqlens)
    return torch.cat(
        [
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in lens.unbind()
        ]
    )


def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """
    This function takes the cumulative sequence lengths and returns a 1D tensor
    of sequence ids for each token in the batch.
    """
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """
    This function takes the cumulative sequence lengths and returns a 2D tensor of shape [S, 2],
    where S is the total number of tokens in the batch. The first column contains the sequence ids for each token,
    and the second column contains the position ids for each token.
    """
    position_ids = prepare_position_ids(cu_seqlens)
    seq_ids = prepare_sequence_ids(cu_seqlens)
    return torch.stack([seq_ids, position_ids], 1).to(cu_seqlens)


@tilelang.jit(out_idx=[-1])
def preprocess(
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """
    Preprocess kernel to compute Delta for sparse MLA backward.
    Delta is the sum of dS across the KV dimension, which is needed for computing dP.
    """

    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    shape = [S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND)) as (bx, by):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)

            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(
                    O[
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    o,
                )
                T.copy(
                    dO[
                        by * block_ND : (by + 1) * block_ND,
                        bx,
                        k * block_ND : (k + 1) * block_ND,
                    ],
                    do,
                )
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]

            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    S_kv,
    D,
    D_tail,
    kv_group=1,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Postprocess kernel to convert dKV from accum_dtype to dtype and store back to global memory."""

    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    dkv_shape = [S_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, threads=threads) as (bx, by):
            T.copy(
                dKV[bx * block_N : (bx + 1) * block_N, by, :],
                dKV_out[bx * block_N : (bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def bwd(
    S,
    S_kv,
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Sparse MLA backward kernel."""

    assert is_causal is True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)

    B_plus_one = T.symbolic("B_plus_one")
    S = T.symbolic("S")
    S_kv = T.symbolic("S_kv")

    H_kv = H // kv_group
    q_shape = [S, H, D + D_tail]
    k_shape = [S_kv, kv_group, D + D_tail]
    o_shape = [S, H, D]
    indices_shape = [S, kv_group, topk]
    delta_shape = [S, H]
    lse_shape = [S, H]
    offsets_shape = [B_plus_one]
    token_indices_shape = [S, 2]

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mla_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        Offsets: T.Tensor(offsets_shape, indices_dtype),
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
        q_start_index_s: T.Tensor(1, indices_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S, kv_group, threads=threads) as (b_s_i, bz):
            Q_shared = T.alloc_shared([padded_H, D], dtype)
            Q_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)

            dO_shared = T.alloc_shared([padded_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([padded_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([padded_H, BS], dtype)

            acc_p = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([padded_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([padded_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)

            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]

            max_kv_i = q_start_index_s[0] + s_i

            T.copy(Q[b_s_i, bz * padded_H : (bz + 1) * padded_H, :D], Q_shared)
            T.copy(Q[b_s_i, bz * padded_H : (bz + 1) * padded_H, D:], Q_tail_shared)
            T.copy(dO[b_s_i, bz * padded_H : (bz + 1) * padded_H, :D], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # validity mask
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = (
                        (Indices[b_s_i, bz, i_i * BS + bi_i] <= max_kv_i)
                        & (Indices[b_s_i, bz, i_i * BS + bi_i] != -1)
                    )

                # init attn scores
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_p.dtype)
                    )

                # load KV block
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[
                        bos + Indices[b_s_i, bz, i_i * BS + bi_i], bz, d_i
                    ]
                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[
                        bos + Indices[b_s_i, bz, i_i * BS + bi_i], bz, D + d_i
                    ]

                T.gemm(
                    Q_shared[:, 0:256],
                    KV_shared[:, 0:256],
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    wg_wait=0,
                )
                T.gemm(
                    Q_tail_shared,
                    KV_tail_shared,
                    acc_p,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    wg_wait=0,
                )

                # softmax prob (sparse)
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.exp(
                        acc_p[h_i, bi_i] * sm_scale - Lse[b_s_i, bz * padded_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                # dP = P * (dS - Delta) * sm_scale
                T.gemm(
                    dO_shared,
                    KV_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i]
                        * (acc_dp[h_i, bi_i] - Delta[bos + s_i, bz * padded_H + h_i])
                        * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)

                # dQ
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol, wg_wait=-1)
                T.gemm(
                    dP_shared_cast,
                    KV_tail_shared,
                    acc_dq_tail,
                    policy=T.GemmWarpPolicy.FullCol,
                    wg_wait=-1,
                )

                # dKV = dP^T Q + P^T dO
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                    wg_wait=-1,
                )
                T.gemm(
                    P_shared_cast,
                    dO_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    wg_wait=-1,
                )

                # dKV_tail = dP^T Q_tail
                T.clear(acc_dkv_tail)
                T.gemm(
                    dP_shared_cast,
                    Q_tail_shared,
                    acc_dkv_tail,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    wg_wait=0,
                )

                # atomic accumulate into global dKV
                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS // split_store, D):
                        acc_dkv_shared[bi_i, d_i] = acc_dkv[
                            bi_i + s * (BS // split_store), d_i
                        ]
                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail):
                        acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[
                            bi_i + s * (BS // split_store), d_i
                        ]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[
                                bos
                                + Indices[
                                    b_s_i, bz, i_i * BS + bi_i + s * (BS // split_store)
                                ],
                                bz,
                                d_i * 4,
                            ],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dKV[
                                bos
                                + Indices[
                                    b_s_i, bz, i_i * BS + bi_i + s * (BS // split_store)
                                ],
                                bz,
                                D + d_i * 4,
                            ],
                            acc_dkv_tail_shared[bi_i, d_i * 4],
                        )

            # store dQ
            T.copy(acc_dq, Q_shared)
            T.copy(acc_dq_tail, Q_tail_shared)

            T.copy(Q_shared, dQ[b_s_i, bz * padded_H : (bz + 1) * padded_H, :D])
            T.copy(Q_tail_shared, dQ[b_s_i, bz * padded_H : (bz + 1) * padded_H, D:])

    return sparse_mla_bwd_kernel


def sparse_mla_bwd_interface(
    q,
    kv,
    o,
    do,
    indices,
    lse,
    offsets,
    chunk_offset,
    sm_scale=None,
    is_casual=True,
    return_kernel=False,
    delta=None,
):
    """
    Sparse MLA backward interface function.
    This function will be called by the PyTorch autograd function for sparse MLA backward.
    """

    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()

    q = q.view(q.shape)
    kv = kv.view(kv.shape)
    indices = indices.view(indices.shape)
    o = o.view(o.shape)
    do = do.view(do.shape)

    S, H, dim_plus_tail_dim = q.shape
    S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim

    D = 512
    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]

    assert indices.shape == (S, kv_group, topk)
    assert lse.shape == (S, H)

    token_indices = prepare_token_indices(offsets)

    preprocess_kernel = preprocess(S, H, D)
    if H == 128:
        # Split H into two halves to avoid shared memory OOM when H=128
        H_half = H // 2
        bwd_kernel = bwd(S, S_kv, H_half, D, D_tail, topk, kv_group, sm_scale, is_casual)
    else:
        bwd_kernel = bwd(S, S_kv, H, D, D_tail, topk, kv_group, sm_scale, is_casual)

    postprocess_kernel = postprocess(S_kv, D, D_tail, kv_group)

    if delta is None:
        delta = preprocess_kernel(o, do)

    dkv = torch.zeros_like(kv, dtype=torch.float32)

    q_start = torch.tensor([int(chunk_offset)], dtype=torch.int32, device="cuda")

    if H == 128:
        H_half = H // 2
        dq_first = bwd_kernel(
            q[:, :H_half, :].contiguous(),
            kv,
            do[:, :H_half, :].contiguous(),
            indices,
            lse[:, :H_half].contiguous(),
            delta[:, :H_half].contiguous(),
            offsets,
            token_indices,
            q_start,
            dkv,
        )
        dq_second = bwd_kernel(
            q[:, H_half:, :].contiguous(),
            kv,
            do[:, H_half:, :].contiguous(),
            indices,
            lse[:, H_half:].contiguous(),
            delta[:, H_half:].contiguous(),
            offsets,
            token_indices,
            q_start,
            dkv,
        )
        dq = torch.cat([dq_first, dq_second], dim=1)
    else:
        dq = bwd_kernel(
            q, kv, do, indices, lse, delta, offsets, token_indices, q_start, dkv
        )

    dkv = postprocess_kernel(dkv)
    return dq, dkv
