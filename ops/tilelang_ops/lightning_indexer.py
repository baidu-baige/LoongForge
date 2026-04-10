"""Lightning Indexer Kernels"""

import torch
import time
from typing import Tuple, Optional
import numpy as np
import tilelang
import tilelang.language as T


# ==================== baseline ====================
def fp8_index_baseline(
    q: torch.Tensor, 
    q_s: torch.Tensor,
    k: torch.Tensor, 
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Baseline for indexer forward.
    """
    b, m, h, d = q.shape
    n = k.shape[1]

    logits = torch.einsum('bmhd,bnd->bmhn', q, k)
    relu_logits = torch.relu(logits)
    scaled_logits = relu_logits * q_s.unsqueeze(-1)     # [B, Seq_q, H, Seq_k] 
    logits_sum = scaled_logits.sum(dim=2)               # [B, Seq_q, Seq_k] 
    index_score = logits_sum * k_s.unsqueeze(1)         # [B, Seq_q, Seq_k] 

    cache = (q, k, logits, relu_logits, scaled_logits, logits_sum)
    
    return index_score, cache


def fp8_index_baseline_backward(d_output, q, q_s, k, k_s, output, cache=None):
    """
    Baseline for indexer backward.
    """
    b, m, h, d = q.shape
    n = k.shape[1]
    
    if cache is None:
        q_fp32 = q.float()
        k_fp32 = k.float()
        logits = torch.einsum('bmhd,bnd->bmhn', q_fp32, k_fp32)
        relu_logits = torch.relu(logits)
        scaled_logits = relu_logits * q_s.unsqueeze(-1)
        logits_sum = scaled_logits.sum(dim=2)
    else:
        q_fp32, k_fp32, logits, relu_logits, scaled_logits, logits_sum = cache

    d_logits_sum = d_output * k_s.unsqueeze(1)  # (b, m, n)
    d_k_s = (d_output * logits_sum).sum(dim=1)  # (b, n)

    d_scaled_logits = d_logits_sum.unsqueeze(2).expand(b, m, h, n)  # (b, m, h, n)

    d_relu_logits = d_scaled_logits * q_s.unsqueeze(-1)  # (b, m, h, n)
    d_q_s = (d_scaled_logits * relu_logits).sum(dim=-1)  # (b, m, h)

    d_logits = d_relu_logits * (logits > 0).float()  # (b, m, h, n)

    d_q = torch.einsum('bmhn,bnd->bmhd', d_logits.float(), k_fp32)  # (b, m, h, d)
    d_k = torch.einsum('bmhn,bmhd->bnd', d_logits.float(), q_fp32)  # (b, n, d)
    
    return d_q, d_q_s, d_k, d_k_s


# ==================== tilelang ====================
tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def fp8_index_kernel(h: int, d: int):
    """Build the TileLang FP8 forward kernel for fixed head-count and head-dim."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def fp8_index_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(m * T.ceildiv(n, blk_n1), b) as (blk_x, blk_y):
            i1_n = blk_x // m
            i_m = blk_x - i1_n * m
            i_b = blk_y

            q_smem = T.alloc_shared((h, d), FP8)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), FP8)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                k_s_frag = T.alloc_fragment(blk_n2, FP32)
                T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                for i3_n in T.Parallel(blk_n2):
                    logits_sum[i3_n] *= k_s_frag[i3_n]

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return fp8_index_kernel_


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    q = q.contiguous().view(q.shape)
    q_s = q_s.contiguous().view(q_s.shape)
    k = k.contiguous().view(k.shape)
    k_s = k_s.contiguous().view(k_s.shape)

    return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)


@tilelang.jit(out_idx=[5, 6], pass_configs=pass_configs)
def fp8_index_q_backward_kernel(h: int, d: int):
    """Build the TileLang backward kernel branch that computes d_q and d_q_s."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 128

    @T.prim_func
    def fp8_index_q_backward_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        d_output: T.Tensor[(b, m, n), FP32],
        d_q: T.Tensor[(b, m, h, d), FP32],
        d_q_s: T.Tensor[(b, m, h), FP32],
    ) -> None:
        with T.Kernel(b, m, threads=256) as (i_b, i_m):
            q_smem = T.alloc_shared((h, d), FP8)
            q_s_frag = T.alloc_fragment(h, FP32)
            T.copy(q[i_b, i_m, 0, 0], q_smem)
            T.copy(q_s[i_b, i_m, 0], q_s_frag)

            d_q_s_frag = T.alloc_fragment((h, blk_n1), FP32)
            d_q_accum = T.alloc_fragment((h, d), FP32)
            d_q_s_accum = T.alloc_fragment(h, FP32)
            T.clear(d_q_s_frag)
            T.clear(d_q_accum)
            T.clear(d_q_s_accum)

            for i_n1 in T.Pipelined(T.ceildiv(n, blk_n1), num_stages=2):
                i_n_offset = i_n1 * blk_n1
                k_smem = T.alloc_shared((blk_n1, d), FP8)
                k_s_frag = T.alloc_fragment(blk_n1, FP32)
                d_output_frag = T.alloc_fragment(blk_n1, FP32)

                T.copy(k[i_b, i_n_offset, 0], k_smem)
                T.copy(k_s[i_b, i_n_offset], k_s_frag)
                T.copy(d_output[i_b, i_m, i_n_offset], d_output_frag)

                d_logits_sum = T.alloc_fragment(blk_n1, FP32)
                for i_n2 in T.Parallel(blk_n1):
                    d_logits_sum[i_n2] = d_output_frag[i_n2] * k_s_frag[i_n2]

                logits = T.alloc_fragment((h, blk_n1), FP32)
                T.gemm(
                    q_smem,   # (h, d)
                    k_smem,   # (blk_n1, d)
                    logits,   # (h, blk_n1)
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                d_logits = T.alloc_shared((h, blk_n1), FP32)
                for i_h, i_n2 in T.Parallel(h, blk_n1):
                    d_logits[i_h, i_n2] = T.if_then_else(
                        logits[i_h, i_n2] > 0,
                        d_logits_sum[i_n2] * q_s_frag[i_h],
                        0,
                    )
                    d_q_s_frag[i_h, i_n2] += T.if_then_else(
                        logits[i_h, i_n2] > 0,
                        d_logits_sum[i_n2] * logits[i_h, i_n2],
                        0,
                    )

                k_smem_fp32 = T.alloc_shared((blk_n1, d), FP32)
                T.copy(k_smem, k_smem_fp32)

                T.gemm(
                    d_logits,         # (h, blk_n1)
                    k_smem_fp32,      # (blk_n1, d)
                    d_q_accum,        # (h, d)
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=False
                )

            T.reduce_sum(d_q_s_frag, d_q_s_accum, dim=1)
            T.copy(d_q_s_accum, d_q_s[i_b, i_m, 0])
            T.copy(d_q_accum, d_q[i_b, i_m, 0, 0])
    
    return fp8_index_q_backward_kernel_


@tilelang.jit(pass_configs=pass_configs)
def fp8_index_k_backward_kernel(h: int, d: int):
    """Build the TileLang backward kernel branch that accumulates d_k and d_k_s."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_m = 64
    blk_n = 128

    @T.prim_func
    def fp8_index_k_backward_kernel_(
        q: T.Tensor[(b, m, h, d), FP8],
        q_s: T.Tensor[(b, m, h), FP32],
        k: T.Tensor[(b, n, d), FP8],
        k_s: T.Tensor[(b, n), FP32],
        d_output: T.Tensor[(b, m, n), FP32],
        d_k: T.Tensor[(b, n, d), FP32],
        d_k_s: T.Tensor[(b, n), FP32],
    ) -> None:
        with T.Kernel(b, T.ceildiv(m, blk_m), T.ceildiv(n, blk_n), threads=256) as (i_b, i_m1, i_n1):
            i_n_offset = i_n1 * blk_n
            
            k_smem = T.alloc_shared((blk_n, d), FP8)
            k_s_frag = T.alloc_fragment(blk_n, FP32)
            T.copy(k[i_b, i_n_offset, 0], k_smem)
            T.copy(k_s[i_b, i_n_offset], k_s_frag)

            d_k_frag = T.alloc_fragment((blk_n, d), FP32)
            d_k_s_accum = T.alloc_fragment(blk_n, FP32)
            T.clear(d_k_frag)
            T.clear(d_k_s_accum)

            for i_m2 in T.Pipelined(blk_m, num_stages=2):
                i_m = i_m1 * blk_m + i_m2

                q_smem = T.alloc_shared((h, d), FP8)
                q_s_frag = T.alloc_fragment(h, FP32)
                d_output_frag = T.alloc_fragment(blk_n, FP32)

                T.copy(q[i_b, i_m, 0, 0], q_smem)
                T.copy(q_s[i_b, i_m, 0], q_s_frag)
                T.copy(d_output[i_b, i_m, i_n_offset], d_output_frag)

                logits = T.alloc_fragment((blk_n, h), FP32)
                T.gemm(
                    k_smem,   # (blk_n, d)
                    q_smem,   # (h, d)
                    logits,   # (blk_n, h)
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                scaled_logits = T.alloc_fragment((blk_n, h), FP32)
                for i_n2, i_h in T.Parallel(blk_n, h):
                    scaled_logits[i_n2, i_h] = T.max(logits[i_n2, i_h], 0) * q_s_frag[i_h]

                logits_sum = T.alloc_fragment(blk_n, FP32)
                T.clear(logits_sum)
                T.reduce_sum(scaled_logits, logits_sum, dim=1)

                for i_n2 in T.Parallel(blk_n):
                    d_k_s_accum[i_n2] += d_output_frag[i_n2] * logits_sum[i_n2]

                d_logits = T.alloc_shared((blk_n, h), FP32)
                for i_n2, i_h in T.Parallel(blk_n, h):
                    d_logits[i_n2, i_h] = T.if_then_else(
                        logits[i_n2, i_h] > 0,
                        d_output_frag[i_n2] * k_s_frag[i_n2] * q_s_frag[i_h],
                        0,
                    )

                q_smem_fp32 = T.alloc_shared((h, d), FP32)
                T.copy(q_smem, q_smem_fp32)
                T.gemm(
                    d_logits,        # (blk_n, h)
                    q_smem_fp32,     # (h, d)
                    d_k_frag,        # (blk_n, d)
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=False,
                )

            for i_n2 in T.Parallel(blk_n):
                T.atomic_add(d_k_s[i_b, i_n_offset + i_n2], d_k_s_accum[i_n2])
            for i_n2, i_d in T.Parallel(blk_n, d):
                T.atomic_add(d_k[i_b, i_n_offset + i_n2, i_d], d_k_frag[i_n2, i_d])
    
    return fp8_index_k_backward_kernel_


def fp8_index_backward(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    d_output: torch.Tensor,
):
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.
        d_output (torch.Tensor): The grad of output, must be contiguous.
    """
    q = q.contiguous().view(q.shape)
    q_s = q_s.contiguous().view(q_s.shape)
    k = k.contiguous().view(k.shape)
    k_s = k_s.contiguous().view(k_s.shape)
    d_output = d_output.contiguous().view(d_output.shape)

    b, m, h, d = q.shape
    _, n, _ = k.shape

    d_k = torch.zeros([b, n, d], dtype=torch.float32, device=q.device)
    d_k_s = torch.zeros([b, n], dtype=torch.float32, device=q.device)

    d_q, d_q_s = fp8_index_q_backward_kernel(h, d)(q, q_s, k, k_s, d_output)
    fp8_index_k_backward_kernel(h, d)(q, q_s, k, k_s, d_output, d_k, d_k_s)

    return d_q, d_q_s, d_k, d_k_s


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def bf16_index_kernel(h: int, d: int):
    """Build the TileLang BF16 forward kernel for index-score computation."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_n1 = 512
    blk_n2 = 128

    @T.prim_func
    def bf16_index_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        weights: T.Tensor[(m, b, h), BF16],
        k: T.Tensor[(b, n, d), BF16],
        softmax_scale: T.float32,
        o: T.Tensor[(b, m, n), FP32],
    ) -> None:
        with T.Kernel(m * T.ceildiv(n, blk_n1), b) as (blk_x, blk_y):
            i1_n = blk_x // m
            i_m = blk_x - i1_n * m
            i_b = blk_y

            q_smem = T.alloc_shared((h, d), BF16)
            T.copy(q[i_b, i_m, 0, 0], q_smem)

            weights_frag = T.alloc_fragment(h, BF16)
            T.copy(weights[i_m, i_b, 0], weights_frag)

            for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                k_smem = T.alloc_shared((blk_n2, d), BF16)
                T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                logits = T.alloc_fragment((blk_n2, h), FP32)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for i_h, i3_n in T.Parallel(h, blk_n2):
                    logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * weights_frag[i_h] * softmax_scale

                logits_sum = T.alloc_fragment(blk_n2, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)

                T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return bf16_index_kernel_


def bf16_index(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float = 0.0
) -> torch.Tensor:
    """
    Perform index score using BF16 precision.
    """
    q = q.contiguous().view(q.shape)
    k = k.contiguous().view(k.shape)
    weights = weights.contiguous().view(weights.shape)
    
    return bf16_index_kernel(q.shape[2], q.shape[3])(q, weights, k, softmax_scale)


@tilelang.jit(pass_configs=pass_configs)
def bf16_index_backward_kernel(h: int, d: int):
    """Build the TileLang BF16 backward kernel for q, k, and weight gradients."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")

    blk_m = 64
    blk_n = 128

    @T.prim_func
    def bf16_index_backward_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        k: T.Tensor[(b, n, d), BF16],
        weights: T.Tensor[(m, b, h), BF16],
        softmax_scale: T.float32,
        d_output: T.Tensor[(b, m, n), FP32],
        d_q: T.Tensor[(b, m, h, d), FP32],
        d_k: T.Tensor[(b, n, d), FP32],
        d_weights: T.Tensor[(m, b, h), FP32],
    ) -> None:
        with T.Kernel(b, T.ceildiv(m, blk_m), T.ceildiv(n, blk_n), threads=256) as (i_b, i_m1, i_n1):
            i_n_offset = i_n1 * blk_n
            
            k_smem = T.alloc_shared((blk_n, d), BF16)
            T.copy(k[i_b, i_n_offset, 0], k_smem)

            d_k_frag = T.alloc_fragment((blk_n, d), FP32)
            T.clear(d_k_frag)

            for i_m2 in T.Pipelined(blk_m, num_stages=2):
                i_m = i_m1 * blk_m + i_m2

                q_smem = T.alloc_shared((h, d), BF16)
                weights_frag = T.alloc_shared(h, BF16)
                d_output_frag = T.alloc_shared(blk_n, FP32)

                T.copy(q[i_b, i_m, 0, 0], q_smem)
                T.copy(weights[i_m, i_b, 0], weights_frag)
                T.copy(d_output[i_b, i_m, i_n_offset], d_output_frag)

                logits = T.alloc_fragment((blk_n, h), FP32)
                T.gemm(
                    k_smem,   # (blk_n, d)
                    q_smem,   # (h, d)
                    logits,   # (blk_n, h)
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                d_logits = T.alloc_shared((blk_n, h), BF16)
                d_weights_frag = T.alloc_fragment((h, blk_n), FP32)
                for i_n2, i_h in T.Parallel(blk_n, h):
                    d_logits[i_n2, i_h] = T.if_then_else(
                        logits[i_n2, i_h] > 0,
                        d_output_frag[i_n2] * weights_frag[i_h] * softmax_scale,
                        0
                    )
                    d_weights_frag[i_h, i_n2] = T.if_then_else(
                        logits[i_n2, i_h] >= 0,
                        d_output_frag[i_n2] * logits[i_n2, i_h] * softmax_scale,
                        0
                    )

                d_weights_accum = T.alloc_fragment(h, FP32)
                T.reduce_sum(d_weights_frag, d_weights_accum, dim=-1)
                for i_h in T.Parallel(h):
                    T.atomic_add(d_weights[i_m, i_b, i_h], d_weights_accum[i_h])

                T.gemm(
                    d_logits,   # (blk_n, h)
                    q_smem,     # (h, d)
                    d_k_frag,   # (blk_n, d)
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=False,
                )

                d_q_frag = T.alloc_fragment((h, d), FP32)
                T.gemm(
                    d_logits,   # (blk_n, h)
                    k_smem,     # (blk_n, d)
                    d_q_frag,   # (h, d)
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=True,
                )
                for i_h, i_d in T.Parallel(h, d):
                    T.atomic_add(d_q[i_b, i_m, i_h, i_d], d_q_frag[i_h, i_d])

            for i_n2, i_d in T.Parallel(blk_n, d):
                T.atomic_add(d_k[i_b, i_n_offset + i_n2, i_d], d_k_frag[i_n2, i_d])
    
    return bf16_index_backward_kernel_


def bf16_index_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    d_output: torch.Tensor,
):
    """
    Perform index score using BF16 precision.
    """
    q = q.contiguous().view(q.shape)
    k = k.contiguous().view(k.shape)
    weights = weights.contiguous().view(weights.shape)

    b, m, h, d = q.shape
    _, n, _ = k.shape

    d_q = torch.zeros([b, m, h, d], dtype=torch.float32, device=q.device)
    d_k = torch.zeros([b, n, d], dtype=torch.float32, device=q.device)
    d_weights = torch.zeros([m, b, h], dtype=torch.float32, device=q.device)

    bf16_index_backward_kernel(h, d)(q, k, weights, softmax_scale, d_output, d_q, d_k, d_weights)
    return d_q, d_k, d_weights


def fp8_indexer(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
    """TileLang FP8 indexer forward interface."""
    return fp8_index(q, q_s, k, k_s)


def fp8_indexer_bwd(
    d_output: torch.Tensor,
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
):
    """TileLang FP8 indexer backward interface (same ordering as tests)."""
    return fp8_index_backward(q, q_s, k, k_s, d_output.contiguous())


def ref_fp8_indexer(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
    """Reference FP8 indexer forward interface."""
    index_score, _ = fp8_index_baseline(q.float(), q_s.float(), k.float(), k_s.float())
    return index_score


def ref_fp8_indexer_bwd(
    d_output: torch.Tensor,
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
):
    """Reference FP8 indexer backward interface."""
    return fp8_index_baseline_backward(
        d_output.contiguous(),
        q.float(),
        q_s.float(),
        k.float(),
        k_s.float(),
        None,
    )


class BF16IndexerFunction(torch.autograd.Function):
    """Autograd wrapper around the TileLang BF16 indexer kernels."""

    @staticmethod
    def forward(ctx, q, k, weights):
        """
        Tilelang BF16Indexer forward.
        """
        assert q.shape[-1] == 128, (
            f"Last dimension size of q must be 128, but got {q.shape[-1]}."
        )
        assert k.shape[-1] == 128, (
            f"Last dimension size of k must be 128, but got {k.shape[-1]}."
        )

        softmax_scale = q.shape[-1] ** -0.5
        ctx.save_for_backward(q, k, weights)
        ctx.softmax_scale = softmax_scale

        index_score = bf16_index(q, weights, k, softmax_scale)
        index_score = index_score.requires_grad_(True)
        return index_score
    
    @staticmethod
    def backward(ctx, d_output):
        """TileLang BF16 indexer backward."""

        q, k, weights = ctx.saved_tensors
        return bf16_index_backward(q, k, weights, ctx.softmax_scale, d_output.contiguous())


class BF16Indexer(torch.nn.Module):
    """Convenience module for the TileLang BF16 indexer autograd function.

    This module wraps the BF16IndexerFunction autograd function, providing
    a PyTorch nn.Module interface for computing index scores between query
    and key tensors using BF16 precision.

    The index score computation involves:
    1. Computing dot products between q and k: (B, M, H, N)
    2. Applying ReLU to the logits
    3. Scaling by weights (pre-multiplied by softmax_scale)
    4. Summing over heads dimension: (B, M, N)

    Example:
        >>> indexer = BF16Indexer()
        >>> q = torch.randn(1, 1024, 64, 128, dtype=torch.bfloat16, device="cuda")
        >>> k = torch.randn(1, 2048, 128, dtype=torch.bfloat16, device="cuda")
        >>> weights = torch.randn(1024, 1, 64, dtype=torch.bfloat16, device="cuda")
        >>> index_score = indexer(q, k, weights)
        >>> # index_score.shape: (1, 1024, 2048)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, weights):
        """Compute index scores between query and key tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (B, M, H, D) where
                B = batch size, M = query sequence length,
                H = number of heads, D = head dimension (must be 128).
            k (torch.Tensor): Key tensor of shape (B, N, D) where
                N = key sequence length.
            weights (torch.Tensor): Weight tensor of shape (M, B, H), typically
                representing attention weights or other scaling factors. These
                weights are internally multiplied by softmax_scale (D ** -0.5).

        Returns:
            torch.Tensor: Index score tensor of shape (B, M, N) containing
                the computed index scores in float32 precision.

        Raises:
            AssertionError: If the last dimension of q or k is not 128.
        """
        return BF16IndexerFunction.apply(q, k, weights)


class BF16IndexerBaseline(torch.nn.Module):
    """PyTorch BF16 baseline used for accuracy and performance comparison."""

    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, weights):
        """
        Baseline BF16Indexer forward.

        Args:
            q: Query tensor.
            k: Key tensor.
            weights: Weights tensor.

        Returns:
            index_score: Computed index score.
        """
        assert q.shape[-1] == 128, (
            f"Last dimension size of q must be 128, but got {q.shape[-1]}."
        )
        assert k.shape[-1] == 128, (
            f"Last dimension size of k must be 128, but got {k.shape[-1]}."
        )

        softmax_scale = q.shape[-1] ** -0.5
        weights_transposed = torch.transpose(weights, 0, 1).contiguous()
        q_s = weights_transposed * softmax_scale

        logits = torch.einsum('bmhd,bnd->bmhn', q, k)  # (B, M, H, N)
        logits = logits.float()
        relu_logits = torch.relu(logits)
        scaled_logits = relu_logits * q_s.unsqueeze(-1)
        logits_sum = scaled_logits.sum(dim=2)  # (B, M, N)
        index_score = logits_sum

        return index_score


class TestArgs(object):
    """Default arguments used by local accuracy and performance checks."""

    scale_fmt: str = None
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048


def quantize_rowwise_to_fp8(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize along the last dim and return fp8 tensor with rowwise inverse scale."""
    fp8_max = torch.tensor(448.0, device=x.device, dtype=torch.float32)
    amax = x.float().abs().amax(dim=-1).clamp_min(eps)
    scale = fp8_max / amax
    x_fp8 = (x.float() * scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    scale_inv = (1.0 / scale).to(torch.float32)
    return x_fp8, scale_inv


def prepare_test_data(args: TestArgs, batch_size: int, seq_len: int, kv_seq_len: int):
    """Prepare inputs following lightning_indexer_bwd/tests quantization style."""
    torch.manual_seed(42)

    num_heads = args.index_n_heads
    head_dim = args.index_head_dim
    q_bf16 = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_bf16 = torch.randn(batch_size, kv_seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

    softmax_scale = head_dim ** -0.5
    weights_raw = (
        torch.randn(seq_len, batch_size, num_heads, device="cuda", dtype=torch.float32)
        * softmax_scale
    )

    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    k_fp8, k_scale_inv = quantize_rowwise_to_fp8(k_bf16)
    q_s = weights_raw.transpose(0, 1).contiguous()

    return q_bf16, k_bf16, weights_raw, q_fp8, q_s, k_fp8, k_scale_inv


def performance_test(
    fn: callable,
    msg: str
):
    """Performance test for fp8 indexer interfaces."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
    print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=50))


def forward_accuracy_test(
    args: TestArgs,
    batch_size: int = 1,
    seq_len: int = 1024,
    kv_seq_len: int = 2048,
    relative_tolerance: float = 1e-4
):
    """Use relative error to verify forward correctness."""
    _, _, _, q_fp8, q_s, k_fp8, k_s = prepare_test_data(args, batch_size, seq_len, kv_seq_len)

    tl_result = fp8_indexer(q_fp8, q_s, k_fp8, k_s)
    ref_result = ref_fp8_indexer(q_fp8, q_s, k_fp8, k_s)

    abs_diff = torch.abs(tl_result - ref_result)
    rel_diff = abs_diff / (torch.abs(ref_result) + 1e-8)

    max_rel_diff = rel_diff.max().item()
    avg_rel_diff = rel_diff.mean().item()

    within_tolerance = rel_diff < relative_tolerance
    pass_rate = within_tolerance.float().mean().item()

    print("Forward Accuracy Test:")
    print(f"  Max-diff: {max_rel_diff:.6f}")
    print(f"  Avg-diff: {avg_rel_diff:.6f}")
    print(f"  Tolerance: {relative_tolerance}")
    print(f"  Pass Rate: {pass_rate:.4f}")


def forward_performance_test(
    args: TestArgs,
    batch_size: int = 1,
    seq_len: int = 1024,
):
    """Forward performance analysis: tilelang vs reference."""
    print("\n" + "=" * 70)
    print("Forward performance Analysis: TileLang vs Reference")
    print("=" * 70 + "\n")

    _, _, _, q_fp8, q_s, k_fp8, k_s = prepare_test_data(args, batch_size, seq_len, seq_len)

    def tilelang_indexer_benchmark():
        return fp8_indexer(q_fp8, q_s, k_fp8, k_s)

    def ref_indexer_benchmark():
        return ref_fp8_indexer(q_fp8, q_s, k_fp8, k_s)

    performance_test(tilelang_indexer_benchmark, "tilelang forward kernel")
    performance_test(ref_indexer_benchmark, "reference forward kernel")


def backward_accuracy_test(
    args: TestArgs,
    batch_size: int = 1,
    seq_len: int = 1024,
    kv_seq_len: int = 2048,
):
    """Verify backward correctness against the FP8 reference implementation."""
    _, _, weights_raw, q_fp8, q_s, k_fp8, k_s = prepare_test_data(
        args, batch_size, seq_len, kv_seq_len
    )
    d_output = torch.ones(batch_size, seq_len, kv_seq_len, device="cuda", dtype=torch.float32)

    ref_d_q, ref_d_q_s, ref_d_k, ref_d_k_s = ref_fp8_indexer_bwd(d_output, q_fp8, q_s, k_fp8, k_s)
    tl_d_q, tl_d_q_s, tl_d_k, tl_d_k_s = fp8_indexer_bwd(d_output, q_fp8, q_s, k_fp8, k_s)

    def print_diff(a, b, msg):
        abs_diff = torch.abs(a - b)
        rel_diff = abs_diff / (torch.abs(b) + 1e-8)
        print(
            f"  {msg} max diff: {abs_diff.max().item():.4f}, "
            f"rel diff: {rel_diff.mean().item() * 100:.4f}%"
        )

    print("Backward Accuracy Test:")
    print_diff(tl_d_q, ref_d_q, "d_q")
    print_diff(tl_d_q_s, ref_d_q_s, "d_q_s")
    print_diff(tl_d_k, ref_d_k, "d_k")
    print_diff(tl_d_k_s, ref_d_k_s, "d_k_s")


def backward_performance_test(
    args: TestArgs,
    batch_size: int = 1,
    seq_len: int = 1024,
):
    """Backward performance analysis: tilelang vs reference."""
    print("\n" + "=" * 70)
    print("Backward performance Analysis: TileLang vs Reference")
    print("=" * 70 + "\n")

    _, _, _, q_fp8, q_s, k_fp8, k_s = prepare_test_data(args, batch_size, seq_len, seq_len)
    d_output = torch.ones(batch_size, seq_len, seq_len, device="cuda", dtype=torch.float32)

    def tilelang_indexer_benchmark():
        return fp8_indexer_bwd(d_output, q_fp8, q_s, k_fp8, k_s)

    def ref_indexer_benchmark():
        return ref_fp8_indexer_bwd(d_output, q_fp8, q_s, k_fp8, k_s)
    performance_test(tilelang_indexer_benchmark, "tilelang backward kernel")
    performance_test(ref_indexer_benchmark, "reference backward kernel")


if __name__ == "__main__":
    args = TestArgs()
    forward_accuracy_test(args)
    forward_performance_test(args)
    backward_accuracy_test(args)
    backward_performance_test(args)

