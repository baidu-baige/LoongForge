# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native Triton RoPE fast path for the native LingBot-VA backend."""

import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_interleaved_pair_kernel(
    output_q,
    output_k,
    values_q,
    values_k,
    cos,
    sin,
    sequence_length,
    rotary_dim,
    rotary_sequence_length,
    stride_output_q_batch,
    stride_output_q_sequence,
    stride_output_q_heads,
    stride_output_q_dim,
    stride_output_k_batch,
    stride_output_k_sequence,
    stride_output_k_heads,
    stride_output_k_dim,
    stride_values_q_batch,
    stride_values_q_sequence,
    stride_values_q_heads,
    stride_values_q_dim,
    stride_values_k_batch,
    stride_values_k_sequence,
    stride_values_k_heads,
    stride_values_k_dim,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    sequence_block = tl.program_id(axis=0)
    batch_index = tl.program_id(axis=1)
    head_index = tl.program_id(axis=2)
    half_rotary_dim = rotary_dim // 2

    values_q += batch_index * stride_values_q_batch + head_index * stride_values_q_heads
    values_k += batch_index * stride_values_k_batch + head_index * stride_values_k_heads
    output_q += batch_index * stride_output_q_batch + head_index * stride_output_q_heads
    output_k += batch_index * stride_output_k_batch + head_index * stride_output_k_heads
    rows = sequence_block * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_K)
    swapped_columns = columns + ((columns + 1) % 2) * 2 - 1
    repeated_columns = columns // 2

    values_q_ptr = (
        values_q
        + rows[:, None] * stride_values_q_sequence
        + columns[None, :] * stride_values_q_dim
    )
    values_k_ptr = (
        values_k
        + rows[:, None] * stride_values_k_sequence
        + columns[None, :] * stride_values_k_dim
    )
    swapped_q_ptr = (
        values_q
        + rows[:, None] * stride_values_q_sequence
        + swapped_columns[None, :] * stride_values_q_dim
    )
    swapped_k_ptr = (
        values_k
        + rows[:, None] * stride_values_k_sequence
        + swapped_columns[None, :] * stride_values_k_dim
    )
    cos_ptr = cos + rows[:, None] * half_rotary_dim + repeated_columns[None, :]
    sin_ptr = sin + rows[:, None] * half_rotary_dim + repeated_columns[None, :]

    row_mask = rows[:, None] < sequence_length
    column_mask = columns[None, :] < rotary_dim
    rotary_mask = (rows[:, None] < rotary_sequence_length) & (
        repeated_columns[None, :] < half_rotary_dim
    )
    cos_values = tl.load(cos_ptr, mask=rotary_mask, other=1.0)
    sin_values = tl.load(sin_ptr, mask=rotary_mask, other=0.0)
    q0 = tl.load(values_q_ptr, mask=row_mask & column_mask, other=0.0).to(tl.float32)
    q1 = tl.load(
        swapped_q_ptr,
        mask=row_mask & (swapped_columns[None, :] < rotary_dim),
        other=0.0,
    ).to(tl.float32)
    k0 = tl.load(values_k_ptr, mask=row_mask & column_mask, other=0.0).to(tl.float32)
    k1 = tl.load(
        swapped_k_ptr,
        mask=row_mask & (swapped_columns[None, :] < rotary_dim),
        other=0.0,
    ).to(tl.float32)
    if BACKWARD:
        rotated_q = tl.where(
            columns[None, :] % 2 == 0,
            q0 * cos_values + q1 * sin_values,
            q0 * cos_values - q1 * sin_values,
        )
        rotated_k = tl.where(
            columns[None, :] % 2 == 0,
            k0 * cos_values + k1 * sin_values,
            k0 * cos_values - k1 * sin_values,
        )
    else:
        rotated_q = tl.where(
            columns[None, :] % 2 == 0,
            q0 * cos_values - q1 * sin_values,
            q0 * cos_values + q1 * sin_values,
        )
        rotated_k = tl.where(
            columns[None, :] % 2 == 0,
            k0 * cos_values - k1 * sin_values,
            k0 * cos_values + k1 * sin_values,
        )
    output_q_ptr = (
        output_q
        + rows[:, None] * stride_output_q_sequence
        + columns[None, :] * stride_output_q_dim
    )
    output_k_ptr = (
        output_k
        + rows[:, None] * stride_output_k_sequence
        + columns[None, :] * stride_output_k_dim
    )
    tl.store(output_q_ptr, rotated_q, mask=row_mask & column_mask)
    tl.store(output_k_ptr, rotated_k, mask=row_mask & column_mask)


def _apply_rotary_pair(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    backward: bool = False,
):
    if query.shape != key.shape:
        raise ValueError(
            f"Fused Q/K RoPE requires identical shapes, got {query.shape} and {key.shape}"
        )
    batch, sequence_length, heads, head_dim = query.shape
    rotary_sequence_length = cos.shape[0]
    rotary_dim = cos.shape[1] * 2
    output_q = torch.empty_like(query)
    output_k = torch.empty_like(key)
    if rotary_dim < head_dim:
        output_q[..., rotary_dim:].copy_(query[..., rotary_dim:])
        output_k[..., rotary_dim:].copy_(key[..., rotary_dim:])
    block_k = triton.next_power_of_2(rotary_dim)
    block_m = QK_ROPE_BLOCK_M
    num_warps = QK_ROPE_NUM_WARPS
    grid = (triton.cdiv(sequence_length, block_m), batch, heads)
    with torch.cuda.device(query.device.index):
        _rotary_interleaved_pair_kernel[grid](
            output_q,
            output_k,
            query,
            key,
            cos,
            sin,
            sequence_length,
            rotary_dim,
            rotary_sequence_length,
            output_q.stride(0),
            output_q.stride(1),
            output_q.stride(2),
            output_q.stride(3),
            output_k.stride(0),
            output_k.stride(1),
            output_k.stride(2),
            output_k.stride(3),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            BLOCK_K=block_k,
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=1,
            BACKWARD=backward,
        )
    return output_q, output_k


# Frozen recipe values for the fused QK-RoPE kernel launch.
QK_ROPE_BLOCK_M = 4
QK_ROPE_NUM_WARPS = 4

_ROPE_COMPONENT_CACHE = {}


def _rope_components(frequencies: torch.Tensor):
    key = (
        id(frequencies),
        tuple(frequencies.shape),
        tuple(frequencies.stride()),
        str(frequencies.device),
        int(frequencies._version),
    )
    cached = _ROPE_COMPONENT_CACHE.get(key)
    if cached is not None and cached[0]() is frequencies:
        return cached[1], cached[2]
    cos = frequencies.real[0, :, 0].contiguous()
    sin = frequencies.imag[0, :, 0].contiguous()
    if len(_ROPE_COMPONENT_CACHE) >= 64:
        dead = [cache_key for cache_key, value in _ROPE_COMPONENT_CACHE.items() if value[0]() is None]
        for cache_key in dead:
            _ROPE_COMPONENT_CACHE.pop(cache_key, None)
        if len(_ROPE_COMPONENT_CACHE) >= 64:
            _ROPE_COMPONENT_CACHE.pop(next(iter(_ROPE_COMPONENT_CACHE)))
    _ROPE_COMPONENT_CACHE[key] = (weakref.ref(frequencies), cos, sin)
    return cos, sin


class _TritonRoPEPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, cos, sin):
        """Rotate Q and K with one Triton launch and save frequencies for backward."""
        ctx.save_for_backward(cos, sin)
        return _apply_rotary_pair(query.contiguous(), key.contiguous(), cos, sin)

    @staticmethod
    def backward(ctx, grad_query, grad_key):
        """Inverse-rotate both gradients with one Triton launch."""
        cos, sin = ctx.saved_tensors
        grad_query, grad_key = _apply_rotary_pair(
            grad_query.contiguous(),
            grad_key.contiguous(),
            cos,
            sin,
            backward=True,
        )
        return grad_query, grad_key, None, None


def apply_triton_rope_pair(
    query: torch.Tensor, key: torch.Tensor, frequencies: torch.Tensor
):
    """Apply identical interleaved RoPE to Q and K in one Triton launch."""
    if query.shape != key.shape:
        raise ValueError(
            f"Fused Q/K RoPE requires identical shapes, got {query.shape} and {key.shape}"
        )
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("Fused Q/K RoPE requires matching device and dtype")
    if frequencies.ndim != 4 or frequencies.shape[0] != 1 or frequencies.shape[2] != 1:
        raise ValueError(
            f"Unsupported native LingBot RoPE shape: {tuple(frequencies.shape)}"
        )
    cos, sin = _rope_components(frequencies)
    sequence_length = query.shape[1]
    return _TritonRoPEPair.apply(
        query,
        key,
        cos[:sequence_length],
        sin[:sequence_length],
    )
