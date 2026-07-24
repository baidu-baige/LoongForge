# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native Triton RoPE fast path for the native LingBot-VA backend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_interleaved_kernel(
    output,
    values,
    cos,
    sin,
    sequence_length,
    rotary_dim,
    rotary_sequence_length,
    stride_output_batch,
    stride_output_sequence,
    stride_output_heads,
    stride_output_dim,
    stride_values_batch,
    stride_values_sequence,
    stride_values_heads,
    stride_values_dim,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    sequence_block = tl.program_id(axis=0)
    batch_index = tl.program_id(axis=1)
    head_index = tl.program_id(axis=2)
    half_rotary_dim = rotary_dim // 2

    values += batch_index * stride_values_batch + head_index * stride_values_heads
    output += batch_index * stride_output_batch + head_index * stride_output_heads
    rows = sequence_block * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_K)
    swapped_columns = columns + ((columns + 1) % 2) * 2 - 1
    repeated_columns = columns // 2

    values_ptr = (
        values
        + rows[:, None] * stride_values_sequence
        + columns[None, :] * stride_values_dim
    )
    swapped_ptr = (
        values
        + rows[:, None] * stride_values_sequence
        + swapped_columns[None, :] * stride_values_dim
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
    x0 = tl.load(values_ptr, mask=row_mask & column_mask, other=0.0).to(tl.float32)
    x1 = tl.load(
        swapped_ptr,
        mask=row_mask & (swapped_columns[None, :] < rotary_dim),
        other=0.0,
    ).to(tl.float32)
    rotated = tl.where(
        columns[None, :] % 2 == 0,
        x0 * cos_values - x1 * sin_values,
        x0 * cos_values + x1 * sin_values,
    )
    output_ptr = (
        output
        + rows[:, None] * stride_output_sequence
        + columns[None, :] * stride_output_dim
    )
    tl.store(output_ptr, rotated, mask=row_mask & column_mask)


def _apply_rotary(
    values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    batch, sequence_length, heads, head_dim = values.shape
    rotary_sequence_length = cos.shape[0]
    rotary_dim = cos.shape[1] * 2
    output = torch.empty_like(values)
    if rotary_dim < head_dim:
        output[..., rotary_dim:].copy_(values[..., rotary_dim:])
    block_k = triton.next_power_of_2(rotary_dim)
    block_m = 4
    grid = (triton.cdiv(sequence_length, block_m), batch, heads)
    with torch.cuda.device(values.device.index):
        _rotary_interleaved_kernel[grid](
            output,
            values,
            cos,
            sin,
            sequence_length,
            rotary_dim,
            rotary_sequence_length,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            values.stride(3),
            BLOCK_K=block_k,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    return output


class _TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, cos, sin):
        """Apply the Triton RoPE kernel and save tensors for backward."""
        ctx.save_for_backward(cos, sin)
        return _apply_rotary(values.contiguous(), cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        """Rotate gradients back through the saved RoPE frequencies."""
        cos, sin = ctx.saved_tensors
        frequencies = torch.complex(cos.double(), sin.double())
        complex_grad = torch.view_as_complex(
            grad_output.double().reshape(*grad_output.shape[:-1], -1, 2)
        )
        grad_values = torch.view_as_real(
            complex_grad * frequencies.conj()[None, :, None]
        )
        return grad_values.flatten(3).to(grad_output.dtype), None, None


def apply_triton_rope(values: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE to ``[B, S, H, D]`` values."""
    if frequencies.ndim != 4 or frequencies.shape[0] != 1 or frequencies.shape[2] != 1:
        raise ValueError(
            f"Unsupported native LingBot RoPE shape: {tuple(frequencies.shape)}"
        )
    cos = frequencies.real[0, : values.shape[1], 0].contiguous()
    sin = frequencies.imag[0, : values.shape[1], 0].contiguous()
    return _TritonRoPE.apply(values, cos, sin)
