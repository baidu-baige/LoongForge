# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM context-parallel gather/compute/scatter fallback."""

import torch
import torch.distributed as dist
from megatron.core import mpu
from torch import Tensor

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None


def _get_partition_indices(origin_len, cp_size, cp_rank, packed_seq_params, device):
    if packed_seq_params is not None:
        assert tex is not None, "transformer-engine is not installed."
        assert packed_seq_params.qkv_format == "thd", (
            "if using Packing, only qkv_format=thd is supported"
        )
        cu_seqlens = packed_seq_params.cu_seqlens_q
        assert cu_seqlens is not None, (
            "cu_seqlens_q can not be None when qkv_format = thd!"
        )
        return tex.thd_get_partitioned_indices(
            cu_seqlens, origin_len, cp_size, cp_rank
        ).long().to(device=device)

    assert origin_len % (2 * cp_size) == 0, (
        "sequence length must be divisible by 2 * context parallel size"
    )
    chunk_len = origin_len // (2 * cp_size)
    chunk_ids = (cp_rank, 2 * cp_size - cp_rank - 1)
    return torch.cat(
        [
            torch.arange(
                chunk_id * chunk_len,
                (chunk_id + 1) * chunk_len,
                device=device,
            )
            for chunk_id in chunk_ids
        ]
    )


class _GatherContextParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, packed_seq_params, cp_group):
        cp_size = dist.get_world_size(cp_group)
        if cp_size == 1:
            ctx.cp_size = 1
            return value

        cp_rank = dist.get_rank(cp_group)
        origin_len = value.shape[0] * cp_size
        local_parts = [torch.empty_like(value) for _ in range(cp_size)]
        dist.all_gather(local_parts, value.contiguous(), group=cp_group)

        output = torch.empty(
            (origin_len, *value.shape[1:]), dtype=value.dtype, device=value.device
        )
        for rank, part in enumerate(local_parts):
            indices = _get_partition_indices(
                origin_len, cp_size, rank, packed_seq_params, value.device
            )
            if indices.numel() != part.shape[0]:
                raise RuntimeError(
                    "context-parallel partitions must have equal sequence lengths"
                )
            output.index_copy_(0, indices, part)

        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.cp_group = cp_group
        ctx.packed_seq_params = packed_seq_params
        ctx.origin_len = origin_len
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.cp_size == 1:
            return grad_output, None, None

        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.cp_group)
        indices = _get_partition_indices(
            ctx.origin_len,
            ctx.cp_size,
            ctx.cp_rank,
            ctx.packed_seq_params,
            grad_output.device,
        )
        return grad_output.index_select(0, indices), None, None


class _ScatterContextParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, packed_seq_params, cp_group):
        cp_size = dist.get_world_size(cp_group)
        if cp_size == 1:
            ctx.cp_size = 1
            return value

        cp_rank = dist.get_rank(cp_group)
        indices = _get_partition_indices(
            value.shape[0], cp_size, cp_rank, packed_seq_params, value.device
        )
        ctx.cp_size = cp_size
        ctx.origin_shape = value.shape
        ctx.save_for_backward(indices)
        return value.index_select(0, indices)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.cp_size == 1:
            return grad_output, None, None

        (indices,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.origin_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        grad_input.index_copy_(0, indices, grad_output)
        return grad_input, None, None


def gather_from_context_parallel_region(
    value: Tensor, packed_seq_params=None, cp_group=None
) -> Tensor:
    """Reconstruct the full sequence from DualChunkSwap CP partitions."""
    cp_group = cp_group or mpu.get_context_parallel_group()
    return _GatherContextParallel.apply(value, packed_seq_params, cp_group)


def scatter_to_context_parallel_region(
    value: Tensor, packed_seq_params=None, cp_group=None
) -> Tensor:
    """Select the current CP rank's partition after replicated computation."""
    cp_group = cp_group or mpu.get_context_parallel_group()
    return _ScatterContextParallel.apply(value, packed_seq_params, cp_group)
