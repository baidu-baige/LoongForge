# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Utils """

import torch
import torch.distributed as dist
from torch import Tensor

from typing import Dict

from megatron.core import  mpu
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as mcore_get_pos_emb_on_this_cp_rank
)

from loongforge.utils import get_args

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None


class _Select(torch.autograd.Function):
    """Transformer Engine's PyTorch CP implementation currently utilizes
    the DualChunkSwap strategy to ensure load balancing across CP ranks.
    For qkv_format = 'thd', DualChunkSwap divides each sequence into (cp_size * 2) chunks
    and distributes 2 chunks of every sequence onto a CP rank.
    """
    @staticmethod
    def forward(ctx, val, packed_seq_params):
        """Forward function."""
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        if cp_size == 1:
            # no partitioning, passthrough
            ctx.cp_size = 1
            return val

        # packed_seq_params is None means not using packing
        if packed_seq_params is None:
            index = get_select_ids(cp_rank, cp_size)
            val = val.view(
                2 * cp_size,
                val.shape[0] // (2 * cp_size),
                *val.shape[1:]
            )
            val = val.index_select(0, index)
            val = val.view(-1, *val.shape[2: ])

            # Save for backward
            ctx.cp_size = cp_size
            ctx.index = index

            return val

        qkv_format = packed_seq_params.qkv_format
        assert tex is not None, "transformer-engine is not installed."
        assert qkv_format == "thd", "if using Packing, only qkv_format=thd is supported"

        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        assert (
            cu_seqlens_q is not None and cu_seqlens_kv is not None
        ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"

        origin_len = val.shape[0]
        seq_idx_val = tex.thd_get_partitioned_indices(
            cu_seqlens_q, origin_len, cp_size, cp_rank
        )
        val = val.index_select(0, seq_idx_val)

        # Save for backward
        ctx.cp_size = cp_size
        ctx.seq_idx_val = seq_idx_val.long().to(device=val.device)
        ctx.origin_len = origin_len
        ctx.device = val.device
        ctx.dtype = val.dtype

        return val

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if grad_output is None:
            return None, None

        if getattr(ctx, "cp_size", 1) == 1:
            return grad_output, None

        if getattr(ctx, "index", None) is not None:
            output = torch.zeros(
                2 * ctx.cp_size,
                grad_output.shape[0] // 2,
                *grad_output.shape[1:],
                dtype=grad_output.dtype,
                device=grad_output.device
            )
            cp_rank = mpu.get_context_parallel_rank()
            output[ctx.index] = grad_output.view(2, -1, *grad_output.shape[1:])
            output = output.view(-1, *output.shape[2: ])
            return output, None

        # grad_output.shape: [local_seq_len, ...]
        out_shape = (int(ctx.origin_len),) + tuple(grad_output.shape[1:])
        output = torch.zeros(out_shape, dtype=ctx.dtype, device=ctx.device)

        seq_idx = ctx.seq_idx_val
        if seq_idx.dtype != torch.long or seq_idx.device != output.device:
            seq_idx = seq_idx.long().to(device=output.device)

        output.index_copy_(0, seq_idx, grad_output)

        cp_group = mpu.get_context_parallel_group()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=cp_group)

        return output, None

def _get_cp_partition_indices(origin_len, cp_size, cp_rank, packed_seq_params, device):
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
    """Reconstruct the full sequence from DualChunkSwap CP partitions."""

    @staticmethod
    def forward(ctx, val, packed_seq_params, cp_group):
        cp_size = dist.get_world_size(cp_group)
        if cp_size == 1:
            ctx.cp_size = 1
            return val

        cp_rank = dist.get_rank(cp_group)
        origin_len = val.shape[0] * cp_size
        local_parts = [torch.empty_like(val) for _ in range(cp_size)]
        dist.all_gather(local_parts, val.contiguous(), group=cp_group)

        output = torch.empty(
            (origin_len, *val.shape[1:]), dtype=val.dtype, device=val.device
        )
        for rank, part in enumerate(local_parts):
            indices = _get_cp_partition_indices(
                origin_len, cp_size, rank, packed_seq_params, val.device
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
        indices = _get_cp_partition_indices(
            ctx.origin_len,
            ctx.cp_size,
            ctx.cp_rank,
            ctx.packed_seq_params,
            grad_output.device,
        )
        return grad_output.index_select(0, indices), None, None


class _ScatterContextParallel(torch.autograd.Function):
    """Select one CP partition from replicated full-sequence computation."""

    @staticmethod
    def forward(ctx, val, packed_seq_params, cp_group):
        cp_size = dist.get_world_size(cp_group)
        if cp_size == 1:
            ctx.cp_size = 1
            return val

        cp_rank = dist.get_rank(cp_group)
        indices = _get_cp_partition_indices(
            val.shape[0], cp_size, cp_rank, packed_seq_params, val.device
        )
        ctx.cp_size = cp_size
        ctx.origin_shape = val.shape
        ctx.save_for_backward(indices)
        return val.index_select(0, indices)

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

def get_select_ids(cp_rank: int, cp_size: int):
    """ Get select ids for each gpu."""
    return torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)

def get_inputs_on_this_cp_rank(val: Tensor, packed_seq_params=None):
    """Get input on each cp_rank

    Args:
        val (Tensor): Input data
        packed_seq_params (PackedSeqParams): Packing parameters

    Returns:
        val (Tensor): Input on this rank
    """

    return _Select.apply(val, packed_seq_params)

def gather_from_context_parallel_region(
    val: Tensor, packed_seq_params=None, cp_group=None
) -> Tensor:
    """Reconstruct a sequence partitioned by ``get_inputs_on_this_cp_rank``."""
    cp_group = cp_group or mpu.get_context_parallel_group()
    return _GatherContextParallel.apply(val, packed_seq_params, cp_group)


def scatter_to_context_parallel_region(
    val: Tensor, packed_seq_params=None, cp_group=None
) -> Tensor:
    """Select this rank's CP partition without performing gradient reduction."""
    cp_group = cp_group or mpu.get_context_parallel_group()
    return _ScatterContextParallel.apply(val, packed_seq_params, cp_group)

def get_pos_emb_on_this_cp_rank(pos_emb: Tensor, seq_dim: int, packed_seq_params=None) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
        packed_seq_params (PackedSeqParams): Packed sequence parameters

    Returns:
        pos_emb: Position embedding on this rank
    """
    if packed_seq_params is None:
        return mcore_get_pos_emb_on_this_cp_rank(pos_emb, seq_dim)
    else:
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        cu_seqlens = packed_seq_params.cu_seqlens_q
        seq_idx_val = tex.thd_get_partitioned_indices(
                        cu_seqlens, pos_emb.shape[seq_dim], cp_size, cp_rank
                    )
        pos_emb = pos_emb.index_select(seq_dim, seq_idx_val)

        return pos_emb


def get_batch_on_this_cp_rank(batch: Dict):
    """slice batch along sequence dimension for context parallelism"""
    # TODO: Currently only split labels and loss_mask, may split other parameters in the future
    args = get_args()

    if args.context_parallel_size > 1:
        labels = batch.get('labels', None)
        loss_mask = batch.get('loss_mask', None)
        packed_seq_params = batch.get('packed_seq_params', None)

        assert labels is not None and loss_mask is not None, "labels and loss_mask must in batch"

        labels = get_inputs_on_this_cp_rank(labels.transpose(0, 1), packed_seq_params).transpose(0, 1)
        loss_mask = get_inputs_on_this_cp_rank(loss_mask.transpose(0, 1), packed_seq_params).transpose(0, 1)

        batch['labels'] = labels
        batch['loss_mask'] = loss_mask

    return batch
