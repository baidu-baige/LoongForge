"""Utils"""

import torch
from torch import Tensor
import torch.distributed as dist

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None

from megatron.core import mpu
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank,
)


class _Select(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, val):
        """Forward function."""
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            return val

        cp_rank = mpu.get_context_parallel_rank()
        index = get_select_ids(cp_rank, cp_size)

        val = val.view(2 * cp_size, val.shape[0] // (2 * cp_size), *val.shape[1:])
        val = val.index_select(0, index)
        val = val.view(-1, *val.shape[2:])
        return val

    @staticmethod
    def backward(ctx, val):
        """Backward function."""
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            return val

        output = torch.zeros(
            2 * cp_size,
            val.shape[0] // 2,
            *val.shape[1:],
            dtype=val.dtype,
            device=val.device,
        )
        cp_rank = mpu.get_context_parallel_rank()
        index = get_select_ids(cp_rank, cp_size)
        output[index] = val.view(2, -1, *val.shape[1:])
        output = output.view(-1, *output.shape[2:])
        return output


def get_select_ids(cp_rank, cp_size):
    """Get select ids for each gpu."""
    return torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)


def get_inputs_on_this_cp_rank(val):
    """Slice input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """
    return _Select.apply(val)


class _SelectByTex(torch.autograd.Function):
    """Transformer Engine's PyTorch CP implementation currently utilizes
    the DualChunkSwap strategy to ensure load balancing across CP ranks.
    For qkv_format = 'thd', DualChunkSwap divides each sequence into (cp_size * 2) chunks and distributes 2 chunks of
    every sequence onto a CP rank.
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

        qkv_format = packed_seq_params.qkv_format
        assert tex is not None, "transformer-engine is not installed."
        assert qkv_format == "thd", "if using Packing, only qkv_format=thd is supported"

        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        assert (
            cu_seqlens_q is not None and cu_seqlens_kv is not None
        ), "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"

        orig_len = val.shape[0]
        seq_idx_val = tex.thd_get_partitioned_indices(
            cu_seqlens_q, orig_len, cp_size, cp_rank
        )
        val_selected = val.index_select(0, seq_idx_val)

        # Save for backward
        ctx.cp_size = cp_size
        ctx.seq_idx_val = seq_idx_val.long().to(device=val.device)
        ctx.orig_len = orig_len
        ctx.device = val.device
        ctx.dtype = val.dtype

        return val_selected

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if grad_output is None:
            return None, None

        if getattr(ctx, "cp_size", 1) == 1:
            return grad_output, None

        orig_len = int(ctx.orig_len)
        device = ctx.device
        dtype = ctx.dtype

        # grad_output.shape: [local_seq_len, ...]
        out_shape = (orig_len,) + tuple(grad_output.shape[1:])
        grad_val = torch.zeros(out_shape, dtype=dtype, device=device)

        seq_idx = ctx.seq_idx_val
        if seq_idx.dtype != torch.long or seq_idx.device != grad_val.device:
            seq_idx = seq_idx.long().to(device=grad_val.device)

        grad_val.index_copy_(0, seq_idx, grad_output)

        cp_group = mpu.get_context_parallel_group()
        dist.all_reduce(grad_val, op=dist.ReduceOp.SUM, group=cp_group)

        return grad_val, None


def get_inputs_on_this_cp_rank_by_tex(val, packed_seq_params=None):
    """Get input on each rank

    Args:
        val (Tensor): Input data
        packed_seq_params (PackedSeqParams): Packing parameters

    Returns:
        val (Tensor): Input on this rank
    """
    # packed_seq_params is None means not using packing
    if packed_seq_params is None:
        return get_inputs_on_this_cp_rank(val)
    else:
        return _SelectByTex.apply(val, packed_seq_params)


def get_pos_emb_on_this_cp_rank_by_tex(
    pos_emb: Tensor, seq_dim: int, packed_seq_params
) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
        packed_seq_params (PackedSeqParams): Packed sequence parameters

    Returns:
        pos_emb: Position embedding on this rank
    """
    if packed_seq_params is None:
        return get_pos_emb_on_this_cp_rank(pos_emb, seq_dim)
    else:
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

        cu_seqlens = packed_seq_params.cu_seqlens_q

        seq_idx_val = tex.thd_get_partitioned_indices(
            cu_seqlens, pos_emb.shape[seq_dim], cp_size, cp_rank
        )
        pos_emb = pos_emb.index_select(seq_dim, seq_idx_val)

        return pos_emb
