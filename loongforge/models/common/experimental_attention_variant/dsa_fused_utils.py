# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Deepseek Sparse Attention."""

from typing import Tuple
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import all_to_all


def shard_packed_cu_seqlens_for_sp_rank(
    global_cu_seqlens: torch.Tensor,
    *,
    sp_rank: int,
    sp_world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-rank packed metadata for sequence-parallel (SP) execution.

    Input `global_cu_seqlens` describes the packed token stream of total length T.
    SP rank `sp_rank` owns a contiguous slice [rank_token_start, rank_token_end)
    of length T / sp_world_size.

    This returns ONLY the sequences that overlap this rank.

    Returns:
        local_cu_seqlens:
            1D int32 tensor [K+1], cumulative offsets over local tokens on this rank,
            with a single leading 0. local_cu_seqlens[-1] == local_token_count.
        local_seq_start_offsets:
            1D int32 tensor [K], within-sequence starting offsets for each overlapped
            sequence fragment on this rank. Aligned with local_cu_seqlens[1:].
    """
    if global_cu_seqlens.dim() != 1 or global_cu_seqlens.numel() < 2:
        raise ValueError(
            f"global_cu_seqlens must be 1D with length >= 2, got {tuple(global_cu_seqlens.shape)}"
        )
    if not (0 <= sp_rank < sp_world_size):
        raise ValueError(f"sp_rank must be in [0, {sp_world_size}), got {sp_rank}")

    total_packed_tokens = int(global_cu_seqlens[-1].item())
    if total_packed_tokens % sp_world_size != 0:
        raise ValueError(
            f"total_packed_tokens ({total_packed_tokens}) must be divisible by "
            f"sp_world_size ({sp_world_size})"
        )

    local_token_count = total_packed_tokens // sp_world_size
    rank_token_start = sp_rank * local_token_count
    rank_token_end = rank_token_start + local_token_count

    num_seqs = global_cu_seqlens.numel() - 1
    device = global_cu_seqlens.device

    local_lengths: list[int] = []
    local_offsets: list[int] = []

    for seq_id in range(num_seqs):
        global_start = int(global_cu_seqlens[seq_id].item())
        global_end = int(global_cu_seqlens[seq_id + 1].item())

        overlap_start = max(global_start, rank_token_start)
        overlap_end = min(global_end, rank_token_end)

        overlap_len = overlap_end - overlap_start
        if overlap_len <= 0:
            continue

        within_seq_offset = overlap_start - global_start
        local_lengths.append(overlap_len)
        local_offsets.append(within_seq_offset)

    if not local_lengths:
        return (
            torch.zeros((1,), device=device, dtype=torch.int32),  # [0]
            torch.empty((0,), device=device, dtype=torch.int32),
        )

    local_lengths_t = torch.tensor(local_lengths, device=device, dtype=torch.int32)
    local_seq_start_offsets = torch.tensor(local_offsets, device=device, dtype=torch.int32)

    local_cu_seqlens = torch.empty((local_lengths_t.numel() + 1,), device=device, dtype=torch.int32)
    local_cu_seqlens[0] = 0
    local_cu_seqlens[1:] = torch.cumsum(local_lengths_t, dim=0)

    if int(local_cu_seqlens[-1].item()) != local_token_count:
        raise RuntimeError(
            "Local cu_seqlens is inconsistent with SP slice. "
            f"local_cu_seqlens[-1]={int(local_cu_seqlens[-1].item())}, "
            f"expected={local_token_count} (total={total_packed_tokens}, "
            f"sp_world_size={sp_world_size}, sp_rank={sp_rank})."
        )

    return local_cu_seqlens, local_seq_start_offsets


def all_to_all_hp2sp_with_padding(input_):
    """
    all_to_all with padding to handle cases where S is not divisible by TP.

    From [S, B, H/TP, D] -> [S_padded/TP, B, H, D]

    The inverse of all_to_all_sp2hp_with_padding, the returned output is padded
    in sequence dimension if padding is needed.
    """
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()

    s, *bhd = input_.shape

    remainder = s % world_size
    padding_size = (world_size - remainder) % world_size
    
    if padding_size > 0:
        out = input_.new_zeros(input_.shape[0] + padding_size, *input_.shape[1:])
        out[:input_.shape[0]] = input_
        input_ = out

    s_padded = input_.size(0)
    input_ = input_.reshape(world_size, s_padded // world_size, *bhd)

    input_exchanged = all_to_all(tp_group, input_)  # [TP, s/TP, *bhd]
    output = input_exchanged.movedim(0, -3)  # [s/TP, (b,) TP, h, d]
    output = output.flatten(-3, -2).contiguous()  # [s/TP, (b,) h, d]
    return output


def all_to_all_sp2hp_with_padding(input_, ori_s=None):
    """
    all_to_all with padding to handle cases where S is not divisible by TP.
    
    From [S/TP, B, H, D] -> [S, B, H/TP, D]
    
    The inverse of all_to_all_hp2sp_with_padding.
    
    Args:
        input_: [S/TP, B, H, D]
        ori_s: original sequence length before padding, used to remove padding after all_to_all
    """
    world_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    s_local, b, h, d = input_.shape
    h_local = h // world_size  # H/TP

    # Split the heads into world_size groups
    # [s_local, b, h, d] -> [s_local, b, world_size, h_local, d]
    input_ = input_.reshape(s_local, b, world_size, h_local, d)

    # move world_size to the front for all_to_all
    input_ = input_.movedim(2, 0).contiguous()  # [world_size, s_local, b, h_local, d]

    # all_to_all to gather sequence and scatter heads
    input_exchanged = all_to_all(tp_group, input_)

    # Concate sequence dim.
    output = input_exchanged.reshape(world_size * s_local, b, h_local, d)

    # Remove padding if needed
    if ori_s is not None:
        output = output[:ori_s, :, :, :].contiguous()

    return output


class _GatherHeadsAndSplitSequence(torch.autograd.Function):
    """Gather heads from tensor parallel region and split along sequence dimension."""

    @staticmethod
    def forward(ctx, input_):
        """input_: [seq_len, batch, heads_per_tp, head_dim]"""
        # Gather head and split sequence at the same time
        output = all_to_all_hp2sp_with_padding(input_)  # [seq_len_padded/tp, batch, total_heads, head_dim]
        ctx.orig_s = input_.size(0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """grad_output: [seq_len_padded/tp, batch, total_heads, head_dim]"""
        # Gather sequence and scatter head at the same time
        orig_s = ctx.orig_s
        output = all_to_all_sp2hp_with_padding(grad_output, orig_s)  # [seq_len, batch, heads_per_tp, head_dim]
        return output


class _GatherSequenceAndScatterHeads(torch.autograd.Function):
    """Gather sequence from sequence parallel region and scatter heads along tensor parallel."""

    @staticmethod
    def forward(ctx, input_):
        """input_: [batch, seq_len_padded/tp, total_heads, head_dim]"""
        # Swap batch and sequence dim to prepare for all_to_all_sp2hp_with_padding
        input_ = input_.transpose(0, 1).contiguous()
        # Gather sequence and scatter head at the same time
        output = all_to_all_sp2hp_with_padding(input_)  # [seq_len, batch, heads_per_tp, head_dim]
        # Swap back batch and sequence dim
        output = output.transpose(0, 1).contiguous()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """grad_output: [batch, seq_len, heads_per_tp, head_dim]"""
        # Swap batch and sequence dim to prepare for all_to_all_hp2sp_with_padding
        grad_output = grad_output.transpose(0, 1).contiguous()
        # Gather head and split sequence at the same time
        output = all_to_all_hp2sp_with_padding(grad_output)  # [seq_len_padded/tp, batch, total_heads, head_dim]
        # Swap back batch and sequence dim
        output = output.transpose(0, 1).contiguous()
        return output


def gather_heads_and_split_sequence(input_):
    """Wrapper for autograd function: gather heads + split sequence"""
    return _GatherHeadsAndSplitSequence.apply(input_)


def gather_sequence_and_scatter_heads(input_):
    """Wrapper for autograd function: gather sequence + scatter heads"""
    return _GatherSequenceAndScatterHeads.apply(input_)
