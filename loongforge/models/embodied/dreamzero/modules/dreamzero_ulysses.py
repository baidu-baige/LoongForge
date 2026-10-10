# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero Ulysses-style 4D all-to-all helpers.

DreamZero self-attention uses [B, L, H, D].  Context parallelism keeps the
sequence dimension sharded outside attention and uses all-to-all inside
attention to switch to a full-sequence, head-sharded layout.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _distributed_world_size(group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=group)


def _distributed_rank(group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group=group)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    return dim


def _split_sizes(total: int, world_size: int) -> list[int]:
    base = total // world_size
    remainder = total % world_size
    return [base + (1 if rank < remainder else 0) for rank in range(world_size)]


def _all_gather_int(value: int, group: dist.ProcessGroup | None, device: torch.device) -> list[int]:
    if device.type != "cuda":
        device = torch.device("cpu")
    local = torch.tensor([int(value)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(_distributed_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    return [int(item.cpu().item()) for item in gathered]


def _slice_dim(x: torch.Tensor, dim: int, start: int, length: int) -> torch.Tensor:
    return x.narrow(dim, start, length)


def _pad_dim(x: torch.Tensor, dim: int, target_size: int) -> torch.Tensor:
    current_size = x.shape[dim]
    if current_size == target_size:
        return x.contiguous()
    if current_size > target_size:
        raise ValueError(f"Cannot pad dim={dim} from {current_size} down to {target_size}")
    pad_shape = list(x.shape)
    pad_shape[dim] = target_size - current_size
    padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=dim).contiguous()


def _all_gather_variable_dim(
    x: torch.Tensor,
    group: dist.ProcessGroup | None,
    dim: int,
    sizes: list[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    world_size = _distributed_world_size(group)
    if world_size == 1:
        return x, [x.shape[dim]]
    dim = _normalize_dim(dim, x.dim())
    if sizes is None:
        sizes = _all_gather_int(x.shape[dim], group, x.device)
    if len(sizes) != world_size:
        raise ValueError(f"Expected {world_size} gather sizes, got {sizes}")
    max_size = max(sizes)
    padded = _pad_dim(x, dim, max_size)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    pieces = [
        _slice_dim(tensor, dim, 0, size)
        for tensor, size in zip(gathered, sizes, strict=True)
    ]
    return torch.cat(pieces, dim=dim).contiguous(), sizes


def _check_4d_all_to_all_input(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    world_size: int,
) -> None:
    if x.dim() != 4:
        raise ValueError(f"DreamZero Ulysses expects a 4D [B, L, H, D] tensor, got {tuple(x.shape)}")
    scatter_dim = _normalize_dim(scatter_dim, x.dim())
    gather_dim = _normalize_dim(gather_dim, x.dim())
    if scatter_dim not in (1, 2) or gather_dim not in (1, 2) or scatter_dim == gather_dim:
        raise ValueError(
            "DreamZero Ulysses currently supports sequence/head swaps only: "
            f"got scatter_dim={scatter_dim}, gather_dim={gather_dim}"
        )
    if scatter_dim == 2 and x.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"Head dimension with size {x.shape[scatter_dim]} must be divisible by "
            f"context-parallel world size {world_size}"
        )


def _all_to_all_4d_list(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None,
    *,
    phase: str,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    scatter_dim = _normalize_dim(scatter_dim, x.dim())
    gather_dim = _normalize_dim(gather_dim, x.dim())
    input_list = [
        chunk.contiguous()
        for chunk in torch.split(
            x, _split_sizes(x.shape[scatter_dim], world_size), dim=scatter_dim
        )
    ]
    if scatter_dim == 2 and gather_dim == 1:
        peer_seq_sizes = _all_gather_int(x.shape[1], group, x.device)
        output_list = []
        for peer_seq_size in peer_seq_sizes:
            shape = list(input_list[0].shape)
            shape[1] = peer_seq_size
            output_list.append(torch.empty(shape, dtype=x.dtype, device=x.device))
    elif scatter_dim == 1 and gather_dim == 2:
        local_shape = list(input_list[rank].shape)
        output_list = [
            torch.empty(local_shape, dtype=x.dtype, device=x.device)
            for _ in range(world_size)
        ]
    else:
        raise ValueError(f"Unsupported DreamZero 4D all-to-all swap: {scatter_dim}->{gather_dim}")
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _single_all_to_all_4d(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None,
    *,
    phase: str,
) -> torch.Tensor:
    """4D all_to_all_single for [B, L, H, D] sequence/head swaps.

    The extra reshapes keep batch order stable when B > 1:
    - scatter H, gather L: [B, L/cp, H, D] -> [B, L, H/cp, D]
    - scatter L, gather H: [B, L, H/cp, D] -> [B, L/cp, H, D]
    """

    world_size = dist.get_world_size(group=group)
    bsz, seq, heads, head_dim = x.shape

    if scatter_dim == 2 and gather_dim == 1:
        heads_per_rank = heads // world_size
        input_t = (
            x.reshape(bsz, seq, world_size, heads_per_rank, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        output_t = torch.empty_like(input_t)
        dist.all_to_all_single(output_t, input_t, group=group)
        return (
            output_t.permute(1, 0, 2, 3, 4)
            .reshape(bsz, seq * world_size, heads_per_rank, head_dim)
            .contiguous()
        )

    if scatter_dim == 1 and gather_dim == 2:
        seq_per_rank = seq // world_size
        input_t = (
            x.reshape(bsz, world_size, seq_per_rank, heads, head_dim)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        output_t = torch.empty_like(input_t)
        dist.all_to_all_single(output_t, input_t, group=group)
        return (
            output_t.permute(1, 2, 0, 3, 4)
            .reshape(bsz, seq_per_rank, heads * world_size, head_dim)
            .contiguous()
        )

    raise ValueError(f"Unsupported DreamZero 4D all-to-all swap: {scatter_dim}->{gather_dim}")


def _can_use_single_all_to_all_4d(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None,
    world_size: int,
) -> bool:
    scatter_dim = _normalize_dim(scatter_dim, x.dim())
    gather_dim = _normalize_dim(gather_dim, x.dim())
    if x.shape[scatter_dim] % world_size != 0:
        return False
    if scatter_dim == 2 and gather_dim == 1:
        peer_seq_sizes = _all_gather_int(x.shape[1], group, x.device)
        return all(size == peer_seq_sizes[0] for size in peer_seq_sizes)
    return True


class DreamZeroSeqAllToAll4D(torch.autograd.Function):
    """Autograd wrapper for DreamZero 4D all-to-all sequence/head swaps."""

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup | None,
        x: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        single_all_to_all: bool,
        phase: str,
    ) -> torch.Tensor:
        """Forward pass performing the 4D all-to-all sequence/head swap."""
        world_size = _distributed_world_size(group)
        if world_size == 1:
            ctx.group = group
            ctx.scatter_dim = scatter_dim
            ctx.gather_dim = gather_dim
            ctx.single_all_to_all = single_all_to_all
            return x
        _check_4d_all_to_all_input(x, scatter_dim, gather_dim, world_size)
        use_single = single_all_to_all and _can_use_single_all_to_all_4d(
            x, scatter_dim, gather_dim, group, world_size
        )
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.single_all_to_all = use_single
        if use_single:
            return _single_all_to_all_4d(x, scatter_dim, gather_dim, group, phase=phase)
        return _all_to_all_4d_list(x, scatter_dim, gather_dim, group, phase=phase)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        """Backward pass swapping the gradient back across the group."""
        grad_input = DreamZeroSeqAllToAll4D.apply(
            ctx.group,
            grad_output,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.single_all_to_all,
            "backward",
        )
        return None, grad_input, None, None, None, None


def all_to_all_4d(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None = None,
    *,
    single_all_to_all: bool = True,
) -> torch.Tensor:
    """Swap one [B, L, H, D] dimension across a distributed group."""

    return DreamZeroSeqAllToAll4D.apply(
        group,
        x,
        scatter_dim,
        gather_dim,
        single_all_to_all,
        "forward",
    )


def sequence_to_head_parallel(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    *,
    single_all_to_all: bool = True,
) -> torch.Tensor:
    """[B, L/cp, H, D] -> [B, L, H/cp, D]."""

    return all_to_all_4d(x, 2, 1, group, single_all_to_all=single_all_to_all)


def head_to_sequence_parallel(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    *,
    single_all_to_all: bool = True,
) -> torch.Tensor:
    """[B, L, H/cp, D] -> [B, L/cp, H, D]."""

    return all_to_all_4d(x, 1, 2, group, single_all_to_all=single_all_to_all)


class DreamZeroSplitGather(torch.autograd.Function):
    """Split in forward and gather in backward for sequence-parallel tensors."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None, dim: int) -> torch.Tensor:
        """Split the tensor along ``dim`` and keep the local rank's shard."""
        world_size = _distributed_world_size(group)
        rank = _distributed_rank(group)
        ctx.group = group
        ctx.dim = dim
        if world_size == 1:
            return x
        dim = _normalize_dim(dim, x.dim())
        sizes = _split_sizes(x.shape[dim], world_size)
        ctx.sizes = sizes
        return torch.split(x, sizes, dim=dim)[rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        """Gather the gradient shards back along ``dim``."""
        group = ctx.group
        dim = ctx.dim
        world_size = _distributed_world_size(group)
        if world_size == 1:
            return grad_output, None, None
        grad_input, _ = _all_gather_variable_dim(
            grad_output.contiguous(),
            group,
            dim,
            sizes=getattr(ctx, "sizes", None),
        )
        return grad_input, None, None


class DreamZeroGatherSplit(torch.autograd.Function):
    """Gather in forward and split in backward for sequence-parallel tensors."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None, dim: int) -> torch.Tensor:
        """Gather the tensor along ``dim`` across the distributed group."""
        world_size = _distributed_world_size(group)
        ctx.group = group
        ctx.dim = dim
        if world_size == 1:
            return x
        gathered, sizes = _all_gather_variable_dim(x.contiguous(), group, dim)
        ctx.sizes = sizes
        return gathered

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        """Split the gradient along ``dim`` and keep the local rank's shard."""
        group = ctx.group
        dim = ctx.dim
        world_size = _distributed_world_size(group)
        rank = _distributed_rank(group)
        if world_size == 1:
            return grad_output, None, None
        dim = _normalize_dim(dim, grad_output.dim())
        sizes = getattr(ctx, "sizes", None)
        if sizes is None:
            sizes = _split_sizes(grad_output.shape[dim], world_size)
        return torch.split(grad_output, sizes, dim=dim)[rank].contiguous(), None, None


def split_sequence_forward_gather_backward(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    *,
    dim: int = 1,
) -> torch.Tensor:
    """[B, L, ...] -> local [B, L/cp, ...], gathering gradients in backward."""

    return DreamZeroSplitGather.apply(x, group, dim)


def gather_sequence_forward_split_backward(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    *,
    dim: int = 1,
) -> torch.Tensor:
    """local [B, L/cp, ...] -> [B, L, ...], splitting gradients in backward."""

    return DreamZeroGatherSplit.apply(x, group, dim)
