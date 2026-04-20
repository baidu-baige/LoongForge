# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""comunication functions"""

import torch
import torch.distributed as dist


# ====================
# Gather-Split
# ====================


def _split(input_, pg: dist.ProcessGroup, dim=-1):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, pg: dist.ProcessGroup, dim=-1):
    # skip if only one rank involved
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """Forward"""
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _gather(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward"""
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _split(grad_output, ctx.mode, ctx.dim), None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        """Forward"""
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _split(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward"""
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim), None, None, None


def split_forward_gather_backward(input_, process_group, dim, grad_scale=1.0):
    """Split then gather"""
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale)


def gather_forward_split_backward(input_, process_group, dim, grad_scale=None):
    """Gather then split"""
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale)
