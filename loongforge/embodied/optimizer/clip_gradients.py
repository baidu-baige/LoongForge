# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Gradient clipping and NaN cleaning."""

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def _local_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Return the local tensor backing an FSDP2 DTensor gradient."""
    if isinstance(gradient, DTensor):
        return gradient._local_tensor
    return gradient


def _local_gradient_groups(model: nn.Module) -> list[list[torch.Tensor]]:
    """Group mutable local gradients by device and dtype."""
    # The global L2 norm is the sum of each gradient's squared norm, and one
    # clip coefficient scales every gradient. Compatible tensors can therefore
    # share foreach kernels without changing the result; grouping by device and
    # dtype satisfies foreach constraints while reducing per-tensor launches.
    groups = defaultdict(list)
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient = _local_gradient(parameter.grad)
            groups[(gradient.device, gradient.dtype)].append(gradient)
    return list(groups.values())


def _local_norm_sq(
    gradient_groups: list[list[torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    """Compute the local squared L2 norm in float32."""
    total_norm_sq = torch.zeros((), device=device)
    for gradients in gradient_groups:
        norms = torch._foreach_norm(gradients, 2.0, dtype=torch.float32)
        total_norm_sq += torch.stack(norms).pow(2).sum()
    return total_norm_sq


def get_grad_norm(model: nn.Module) -> float:
    """Compute global gradient norm, accounting for FSDP sharding.

    For FSDP models, gradients are sharded across ranks. Each rank computes
    its local norm squared, then all-reduce sums them to get the global norm.
    For non-FSDP models, computes the norm directly.

    Args:
        model: The model whose gradients to analyze. Can be a vanilla PyTorch
            module, FSDP-wrapped module, or module with FSDP sub-modules.

    Returns:
        The L2 norm of all model gradients (global norm for distributed).
    """

    is_fsdp = isinstance(model, FSDPModule)
    gradient_groups = _local_gradient_groups(model)
    total_norm_sq = _local_norm_sq(
        gradient_groups,
        next(model.parameters()).device,
    )

    if is_fsdp and dist.is_initialized():
        dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
    return total_norm_sq.sqrt().item()


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Gradient clipping for FSDP with mixed-dtype gradients (fp32 + bf16).

    FSDP shards parameters across ranks, so each rank only holds a shard of
    gradients. We compute local norm in float32, all-reduce to get global norm,
    then clip.

    Returns:
        The global gradient L2 norm computed *before* clipping. Reuse this for
        logging instead of recomputing the (post-clip) norm separately.
    """

    is_fsdp = isinstance(model, FSDPModule)
    if not is_fsdp:
        # clip_grad_norm_ returns the total norm *before* clipping. Under DDP
        # grads are already all-reduced, so the local norm equals the global one.
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return float(total_norm)

    # Compute local sharded norm in float32 (handles mixed dtype)
    gradient_groups = _local_gradient_groups(model)
    local_norm_sq = _local_norm_sq(
        gradient_groups,
        next(model.parameters()).device,
    )

    # All-reduce to get global norm across all ranks
    if dist.is_initialized():
        torch.distributed.all_reduce(local_norm_sq)
    total_norm = local_norm_sq.sqrt()

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    for gradients in gradient_groups:
        torch._foreach_mul_(gradients, clip_coef)

    return total_norm.item()


def clean_nan_gradients(model: nn.Module):
    """Replace NaN/Inf gradients with 0."""
    for param in model.parameters():
        if param.grad is not None:
            grad = _local_gradient(param.grad)
            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0, out=grad)
