# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Splitting minority-dtype subtrees into their own FSDP groups.

FSDP2 requires a uniform original parameter dtype per group, which these passes
enforce by moving the offending subtrees into separate groups before the caller
creates its own.

Both entry points take a ``shard_unit`` callback instead of importing
``sharding`` directly: the recursion is mutual (creating a group for a subtree
re-enters this module for that subtree), and passing the callback keeps the
dependency one-way.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from ..utils import unwrap_checkpoint_module
from .context import FSDPWrapContext
from .inspection import group_numel_by_dtype, is_valid_fsdp_wrap_target

# Creates one FSDP group out of the given modules.
ShardUnitFn = Callable[[list[nn.Module]], None]


def isolate_minority_dtypes(
    module: nn.Module,
    wrap_ctx: FSDPWrapContext,
    shard_unit: ShardUnitFn,
) -> None:
    """Recursively split minority-dtype descendants into their own FSDP groups.

    The dominant dtype (by numel) stays in ``module``'s group; subtrees that do
    not hold the dominant dtype are wrapped as separate groups.

    Dominance is measured by parameter count rather than by module count so that
    the expensive majority keeps the caller's group and only the small outliers
    (typically an fp32 norm or head inside a bf16 stack) pay for extra groups.

    Args:
        module: Subtree root, not yet sharded. Descendants that are already
            ``FSDPModule`` are left alone: their parameters belong to an inner
            group and no longer affect this one.
        wrap_ctx: Supplies ``ignored_params`` and the no-wrap class filter.
        shard_unit: Callback creating one group from a list of modules, injected
            to keep the recursion between this module and ``sharding`` one-way.

    Note:
        Mutates the tree in place and returns ``None``: an unknown number of
        nested groups may exist below ``module`` afterwards. Uniform subtrees
        return immediately, so the common single-dtype model pays only one
        ``group_numel_by_dtype`` walk.

        Best-effort by design — it cannot fix a module that owns mixed-dtype
        parameters directly, since a group boundary can only be placed at a
        module. The caller re-checks and raises in that case.
    """
    numel_by_dtype = group_numel_by_dtype(module, wrap_ctx.ignored_params)
    if len(numel_by_dtype) <= 1:
        return
    dominant_dtype = max(numel_by_dtype, key=numel_by_dtype.get)

    for child in module.children():
        if isinstance(child, FSDPModule):
            continue
        child_numel_by_dtype = group_numel_by_dtype(child, wrap_ctx.ignored_params)
        if not child_numel_by_dtype:
            continue
        if dominant_dtype not in child_numel_by_dtype:
            shard_boundary_or_children(child, wrap_ctx, shard_unit)
        elif len(child_numel_by_dtype) > 1:
            isolate_minority_dtypes(child, wrap_ctx, shard_unit)
            # Still mixed: the leftover minority params are owned by ``child``
            # itself, not by a subtree, so the only way out is to move all of
            # ``child`` into its own group.
            if len(group_numel_by_dtype(child, wrap_ctx.ignored_params)) > 1:
                shard_boundary_or_children(child, wrap_ctx, shard_unit)


def shard_boundary_or_children(
    module: nn.Module,
    wrap_ctx: FSDPWrapContext,
    shard_unit: ShardUnitFn,
) -> None:
    """Wrap a valid boundary, or descend when the module cannot be a unit.

    Callers have already decided that this subtree *must* leave the enclosing
    group, so giving up is not an option: a pure container or a
    ``--fsdp-no-wrap-modules`` class cannot host the group itself, and the
    minority-dtype params must instead be claimed by groups further down. If no
    boundary is found anywhere below, the caller reports the mixed dtypes.

    Args:
        module: Subtree that must leave the enclosing group. Becomes a group
            itself when it is a valid wrap target and its class is not in
            ``--fsdp-no-wrap-modules``; otherwise every child holding managed
            parameters is tried in turn.
        wrap_ctx: Supplies ``ignored_params`` and ``no_wrap_classes``.
        shard_unit: Callback creating one group from a list of modules.

    Note:
        Mutates the tree in place and returns ``None``, and gives no signal about
        whether it succeeded — descending can end without creating any group
        (e.g. a no-wrap class whose children are all containers). The caller
        detects that by re-measuring dtypes, so this function stays free to be
        best-effort.
    """
    cls_name = unwrap_checkpoint_module(module).__class__.__name__
    if is_valid_fsdp_wrap_target(module) and cls_name not in wrap_ctx.no_wrap_classes:
        shard_unit([module])
        return

    for child in module.children():
        if isinstance(child, FSDPModule):
            continue
        # Skip subtrees with nothing left to manage: wrapping them would install
        # hooks and swap classes for an empty parameter group.
        if group_numel_by_dtype(child, wrap_ctx.ignored_params):
            shard_boundary_or_children(child, wrap_ctx, shard_unit)
