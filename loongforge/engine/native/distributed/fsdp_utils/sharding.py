# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Creating a single FSDP group, including its per-group reshard policy."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from ..utils import unwrap_checkpoint_module
from .context import FSDPWrapContext
from .inspection import group_numel_by_dtype
from .mixed_dtype import isolate_minority_dtypes


def fully_shard_unit(unit: list[nn.Module], wrap_ctx: FSDPWrapContext) -> None:
    """Make the unit dtype-uniform, then create one FSDP group for it.

    Args:
        unit: Modules that should share one parameter group, in execution order;
            ``unit[0]`` is the representative whose class decides the reshard
            policy and which later owns the FSDP state. Usually one module.
        wrap_ctx: Shared per-pass configuration. Its ``fsdp_kwargs`` is copied
            before the per-group reshard value is added, so the shared dict is
            never mutated.

    Raises:
        ValueError: If the unit still holds more than one parameter dtype after
            minority-dtype isolation, which means those parameters are owned
            directly by the unit's modules and no subtree boundary can separate
            them.

    Note:
        Mutates the modules in place and returns ``None``: ``fully_shard``
        swaps in an ``FSDPModule`` class and installs hooks, and isolation may
        create additional nested groups below this unit as a side effect. Call
        deepest-first — a parameter is claimed by the first group that wraps it,
        so wrapping an ancestor first would leave inner units with nothing.

        Collective in effect: all ranks must build the same groups in the same
        order, since group creation fixes the all-gather sequence.
    """
    # Isolating a minority dtype means creating groups for the subtrees being
    # split off, which is this same function one level down.
    shard_unit = partial(fully_shard_unit, wrap_ctx=wrap_ctx)

    unit_numel_by_dtype: dict[torch.dtype, int] = {}
    for module in unit:
        isolate_minority_dtypes(module, wrap_ctx, shard_unit)
        for dtype, numel in group_numel_by_dtype(module, wrap_ctx.ignored_params).items():
            unit_numel_by_dtype[dtype] = unit_numel_by_dtype.get(dtype, 0) + numel

    if len(unit_numel_by_dtype) > 1:
        raise ValueError(
            f"FSDP cannot create a group for "
            f"{unwrap_checkpoint_module(unit[0]).__class__.__name__} with mixed "
            f"parameter dtypes {sorted(str(d) for d in unit_numel_by_dtype)}: the "
            f"minority dtype parameters are owned directly by these modules, so they "
            f"cannot be split into a separate group. Cast them to a single dtype "
            f"or name a finer wrap unit via --fsdp-wrap-modules."
        )

    kwargs = dict(wrap_ctx.fsdp_kwargs)
    reshard = resolve_reshard_policy(unit[0], wrap_ctx)
    if reshard is not None:
        kwargs["reshard_after_forward"] = reshard
    fully_shard(unit, **kwargs)


def resolve_reshard_policy(module: nn.Module, wrap_ctx: FSDPWrapContext):
    """Determine the reshard_after_forward value for a module.

    Priority: per-class override > default. Returns None if neither is set, which
    leaves FSDP2's own default (reshard after forward) in place.

    The root group needs no special case: FSDP2 clears its ``post_forward_mesh_info``
    during lazy init, so whatever is passed here is discarded and the root always
    keeps its parameters unsharded after forward.

    Args:
        module: Unit representative. Its class name after unwrapping the
            checkpoint wrapper is the override key, so overrides are written in
            terms of model classes and stay valid whether or not activation
            checkpointing is enabled.
        wrap_ctx: Holds ``reshard_overrides`` (per class) and
            ``reshard_default``.

    Returns:
        The value for ``reshard_after_forward``: ``True`` reshards immediately
        after forward (lowest memory, re-all-gathers in backward), ``False``
        keeps parameters unsharded until backward finishes (fastest, highest
        memory), and an ``int`` reshards to that many ranks as a middle ground.
        ``None`` means "pass nothing" so FSDP2's own default applies.
    """
    cls_name = unwrap_checkpoint_module(module).__class__.__name__
    if cls_name in wrap_ctx.reshard_overrides:
        return wrap_ctx.reshard_overrides[cls_name]
    return wrap_ctx.reshard_default
