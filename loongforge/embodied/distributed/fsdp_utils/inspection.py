# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read-only queries that decide whether and how a module can host an FSDP group.

Nothing here mutates the model; the wrapping passes in ``units``, ``sharding``
and ``mixed_dtype`` rely on these predicates to make their decisions.

``group_numel_by_dtype`` is the single definition of "which parameters would join
this group", so unit selection and mixed-dtype handling cannot drift apart on
whether ignored or already-claimed parameters count.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule


def is_container_module(module: nn.Module) -> bool:
    """Return True for modules that are traversal containers without forward.

    Args:
        module: Any module; checked by type, not by whether it happens to define
            ``forward``.

    Returns:
        Whether the module is one of the stdlib containers whose only job is to
        hold children. ``fully_shard`` rejects these, so they can never host a
        group even though their subtrees hold parameters.

    Note:
        Type-based rather than duck-typed on purpose: ``nn.Sequential`` does have
        a callable ``forward``, so a capability check would let it through and the
        rejection would surface later inside ``fully_shard``. A user-defined
        container that subclasses ``nn.Module`` directly is not recognised here
        and is treated as a normal module.
    """
    return isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential))


def is_valid_fsdp_wrap_target(module: nn.Module) -> bool:
    """Return whether a module is a valid FSDP wrapping boundary.

    A valid boundary must:
    - Not be a pure container (``fully_shard`` rejects ModuleList/ModuleDict)
    - Override ``nn.Module.forward`` (i.e., participate in forward execution)

    Args:
        module: Candidate for hosting an FSDP group. Pass the checkpoint wrapper
            itself, not its inner module — the wrapper is the boundary FSDP must
            see, and it satisfies both conditions.

    Returns:
        Whether ``fully_shard`` can be applied to this module. Says nothing about
        whether it *should* be: emptiness (``managed_param_numel``), class
        filters and no-wrap lists are separate decisions made by the callers.

    Note:
        The class-level override check excludes modules that only carry parameters
        and inherit the unimplemented ``nn.Module.forward``. It cannot detect a
        module that overrides ``forward`` but is never actually called — such a
        module passes here, gets its own group, and simply never all-gathers,
        which shows up as unused parameters rather than an error.
    """
    if is_container_module(module):
        return False

    return module.__class__.forward is not nn.Module.forward


def group_numel_by_dtype(
    module: nn.Module,
    ignored_params: set,
) -> dict[torch.dtype, int]:
    """Parameter numel per dtype for the params that would join ``module``'s group.

    FSDP2 requires one original parameter dtype per group, so callers use this
    both to detect a mixed-dtype group and to pick the dominant dtype by weight.

    The walk is hand-rolled instead of ``module.modules()`` because it must prune
    whole subtrees, which a recursive generator cannot express: parameters below
    a nested ``FSDPModule`` already belong to an inner group, as do
    ``ignored_params``. Tied/shared parameters are counted once so that the numel
    weighting is not skewed by the number of paths reaching them.

    Args:
        module: Subtree root. Counting starts at its own direct parameters, so
            calling this on a child and on its parent legitimately gives
            different totals.
        ignored_params: Parameters FSDP will not manage, from
            ``build_ignored_params``. Matched by object identity, not by name or
            value, so it must be the same objects the model holds — a set of
            clones or of ``.data`` tensors would match nothing and silently
            inflate every count.

    Returns:
        Plain dict mapping dtype to total numel, containing only dtypes actually
        present. Empty when the subtree contributes nothing — every parameter is
        ignored or already claimed by an inner group — which callers read as "not
        worth a group". Exactly one entry means the subtree is dtype-uniform and
        can host a group as-is.

    Note:
        Reflects the tree *at call time*: because a nested ``FSDPModule`` prunes
        the walk, the result for the same module changes once inner groups have
        been created. This is what lets ``fully_shard_unit`` re-measure after
        isolation instead of tracking state, but it also means results must not be
        cached across wrapping steps.

        Counts parameters only. Buffers are excluded because FSDP does not shard
        them, so a subtree holding a large fp32 buffer still counts as empty.
    """
    ignored_param_ids = {id(p) for p in ignored_params}
    numel_by_dtype: dict[torch.dtype, int] = defaultdict(int)
    counted_param_ids: set[int] = set()

    modules_to_visit = [module]
    while modules_to_visit:
        current_module = modules_to_visit.pop()
        for param in current_module.parameters(recurse=False):
            param_id = id(param)
            if param_id in counted_param_ids or param_id in ignored_param_ids:
                continue
            counted_param_ids.add(param_id)
            numel_by_dtype[param.dtype] += param.numel()
        for child in current_module.children():
            if isinstance(child, FSDPModule):
                continue
            modules_to_visit.append(child)

    return dict(numel_by_dtype)


def managed_param_numel(module: nn.Module, ignored_params: set) -> int:
    """Total numel of the parameters FSDP would manage for ``module``.

    Used to size wrap units, so it must agree with ``group_numel_by_dtype`` about
    what a group actually owns — a module holding only ignored parameters counts
    as zero and is not worth a group of its own.

    Args:
        module: Subtree root, same semantics as ``group_numel_by_dtype``.
        ignored_params: Parameters FSDP will not manage, matched by identity.

    Returns:
        Element count summed across dtypes — element *count*, not bytes, so it is
        not a memory estimate and comparing it against a threshold treats fp32 and
        bf16 parameters as equally expensive. That is intentional for unit sizing:
        ``--fsdp-min-param-num`` is about how many parameters a group manages, and
        the same value keeps behaving the same way when a model's dtype changes.
        ``0`` means the subtree contributes nothing to shard.
    """
    return sum(group_numel_by_dtype(module, ignored_params).values())
