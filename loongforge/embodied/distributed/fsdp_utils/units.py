# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Selecting which modules become FSDP units, grouped into prefetch runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch.nn as nn

from ..utils import unwrap_checkpoint_module
from .inspection import group_numel_by_dtype, is_valid_fsdp_wrap_target

logger = logging.getLogger(__name__)


# Attribute under which ``checkpoint_wrapper`` stores the original module.
#
# FSDP must wrap the ``CheckpointWrapper``, never the module inside it:
#     FSDP(CheckpointWrapper(module))   correct
#     CheckpointWrapper(FSDP(module))   wrong
#
# The wrapper is one atomic forward whose activations are dropped and recomputed
# in backward. Sharding the inner module would make that recomputation re-enter
# FSDP hooks, duplicating all-gathers and desynchronizing parameter lifecycles.
#
# Unit selection therefore skips any module whose path contains this attribute,
# and reads class names through ``unwrap_checkpoint_module`` so that wrapping
# stays transparent to the --fsdp-*-modules class filters.
_CHECKPOINT_INNER_ATTR = "_checkpoint_wrapped_module"


@dataclass
class FSDPWrapRun:
    """Units that execute in sequence, used to derive prefetch edges.

    ``units`` keeps registration order; each unit is the list of modules that
    share one FSDP parameter group (one module in the common case).

    A run is *not* a sharding boundary — ``fully_shard`` only ever sees units.
    Runs exist solely to scope prefetch, which needs execution order, and that
    order can only be inferred inside an ordered container (``nn.ModuleList`` /
    ``nn.Sequential``). Hence grouping follows container membership:

    * One run per ordered container, i.e. roughly one run per layer stack. Its
      units are prefetch-linked to each other in registration order.
    * A module selected by class name outside any container becomes a degenerate
      single-unit run, which ``configure_prefetch`` skips — correctly, since an
      isolated module has no inferable neighbour.

    Two separate stacks never share a run: their relative execution order is
    unknowable from the module tree, and linking across them would prefetch the
    wrong parameters. Even within a run the order is *registration* order, which
    matches execution only for models that call their layers sequentially.

    ``depth`` is the container's nesting depth, used to order the wrapping pass
    deepest-first so inner units claim their parameters before outer ones.
    """

    depth: int
    units: list[list[nn.Module]]


def resolve_wrap_runs(model: nn.Module, training_args, ignored_params: set) -> list[FSDPWrapRun]:
    """Resolve the FSDP units to create, grouped into execution runs.

    Two passes with different jobs: containers first, because only an ordered
    container reveals execution order and therefore permits prefetch links, then
    class-name matches that live outside any container. Anything Pass 1 claimed
    is excluded from Pass 2 by object identity — registering a module in two
    units would make ``fully_shard`` claim its parameters twice.

    Args:
        model: Raw model, already on device and dtype-cast, with activation
            checkpointing applied but not yet sharded. Paths and class names are
            read from the tree as it exists now.
        training_args: Supplies ``fsdp_wrap_modules`` (explicit unit classes),
            ``fsdp_no_wrap_modules`` (never a unit) and ``fsdp_min_param_num``
            (unit-stacking threshold, only consulted in automatic mode).
        ignored_params: Parameters FSDP will not manage, from
            ``build_ignored_params``. A module whose parameters are all ignored
            counts as empty and never becomes a unit.

    Returns:
        Runs in discovery order — containers first, then isolated matches. The
        list is deliberately **not** sorted by ``depth``: the caller must sort
        deepest-first when wrapping (see ``parallel._wrap_fsdp``) so inner units
        claim their parameters before outer ones, while ``configure_prefetch``
        needs this original order to derive prefetch edges.

    Raises:
        ValueError: If a class name appears in both ``fsdp_wrap_modules`` and
            ``fsdp_no_wrap_modules`` (the intent is unknowable), or if an
            explicitly named class cannot host a group (see
            :func:`resolve_wrap_modules`).

    Note:
        Read-only with respect to ``model``: nothing is wrapped here, the
        returned modules are live references that ``fully_shard_unit`` mutates
        later. Selection must be identical on every rank, since diverging unit
        sets desynchronize the all-gather sequence and hang instead of failing.

        Explicit and automatic mode differ in more than the class filter:
        explicit mode keeps a 1:1 module-to-unit mapping (``min_param_num`` is
        not applied) and is strict about unmatched names, whereas automatic mode
        falls back to ``model._no_split_modules`` and silently tolerates classes
        that match nothing.

    Examples:
        Automatic mode on a stack of 32 blocks, stacking blocks until each unit
        holds at least 1e8 parameters::

            >>> runs = resolve_wrap_runs(model, training_args, ignored)
            >>> [len(run.units) for run in runs]
            [8]

        Explicit mode gives one unit per matched module, so a stack of 32 blocks
        yields 32 single-module units in one run::

            >>> training_args.fsdp_wrap_modules = ["Qwen3DecoderLayer"]
            >>> runs = resolve_wrap_runs(model, training_args, ignored)
            >>> [len(unit) for run in runs for unit in run.units][:3]
            [1, 1, 1]
    """
    no_wrap = set(training_args.fsdp_no_wrap_modules or [])
    explicit = set(training_args.fsdp_wrap_modules or [])
    if explicit:
        conflict = explicit & no_wrap
        if conflict:
            raise ValueError(
                "Module classes appear in both --fsdp-wrap-modules and "
                f"--fsdp-no-wrap-modules: {', '.join(sorted(conflict))}."
            )

    # Pass 1: children of ordered containers. Explicit class names keep a 1:1
    # module-to-unit mapping; automatic selection stacks small layers.
    runs = resolve_container_runs(
        model,
        ignored_params,
        keep_class_names=explicit or None,
        skip_class_names=no_wrap,
        min_param_num=None if explicit else training_args.fsdp_min_param_num,
    )
    # Flatten run -> unit -> module into an identity set of everything Pass 1
    # already claimed, so Pass 2 cannot register a second, conflicting unit for
    # the same module.
    covered = {id(m) for run in runs for unit in run.units for m in unit}

    # Pass 2: class names that do not live under an ordered container. Explicit
    # names win outright; only the automatic set needs no_wrap subtracted, since
    # an explicit/no_wrap conflict was already rejected above.
    class_names = explicit or (set(getattr(model, "_no_split_modules", None) or ()) - no_wrap)
    if class_names:
        named = resolve_wrap_modules(model, class_names, strict=bool(explicit))
        for path, module in named:
            if id(module) in covered:
                continue
            # A module selected for both wrapping and ignoring can have no
            # managed parameters left. Do not create an empty FSDP boundary;
            # it would add hooks and state without owning a communication group.
            if not group_numel_by_dtype(module, ignored_params):
                logger.debug(
                    "Skipping empty FSDP unit %s (%s): all parameters are ignored",
                    path,
                    unwrap_checkpoint_module(module).__class__.__name__,
                )
                continue
            # Degenerate run: one unit, no siblings to infer order against, so
            # configure_prefetch skips it.
            runs.append(FSDPWrapRun(depth=path.count("."), units=[[module]]))

    logger.info(
        "FSDP units: %d in %d runs (sizes=%s)",
        sum(len(run.units) for run in runs), len(runs),
        [len(unit) for run in runs for unit in run.units],
    )
    return runs


def resolve_container_runs(
    model: nn.Module,
    ignored_params: set,
    keep_class_names: set[str] | None,
    skip_class_names: set[str],
    min_param_num: int | None,
) -> list[FSDPWrapRun]:
    """Build one run per ordered container from its eligible children.

    Consecutive children of the same class accumulate into ``current_unit``.
    When ``min_param_num`` is set, ``current_unit`` is closed once its parameter
    count reaches the threshold so that small layers are stacked into a single
    FSDP unit; otherwise every child becomes its own unit.

    Stacking is bounded by class identity as well as size: a unit spanning two
    different classes would tie their parameter lifecycles together even though
    the model may not execute them adjacently. An ineligible child also closes
    the open unit, because the modules on either side of it are no longer
    consecutive in execution.

    Args:
        model: Model to scan. Only ``nn.ModuleList`` / ``nn.Sequential`` nodes
            become runs; other containers (e.g. ``nn.ModuleDict``) expose no
            order and are ignored here. Subtrees under a checkpoint wrapper's
            inner module are skipped so the wrapper stays the FSDP boundary.
        ignored_params: Parameters FSDP will not manage; children whose managed
            numel is zero are ineligible.
        keep_class_names: Whitelist of class names allowed to become units.
            ``None`` means "no whitelist" — every eligible child qualifies,
            which is the automatic mode. Names are compared after unwrapping the
            checkpoint wrapper, so wrapping does not change what matches.
        skip_class_names: Class names that must never host a unit. Their
            parameters are still sharded by the closest enclosing group.
        min_param_num: Parameter-count threshold at which an open unit is
            closed. ``None`` disables stacking entirely (one unit per child)
            rather than meaning "no minimum".

    Returns:
        One :class:`FSDPWrapRun` per container that yielded at least one unit,
        in ``named_modules()`` traversal order. ``depth`` is the depth of the
        container's children, i.e. the level the units actually sit at.

    Note:
        Unit membership is decided by registration order, not by observed
        execution order; for models whose ``forward`` reorders or conditionally
        skips container children, the resulting prefetch edges are wrong even
        though sharding stays correct.

    Examples:
        Automatic mode — stack same-class blocks up to the threshold::

            >>> runs = resolve_container_runs(
            ...     model, ignored,
            ...     keep_class_names=None,
            ...     skip_class_names=set(),
            ...     min_param_num=100_000_000,
            ... )

        Explicit mode — one unit per matched child, no stacking::

            >>> runs = resolve_container_runs(
            ...     model, ignored,
            ...     keep_class_names={"Qwen3DecoderLayer"},
            ...     skip_class_names=set(),
            ...     min_param_num=None,
            ... )
    """
    runs: list[FSDPWrapRun] = []
    for path, container in model.named_modules():
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            continue
        if _CHECKPOINT_INNER_ATTR in path.split("."):
            continue

        units: list[list[nn.Module]] = []
        current_unit: list[nn.Module] = []
        current_unit_numel = 0
        current_unit_cls: type | None = None
        current_unit_dtype = None

        for child in container.children():
            cls = type(unwrap_checkpoint_module(child))
            numel_by_dtype = group_numel_by_dtype(child, ignored_params)
            numel = sum(numel_by_dtype.values())
            dominant_dtype = (
                max(numel_by_dtype, key=numel_by_dtype.get)
                if numel_by_dtype else None
            )
            eligible = (
                numel > 0
                and is_valid_fsdp_wrap_target(child)
                and cls.__name__ not in skip_class_names
                and (keep_class_names is None or cls.__name__ in keep_class_names)
            )
            starts_new_unit = current_unit and (
                cls is not current_unit_cls or dominant_dtype != current_unit_dtype
            )
            if not eligible or starts_new_unit:
                if current_unit:
                    units.append(current_unit)
                current_unit, current_unit_numel, current_unit_cls = [], 0, None
                current_unit_dtype = None
            if not eligible:
                continue

            current_unit.append(child)
            current_unit_numel += numel
            current_unit_cls = cls
            current_unit_dtype = dominant_dtype
            if min_param_num is None or current_unit_numel >= min_param_num:
                units.append(current_unit)
                current_unit, current_unit_numel, current_unit_cls = [], 0, None
                current_unit_dtype = None

        if current_unit:
            units.append(current_unit)
        if units:
            # One run per container: these units are prefetch-linked to each
            # other, but never to units from another container.
            runs.append(FSDPWrapRun(depth=path.count(".") + 1, units=units))

    return runs


def resolve_wrap_modules(
    model: nn.Module,
    module_class_names: set[str],
    strict: bool = True,
) -> list[tuple[str, nn.Module]]:
    """Find modules matching class names; return deepest-first for bottom-up wrap.

    Args:
        model: Model to scan. The root (empty path) is never a candidate, and
            modules inside a checkpoint wrapper are skipped so that FSDP wraps
            the wrapper rather than its inner module.
        module_class_names: Class names to match, compared after unwrapping the
            checkpoint wrapper.
        strict: Whether a name matching nothing is an error. Explicit
            ``--fsdp-wrap-modules`` uses ``True`` so a typo fails the job
            instead of silently degrading to root-only sharding; the
            ``_no_split_modules`` fallback uses ``False`` because that list is a
            model-provided hint that legitimately lists classes this
            configuration does not contain.

    Returns:
        ``(path, module)`` pairs sorted by path depth descending, so callers can
        wrap bottom-up. Order among equal depths is arbitrary but stable for a
        given model; parameters are claimed by the innermost group either way.

    Raises:
        ValueError: If a matched module cannot host a group (a pure container or
            something without a callable ``forward``) — this is always a
            configuration error, since the class name was named on purpose. Also
            raised when ``strict`` and some name matched nothing.

    Note:
        Matching is by class name, not by identity or path, so every instance of
        the class becomes a unit. A class that appears both inside and outside an
        ordered container will therefore be found here too; ``resolve_wrap_runs``
        is what filters out the instances already claimed by a container run.
    """
    targets: dict[str, nn.Module] = {}
    for path, module in model.named_modules():
        if not path:
            continue
        # The checkpoint wrapper, not its inner module, is the FSDP boundary.
        if _CHECKPOINT_INNER_ATTR in path.split("."):
            continue
        class_name = unwrap_checkpoint_module(module).__class__.__name__
        if class_name not in module_class_names:
            continue
        if not is_valid_fsdp_wrap_target(module):
            raise ValueError(
                f"Cannot use {path!r} (class {class_name!r}) as an FSDP wrap unit: "
                f"a unit must have a callable forward() and must not be a plain "
                f"container. Got type {type(module).__name__!r}."
            )
        targets[path] = module

    matched = {unwrap_checkpoint_module(m).__class__.__name__ for m in targets.values()}
    missing = module_class_names - matched
    if missing and strict:
        raise ValueError(
            "FSDP wrap module classes matched no callable boundaries: "
            + ", ".join(sorted(missing))
        )

    return sorted(targets.items(), key=lambda item: item[0].count("."), reverse=True)


def find_fsdp_root_module(model: nn.Module) -> nn.Module | None:
    """Find the outermost FSDP2 module under an application wrapper."""
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return None

    fsdp_ids = {
        id(module)
        for module in model.modules()
        if isinstance(module, FSDPModule)
    }
    if not fsdp_ids:
        return None

    parents: dict[int, nn.Module] = {}
    for parent in model.modules():
        for child in parent.children():
            parents[id(child)] = parent

    for module in model.modules():
        if id(module) not in fsdp_ids:
            continue
        parent = parents.get(id(module))
        while parent is not None:
            if id(parent) in fsdp_ids:
                break
            parent = parents.get(id(parent))
        else:
            return module
    return None


def get_fsdp_root_sharded_params(root: nn.Module) -> list[nn.Parameter]:
    """Return only the parameters owned by the root FSDP param group."""
    try:
        state = root._get_fsdp_state()
    except (AttributeError, RuntimeError):
        return []
    params = []
    seen = set()
    for group in getattr(state, "_fsdp_param_groups", ()):
        for fsdp_param in getattr(group, "fsdp_params", ()):
            param = getattr(fsdp_param, "sharded_param", None)
            if param is not None and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params