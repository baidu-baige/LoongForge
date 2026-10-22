# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 prefetch edges derived from the resolved wrap runs."""

from __future__ import annotations

import logging

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from .units import FSDPWrapRun

logger = logging.getLogger(__name__)


def configure_prefetch(runs: list[FSDPWrapRun], training_args) -> None:
    """Configure FSDP2 prefetch edges along each run of units.

    Only the first module of a unit owns the FSDP state, so edges are set between
    unit representatives, in the order the units were resolved. See
    ``FSDPWrapRun`` for why runs bound this and what the order does not guarantee.

    Args:
        runs: Runs from ``resolve_wrap_runs``, in their original resolution order
            — not the depth-sorted order used for wrapping, since prefetch needs
            execution order rather than nesting order. Must be called after
            ``fully_shard``, otherwise the representatives are not yet
            ``FSDPModule`` and every edge is silently dropped.
        training_args: Supplies ``fsdp_forward_prefetch_distance`` and
            ``fsdp_backward_prefetch_distance``.

    Note:
        Mutates the modules in place and returns ``None``. Both distances
        non-positive is a full no-op — the default CLI flags of ``0`` mean
        "prefetch off", so the modules are not even scanned. Single-unit runs
        are also skipped: an isolated module has no sibling whose execution
        order can be inferred, so any edge would be a guess.

        Prefetching trades memory for overlap — a distance of ``n`` keeps up to
        ``n`` extra unsharded parameter buffers alive, so raising it on a
        memory-tight job converts a throughput win into an OOM. Wrong edges do not
        corrupt results (FSDP still all-gathers on demand) but do cost bandwidth
        and peak memory for parameters that are not used next.
    """
    forward_distance = training_args.fsdp_forward_prefetch_distance
    backward_distance = training_args.fsdp_backward_prefetch_distance
    if forward_distance <= 0 and backward_distance <= 0:
        return

    configured = 0
    for run in runs:
        if len(run.units) <= 1:
            continue
        set_prefetch(
            [unit[0] for unit in run.units],
            forward_distance,
            backward_distance,
        )
        configured += 1

    if configured:
        logger.info("FSDP configured %d prefetch runs.", configured)


def set_prefetch(
    fsdp_modules: list[nn.Module],
    forward_distance: int,
    backward_distance: int,
) -> None:
    """Configure FSDP2 prefetch edges from an ordered module sequence.

    Args:
        fsdp_modules: Unit representatives in execution order. Entries that are
            not ``FSDPModule`` are dropped rather than rejected, because a unit
            can legitimately end up unsharded (all its parameters ignored or
            already claimed by an inner group); the remaining ones keep their
            relative order, so edges skip over the dropped modules instead of
            being misaligned.
        forward_distance: How many following units to prefetch during forward.
        backward_distance: How many preceding units to prefetch during backward
            — backward runs in reverse, so "preceding" is what comes next in
            time.

    Note:
        Mutates the modules and returns ``None``. Both distances non-positive,
        or fewer than two surviving modules, is a no-op. A single non-positive
        distance disables that direction only.
    """
    if forward_distance <= 0 and backward_distance <= 0:
        return
    fsdp_modules = [m for m in fsdp_modules if isinstance(m, FSDPModule)]
    if len(fsdp_modules) <= 1:
        return

    fwd = set_prefetch_edges(fsdp_modules, forward_distance, forward=True)
    bwd = set_prefetch_edges(fsdp_modules, backward_distance, forward=False)
    logger.info(
        "FSDP2 prefetch: modules=%d fwd_dist=%d bwd_dist=%d fwd_edges=%d bwd_edges=%d",
        len(fsdp_modules), forward_distance, backward_distance, fwd, bwd,
    )


def set_prefetch_edges(fsdp_modules: list[nn.Module], distance: int, *, forward: bool) -> int:
    """Configure one direction of prefetch; return edge count.

    Args:
        fsdp_modules: Already filtered to ``FSDPModule``, in execution order.
        distance: Number of neighbours each module prefetches. Values ``<= 0``
            disable this direction and produce no edges, which is how the
            ``0``-default CLI flags mean "off".
        forward: Selects the direction. Forward links each module to the ``n``
            modules after it; backward links to the ``n`` before it, reversed so
            the nearest neighbour is prefetched first — FSDP2 treats the list as
            priority-ordered, and backward reaches the closest predecessor
            soonest.

    Returns:
        Total number of edges installed, for logging only.

    Note:
        Mutates each module via ``set_modules_to_*_prefetch``, replacing any
        previously configured list rather than appending, so calling this twice
        for the same direction discards the earlier configuration. Edges near the
        sequence ends are naturally shorter than ``distance``; slicing clamps
        instead of wrapping, because a wrap-around edge would prefetch the first
        unit while the last one executes.
    """
    if distance <= 0:
        return 0
    edges = 0
    for i, module in enumerate(fsdp_modules):
        if forward:
            targets = fsdp_modules[i + 1 : i + 1 + distance]
            setter = module.set_modules_to_forward_prefetch
        else:
            start = max(0, i - distance)
            targets = list(reversed(fsdp_modules[start:i]))
            setter = module.set_modules_to_backward_prefetch
        if targets:
            setter(targets)
            edges += len(targets)
    return edges
