# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Owner assignment for replicated-compute optimizer-state-shard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn


@dataclass(frozen=True)
class ParameterOwnership:
    """Which rank owns a parameter's fp32 master.

    Kept as the on-the-wire ownership schema: ``state_dict`` serializes these
    fields, so renaming or reordering them would invalidate the checkpoint
    compatibility check.
    """

    name: str
    shape: tuple[int, ...]
    owner: int


def assign_parameter_owners(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    world_size: int,
):
    """Assign every full tensor to the least-loaded owner rank."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    items = list(named_parameters)
    loads = [0] * world_size
    result = []
    for name, parameter in items:
        shape = tuple(parameter.shape)
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        loads[owner] += parameter.numel() * 4
        result.append(ParameterOwnership(name, shape, owner))
    return result


class OwnershipPlanner:
    """Turn named parameters into whole-tensor ownership records.

    Only the validated greedy scheme is implemented. Reordering owners changes
    Muon's same-shape megabatch grouping (measured up to 2% gradient-norm drift),
    so an alternative balancing scheme belongs behind a new mode, never as a
    tweak here.
    """

    def __init__(self, world_size):
        """Record the target world size."""
        self._world_size = int(world_size)

    def plan(self, named_parameters):
        """Return the ownership records, in registration order."""
        return assign_parameter_owners(named_parameters, self._world_size)


__all__ = ["OwnershipPlanner", "ParameterOwnership", "assign_parameter_owners"]
