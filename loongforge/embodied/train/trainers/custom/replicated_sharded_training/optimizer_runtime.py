# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic distributed-optimizer runtime over the optimizer-state shard.

``ReplicatedShardedManager`` keeps owning the parameters, the fp32 masters and
the optimizer state. This module adds the layer above it that a trainer -- any
trainer, for any model -- can talk to:

    ReplicatedShardedRuntime   the in-tree implementation
    build_optimizer_runtime()    the single assembly entry point

The point of the seam is that a new model no longer copies the LingBot trainer's
wiring: registry, ownership plan, masters, reducer, synchronizer, optimizer
adapter and checkpoint IO are all assembled by the builder, in one place, from a
policy plus a single ``CollectiveConfig`` (precision plan + overlap knobs).

The lifecycle a trainer is expected to drive, once per step:

    setup
    -> begin_gradient_sync        (arm, before the last accumulation micro-step)
    -> forward / backward
    -> finish_gradient_sync       (drain the reduce-to-owner collectives)
    -> clip_grad_norm_
    -> begin_parameter_sync / optimizer step / finish_parameter_sync
    -> close
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from loongforge.embodied.train.trainers.custom.replicated_sharded_training.checkpoint_io import (
    ReplicatedShardedCheckpointIO,
)
from loongforge.embodied.train.trainers.custom.replicated_sharded_training.parameter_manager import (
    CollectiveConfig,
    ReplicatedShardedManager,
)
from loongforge.embodied.train.trainers.custom.replicated_sharded_training.registry import (
    MasterParameterView,
)


@dataclass(frozen=True)
class DistributedContext:
    """Minimal rank view the runtime needs.

    A trainer normally passes its own ``ctx`` (duck-typed on ``rank`` /
    ``world_size``); this dataclass exists so the runtime can be built and tested
    without a trainer or an initialised process group.
    """

    rank: int = 0
    world_size: int = 1
    group: object | None = None


class ReplicatedShardedRuntime:
    """Drive the replicated-compute optimizer-state shard through one interface.

    Delegation only: every collective and every master-weight mutation still runs
    inside ``ReplicatedShardedManager`` exactly as before, so adopting the
    runtime does not change numerics.
    """

    def __init__(self, manager: ReplicatedShardedManager | None = None):
        self.manager = manager
        self.optimizer = None
        self.checkpoint_io = None

    # -- construction ---------------------------------------------------------

    def setup(self, module, parameter_policy, context, collective_config) -> None:
        """Build the manager for ``module``; idempotent per runtime instance."""
        if self.manager is not None:
            raise RuntimeError("optimizer runtime is already set up")
        if not isinstance(context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        self.manager = ReplicatedShardedManager(
            module,
            group=context.group,
            rank=context.rank,
            world_size=context.world_size,
            parameter_policy=parameter_policy,
            collective_config=collective_config or CollectiveConfig(),
        )

    def attach_optimizer(self, optimizer):
        """Bind the optimizer over the fp32 masters and its rank-local checkpoint IO.

        The shared checkpoint backend owns the file layout and the aux files; the
        object installed here owns the masters and the per-parameter optimizer
        state, which are rank-local.
        """
        self.optimizer = optimizer
        self.checkpoint_io = ReplicatedShardedCheckpointIO(self.manager, optimizer)
        optimizer.zero_checkpoint_io = self.checkpoint_io
        return optimizer

    # -- gradient lifecycle ---------------------------------------------------

    def begin_gradient_sync(self) -> None:
        """Arm the step so per-parameter hooks can launch collectives."""
        self.manager.begin_gradient_sync()

    def finish_gradient_sync(self) -> None:
        """Drain the in-flight gradient collectives, or reduce serially if off."""
        self.manager.finish_gradient_sync()

    def clip_grad_norm_(self, max_norm: float) -> float:
        """Clip owned master gradients by global norm and return that norm."""
        return self.manager.clip_grad_norm_(max_norm)

    @torch.no_grad()
    def scrub_nan_gradients(self) -> None:
        """Replace non-finite master gradients in place."""
        for parameter in self.manager.master.values():
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    # -- optimizer lifecycle --------------------------------------------------

    def begin_parameter_sync(self) -> None:
        """Reset the publish plan so optimizer updates can start collectives."""
        self.manager.begin_parameter_sync()

    def finish_parameter_sync(self) -> None:
        """Drain the in-flight publishes, or publish serially if overlap is off."""
        self.manager.finish_parameter_sync()

    def optimizer_view(self):
        """Return a module-like view over the fp32 master parameters."""
        return self.manager.optimizer_view()

    # -- parameter access for model-side optimizer wiring ---------------------
    #
    # A model builds its own optimizer (e.g. LingBot's Muon split) and has to
    # reach the resolved parameter capabilities, the compute replicas and the
    # partial-publish hooks. Exposing them here keeps the runtime the single seam
    # so model code never has to reach into the manager or its registry.

    @property
    def parameter_policy(self):
        """Resolved parameter capability policy (answers ``optimizer_kind`` etc.)."""
        return self.manager.registry

    def compute_parameter_view(self):
        """Module-like view over the compute replicas, for optimizer param splits."""
        return MasterParameterView(self.manager.compute.items())

    def set_in_step_updated_names(self, names) -> None:
        """Record which masters the inner optimizer updates during ``step()``."""
        self.manager.set_in_step_updated_names(names)

    def on_master_updated(self, parameter) -> None:
        """Mark a master as updated so its owner-to-replica publish can start."""
        self.manager.on_master_updated(parameter)

    # -- precision (startup logging) ------------------------------------------

    @property
    def grad_reduce_mode(self) -> str:
        """Resolved gradient-reduction precision mode."""
        return self.manager.grad_reduce_mode

    @property
    def param_sync_precision(self) -> str:
        """Resolved parameter-publication precision (the single public enum)."""
        return self.manager.param_sync_precision

    def precision_summary(self):
        """Per-class resolved compute/collective precision rows, for logging."""
        return self.manager.precision_summary()

    # -- state ----------------------------------------------------------------

    def summary(self) -> dict:
        """Resolved collective plan, for the startup log."""
        return self.manager.collective_summary()

    def close(self) -> None:
        """Abandon half-issued collectives; safe to call more than once."""
        if self.manager is not None:
            self.manager.clear_pending_work()


def build_optimizer_runtime(
    module,
    parameter_policy,
    distributed_context,
    collective_config=None,
) -> ReplicatedShardedRuntime:
    """Assemble the whole optimizer stack from a policy and the collective config.

    Covers registry, ownership plan, fp32 masters, gradient reducer, parameter
    synchronizer and optimizer adapter. Checkpoint IO is bound later, by
    ``attach_optimizer``, because the optimizer is built over the masters this
    call produces.
    """
    runtime = ReplicatedShardedRuntime()
    runtime.setup(module, parameter_policy, distributed_context, collective_config)
    return runtime


__all__ = [
    "DistributedContext",
    "ReplicatedShardedRuntime",
    "build_optimizer_runtime",
]
