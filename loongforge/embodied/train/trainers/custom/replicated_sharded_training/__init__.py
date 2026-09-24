# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replicated-sharded training.

Compute replicas stay complete on every rank; the fp32 masters and the optimizer
state that follows them are distributed.

    registry.py           ownership, capabilities, fp32 masters, checkpoint manifest
    gradient_reducer.py   reduce-to-owner, plus the bucket/buffer layout both paths share
    parameter_sync.py     owner-to-replica publication, including the fp8 delta path
    parameter_manager.py  wires the above and owns the collective precision plan
    optimizer_runtime.py  the trainer-facing lifecycle and its builder
    checkpoint_io.py      rank-local master and optimizer-state save/load
    trainer_mixin.py      compatibility adapter into the finetune loop
    async_runtime/        side-stream CUDA execution and bounded look-ahead
"""

from .checkpoint_io import ReplicatedShardedCheckpointIO
from .gradient_reducer import (
    WHOLE_TENSOR,
    BufferPool,
    GradientReducer,
    GradientSyncEntry,
)
from .optimizer_runtime import (
    DistributedContext,
    ReplicatedShardedRuntime,
    build_optimizer_runtime,
)
from .parameter_manager import CollectiveConfig, ReplicatedShardedManager
from .parameter_sync import ParameterSynchronizer, ParameterSyncEntry
from .registry import (
    MasterParameterView,
    OwnershipPlanner,
    ParameterOwnership,
    ParameterRecord,
    ParameterRegistry,
    assign_parameter_owners,
)

__all__ = [
    "WHOLE_TENSOR",
    "BufferPool",
    "CollectiveConfig",
    "DistributedContext",
    "GradientReducer",
    "GradientSyncEntry",
    "MasterParameterView",
    "OwnershipPlanner",
    "ParameterOwnership",
    "ParameterRecord",
    "ParameterRegistry",
    "ParameterSyncEntry",
    "ParameterSynchronizer",
    "ReplicatedShardedCheckpointIO",
    "ReplicatedShardedManager",
    "ReplicatedShardedRuntime",
    "assign_parameter_owners",
    "build_optimizer_runtime",
]
