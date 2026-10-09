# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replicated-sharded training.

Compute replicas stay complete on every rank; the fp32 masters and the optimizer
state that follows them are distributed.

    registry.py           ownership, capabilities, fp32 masters, checkpoint manifest
    gradient_reducer.py   reduce-to-owner, plus the bucket/buffer layout both paths share
    parameter_sync.py     owner-to-replica publication, including the fp8 delta path
    checkpoint_io.py      rank-local master and optimizer-state save/load
    parameter_manager.py  wires the above, owns the plan, and is the trainer seam
    trainer_mixin.py      the finetune-loop integration

Side-stream CUDA execution and bounded look-ahead live one level up, in
``custom/async_runtime`` -- they are not specific to this subsystem.
"""

from .checkpoint_io import ReplicatedShardedCheckpointIO
from .gradient_reducer import (
    WHOLE_TENSOR,
    BufferPool,
    GradientReducer,
    GradientSyncEntry,
)
from .parameter_manager import (
    CollectiveConfig,
    ReplicatedShardedManager,
    build_shard_manager,
)
from .parameter_sync import ParameterSynchronizer, ParameterSyncEntry
from .registry import (
    MasterParameterView,
    ParameterOwnership,
    ParameterRecord,
    ParameterRegistry,
    assign_parameter_owners,
)

__all__ = [
    "WHOLE_TENSOR",
    "BufferPool",
    "CollectiveConfig",
    "GradientReducer",
    "GradientSyncEntry",
    "MasterParameterView",
    "ParameterOwnership",
    "ParameterRecord",
    "ParameterRegistry",
    "ParameterSyncEntry",
    "ParameterSynchronizer",
    "ReplicatedShardedCheckpointIO",
    "ReplicatedShardedManager",
    "assign_parameter_owners",
    "build_shard_manager",
]
