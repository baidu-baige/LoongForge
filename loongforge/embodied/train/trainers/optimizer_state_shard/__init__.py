# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""optimizer-state-shard parameter ownership and master-weight management."""
from .checkpoint_io import OptimizerStateShardCheckpointIO
from .gradient_reducer import GradientReducer, GradientSyncEntry
from .layout import WHOLE_TENSOR, BufferPool
from .optimizer_adapter import OptimizerAdapter
from .ownership import OwnershipPlanner, ParameterOwnership, assign_parameter_owners
from .parameter_manager import OptimizerStateShardManager
from .parameter_sync import ParameterSynchronizer, ParameterSyncEntry
from .registry import MasterParameterView, ParameterRecord, ParameterRegistry

__all__ = [
    "WHOLE_TENSOR",
    "BufferPool",
    "GradientReducer",
    "GradientSyncEntry",
    "MasterParameterView",
    "OptimizerAdapter",
    "OwnershipPlanner",
    "ParameterOwnership",
    "ParameterRecord",
    "ParameterRegistry",
    "ParameterSyncEntry",
    "ParameterSynchronizer",
    "OptimizerStateShardCheckpointIO",
    "OptimizerStateShardManager",
    "assign_parameter_owners",
]
