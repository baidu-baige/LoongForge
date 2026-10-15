# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Custom data parallel (custom_dp) training.

Compute replicas stay complete on every rank and no tensor is split: each
parameter is assigned to one owner, gradients are reduced to that owner, and the
owner publishes the updated parameter back. The fp32 masters and the optimizer
state that follows them live only on the owner.

    parameter_plan.py     ownership, capabilities, fp32 masters, checkpoint manifest
    gradient_reducer.py   reduce-to-owner, plus the bucket/buffer layout both paths share
    parameter_sync.py     owner-to-replica publication, including the fp8 delta path
    checkpoint_io.py      rank-local master and optimizer-state save/load
    parameter_manager.py  wires the above, owns the plan, and is the trainer seam
    optimizer_builder.py  Muon plus its AdamW fallback, over the fp32 masters
    custom_dp_trainer.py  the finetune-loop integration

Side-stream CUDA execution and bounded look-ahead live one level up, in
``custom/async_runtime`` -- they are not specific to this subsystem.
"""

from .checkpoint_io import RankLocalCheckpointIO
from .gradient_reducer import (
    WHOLE_TENSOR,
    BufferPool,
    GradientReduceBucket,
    GradientReducer,
)
from .parameter_manager import (
    CollectiveConfig,
    ParameterManager,
    build_parameter_manager,
)
from .parameter_plan import (
    MasterParameterView,
    ParameterOwnership,
    ParameterPlan,
    ParameterRecord,
    assign_parameter_owners,
)
from .parameter_sync import ParameterBroadcastBucket, ParameterSynchronizer

__all__ = [
    "WHOLE_TENSOR",
    "BufferPool",
    "CollectiveConfig",
    "GradientReduceBucket",
    "GradientReducer",
    "MasterParameterView",
    "ParameterBroadcastBucket",
    "ParameterOwnership",
    "ParameterPlan",
    "ParameterRecord",
    "ParameterSynchronizer",
    "RankLocalCheckpointIO",
    "ParameterManager",
    "assign_parameter_owners",
    "build_parameter_manager",
]
