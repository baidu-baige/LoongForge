# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 扩散、timestep、噪声调度、损失与推理步进（Lane C）。"""

from __future__ import annotations

from .broadcast import DEFAULT_PER_SAMPLE, BroadcastSpec
from .loss import LossSpec, WeightRamp
from .schedule import ConstantSigma, NoiseSchedule, RectifiedFlow
from .spec import DiffusionSpec
from .stepping import Sampler, SteppingGate, Trajectory, UniformSteps
from .timesteps import (
    BetaDist, Constant, LogitNormal, TimestepDistribution, TimestepSampler,
    UniformShift,
)

__all__ = [
    # spec
    "DiffusionSpec",
    # timesteps
    "TimestepDistribution", "TimestepSampler",
    "UniformShift", "LogitNormal", "BetaDist", "Constant",
    # schedule
    "NoiseSchedule", "RectifiedFlow", "ConstantSigma",
    # loss
    "LossSpec", "WeightRamp",
    # broadcast
    "BroadcastSpec", "DEFAULT_PER_SAMPLE",
    # stepping
    "Trajectory", "SteppingGate", "UniformSteps", "Sampler",
]
