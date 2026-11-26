# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 diffusion: timesteps, noise schedules, losses and inference stepping (Lane C)."""

from __future__ import annotations

from .broadcast import DEFAULT_PER_SAMPLE, BroadcastSpec
from .loss import (BatchPadding, Bsmntw, CosineDistance, LossObjective,
                   LossSpec, MeanSquaredError, TimestepWeight,
                   TokenMeanSquaredError, UniformWeight, WeightRamp)
from .schedule import ConstantSigma, NoiseSchedule, RectifiedFlow
from .spec import DiffusionSpec
from .stepping import (EulerSampler, Sampler, SteppingGate, Trajectory,
                       UniformSteps)
from .timesteps import (
    BetaDist, Constant, LogitNormal, RngPolicy, TimestepDistribution,
    TimestepSampler, UniformShift,
)

__all__ = [
    # spec
    "DiffusionSpec",
    # timesteps
    "TimestepDistribution", "TimestepSampler", "RngPolicy",
    "UniformShift", "LogitNormal", "BetaDist", "Constant",
    # schedule
    "NoiseSchedule", "RectifiedFlow", "ConstantSigma",
    # loss
    "LossSpec", "WeightRamp",
    "LossObjective", "MeanSquaredError", "TokenMeanSquaredError", "CosineDistance",
    "TimestepWeight", "UniformWeight", "Bsmntw", "BatchPadding",
    # broadcast
    "BroadcastSpec", "DEFAULT_PER_SAMPLE",
    # stepping
    "Trajectory", "SteppingGate", "UniformSteps", "Sampler", "EulerSampler",
]
