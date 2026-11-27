# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 diffusion: timesteps, noise schedules, timestep weights and inference stepping (Lane C)."""

from __future__ import annotations

from .schedule import ConstantSigma, NoiseSchedule, RectifiedFlow
from .spec import DiffusionSpec
from .stepping import (EulerSampler, Sampler, SteppingGate, Trajectory,
                       UniformSteps)
from .timesteps import (
    BetaDist, Constant, LogitNormal, RngPolicy, TimestepDistribution,
    TimestepSampler, UniformShift,
)
from .weights import Bsmntw

__all__ = [
    # spec
    "DiffusionSpec",
    # timesteps
    "TimestepDistribution", "TimestepSampler", "RngPolicy",
    "UniformShift", "LogitNormal", "BetaDist", "Constant",
    # schedule
    "NoiseSchedule", "RectifiedFlow", "ConstantSigma",
    # diffusion timestep weights
    "Bsmntw",
    # stepping
    "Trajectory", "SteppingGate", "UniformSteps", "Sampler", "EulerSampler",
]
