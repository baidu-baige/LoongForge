# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion-specific timestep loss weights implementing ``TimestepWeight``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from .timesteps import Constant, TimestepSampler, UniformShift

__all__ = ["Bsmntw"]


@lru_cache(maxsize=None)
def _bsmntw_stats(shift: float, steps: int) -> tuple[float, float]:
    """Minimum and mean-normalizer of the bell curve on the training grid.

    Same grid as master ``WanContinuousFlowMatchScheduler``: descending fp64 u
    excluding the endpoint 0, pushed through the shift and scaled to steps.
    """
    u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
    t = shift * u / (1.0 + (shift - 1.0) * u) * float(steps)
    y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
    y_min = float(y.min().item())
    return y_min, float((y - y_min).mean().item())


@dataclass(frozen=True)
class Bsmntw:
    """Bell-shaped, mean-normalized timestep weight (DiffSynth ``bsmntw``).

    ``exp(-2 * ((t - T/2) / T)^2)``, shifted so its minimum on the training
    grid is 0 and divided by its mean there. The grid depends on the source
    stream's ``UniformShift.shift``, which is read from the sampler rather
    than written here a second time. Used by the VPP2 action loss and both
    fastwam losses (master ``scheduler.py:training_weight``).

    Attributes:
        eps: Floor added to the normalizer; master uses 1e-10.
    """

    eps: float = 1e-10

    def check(self, sampler: TimestepSampler, *, loss: str) -> None:
        if isinstance(sampler.distribution, Constant):
            # The grid minimum sits at u=1, so a Constant(1.0) stream would get
            # weight 0 forever: a silently dead loss term (v2 T24).
            raise ValueError(
                f"LossSpec({loss!r}) weights a Constant timestep stream with Bsmntw, "
                "whose weight at sigma=1 is exactly 0; use UniformWeight")
        if not isinstance(sampler.distribution, UniformShift):
            raise ValueError(
                f"LossSpec({loss!r}): Bsmntw normalizes on the UniformShift training "
                f"grid, but the source stream draws from {type(sampler.distribution).__name__}")

    def __call__(self, t: Tensor, *, sampler: TimestepSampler,
                 num_train_timesteps: int) -> Tensor:
        y_min, norm = _bsmntw_stats(float(sampler.distribution.shift), int(num_train_timesteps))
        t = t.to(dtype=torch.float32)
        steps = float(num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        weight = (y - y_min) / (norm + self.eps)
        return weight.reshape(()) if weight.numel() == 1 else weight
