# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 noise schedules ``NoiseSchedule`` / ``ConstantSigma`` (Lane C).

The signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.3.

Division of labour with ``timesteps.py``: the distribution decides **which t is
drawn**, the schedule decides **how much noise t corresponds to**. The direct
reason for keeping them apart is that under rectified flow
``x_t = (1-t)*x_0 + t*eps`` the same t is both the sample from the distribution
and the interpolation coefficient; written together, swapping the distribution
would also swap the interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

__all__ = ["NoiseSchedule", "ConstantSigma", "RectifiedFlow"]


class NoiseSchedule(Protocol):
    """t -> (noising coefficients, regression target).

    These three methods are the whole of a forward process.
    """

    def add_noise(self, clean: Tensor, noise: Tensor,
                  t: Tensor) -> Tensor:
        """Produce ``x_t``. ``t`` is already broadcast to a multipliable shape
        by ``BroadcastSpec.expand``."""
        ...

    def target(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Regression target. Under rectified flow this is ``noise - clean``
        (the velocity).

        It is a **method** rather than a constant string: enumerations like
        "v_prediction" force a separate ``if`` in three places today.
        """
        ...

    def sigma(self, t: Tensor) -> Tensor:
        """Noise scale for t. ``visibility.rules.noise_gate`` converts its
        discrete levels through this."""
        ...


@dataclass(frozen=True)
class RectifiedFlow:
    """Rectified flow. **12 of the 13 models use it**, so it is the default.

    ``x_t = (1-t)*clean + t*noise``, with target ``noise - clean``.
    """

    def add_noise(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        return (1.0 - t) * clean + t * noise

    def target(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        del t                      # target is t-independent under rectified flow
        return noise - clean

    def sigma(self, t: Tensor) -> Tensor:
        return t


@dataclass(frozen=True)
class ConstantSigma:
    """Fixed noise scale. With ``timesteps.Constant`` it expresses "this stream
    is not noised".

    When ``sigma=0.0``, ``add_noise`` returns ``clean`` unchanged, which lets a
    clean conditioning stream and a noised stream take the **same code path**
    instead of branching on ``if is_condition``.

    Attributes:
        sigma_value: Constant noise scale; 0.0 makes ``add_noise`` the identity.
    """

    sigma_value: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_value < 0:
            raise ValueError(
                f"ConstantSigma.sigma_value must be non-negative, "
                f"got {self.sigma_value}"
            )

    def add_noise(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        del t
        if self.sigma_value == 0.0:
            return clean
        return clean + self.sigma_value * noise

    def target(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        del clean, t
        return noise

    def sigma(self, t: Tensor) -> Tensor:
        return t.new_full(t.shape, self.sigma_value)
