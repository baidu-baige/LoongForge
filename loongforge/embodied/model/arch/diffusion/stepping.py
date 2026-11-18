# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 inference stepping: ``Sampler`` / ``SteppingGate`` / ``Trajectory``
(Lane C).

The signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.5.
**Where ``Trajectory`` / ``SteppingGate`` belong at configuration time** is an
open item in the master plan §10-7; only the runtime signatures are fixed here.

What ``SteppingGate`` expresses has no abstraction today: **different streams
may take a different number of steps**. Fast-WAM dropping the video stream at
inference (video stream 0 steps, action stream 10) is this case, and today it
shows up as a block inside ``infer_action`` that forks completely away from the
training path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from torch import Tensor

__all__ = ["Trajectory", "SteppingGate", "UniformSteps", "Sampler",
           "EulerSampler"]


@dataclass
class Trajectory:
    """State of one sampling run: one latent per stream plus the current step.

    ``step`` is the **global** step number; whether a given stream moves on this
    step is answered by ``SteppingGate``.

    Attributes:
        latents: Stream name -> current latent.
        step: Global step number.
        extras: Free-form carry-over state for samplers that need it.
    """

    latents: dict[str, Tensor]
    step: int = 0
    extras: dict[str, object] = field(default_factory=dict)

    def active_streams(self, gate: SteppingGate) -> tuple[str, ...]:
        """Stream names that advance on this step.

        An empty tuple is legal: it means nothing moves this step.

        Args:
            gate: The gate deciding per-stream stepping.

        Returns:
            The active stream names, sorted.
        """
        return tuple(name for name in sorted(self.latents)
                     if gate.should_step(name, self.step))

    def advance(self, updates: Mapping[str, Tensor]) -> Trajectory:
        """Write back the updated streams and increment the step number.

        Streams absent from ``updates`` are carried over unchanged.

        Args:
            updates: Stream name -> new latent.

        Returns:
            A new trajectory at ``step + 1``.

        Raises:
            KeyError: If ``updates`` names a stream not in this trajectory.
        """
        unknown = set(updates) - set(self.latents)
        if unknown:
            raise KeyError(
                f"Trajectory.advance got unknown streams {sorted(unknown)}; "
                f"this trajectory's streams are {sorted(self.latents)}"
            )
        latents = dict(self.latents)
        latents.update(updates)
        return Trajectory(latents=latents, step=self.step + 1,
                          extras=dict(self.extras))


class SteppingGate(Protocol):
    """Which stream advances on which step. The only place where a per-stream
    step count is expressed at inference."""

    def total_steps(self) -> int:
        """Global upper bound on the step count. A construction-time constant
        (INV-2: the inference graph must be static too)."""
        ...

    def should_step(self, stream: str, step: int) -> bool:
        """Whether ``stream`` advances on step ``step``."""
        ...


@dataclass(frozen=True)
class UniformSteps:
    """Every stream steps in lockstep for ``num_steps`` steps. **The setting
    most models use**, and the default.

    ``skip`` lists streams dropped entirely at inference (Fast-WAM's video
    stream). Dropping is expressed as "``should_step`` is always False for this
    stream", not as a separate code path.

    Attributes:
        num_steps: Number of sampling steps; must be positive.
        skip: Streams that never advance.
    """

    num_steps: int
    skip: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(
                f"UniformSteps.num_steps must be positive, "
                f"got {self.num_steps}"
            )

    def total_steps(self) -> int:
        return self.num_steps

    def should_step(self, stream: str, step: int) -> bool:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        return stream not in self.skip and step < self.num_steps


class Sampler(Protocol):
    """One ODE / SDE update step. The inference loop of ``WorldModel`` calls it
    in step (2)."""

    def step(self, trajectory: Trajectory,
             prediction: Mapping[str, Tensor], *,
             gate: SteppingGate) -> Trajectory:
        """Consume the model prediction and produce the next trajectory point.

        Only the streams in ``trajectory.active_streams(gate)`` may be updated:
        updating a stream that should not move defeats "drop the video stream",
        and nothing in the numbers reveals it.
        """
        ...


@dataclass(frozen=True)
class EulerSampler:
    """First-order Euler ODE update. The default sampler for rectified flow.

    **This class carries no step count**: the step count belongs to
    ``SteppingGate``. Today both are crammed into one scheduler object (VPP2's
    four scheduler instances differ only by a shift, ``vpp2.py:122-137``), so
    "change the step count" and "change the distribution" interfere.
    """

    def step(self, trajectory: Trajectory,
             prediction: Mapping[str, Tensor], *,
             gate: SteppingGate) -> Trajectory:
        """``x <- x - dt * v`` with dt = 1 / total_steps.

        A rectified flow trajectory is ``x_t = (1-t)*x_0 + t*eps``, so
        ``dx/dt = eps - x_0``, which is exactly the velocity the head regresses
        (``RectifiedFlow.target``). Sampling walks from t=1 down to t=0, so the
        step **subtracts**. The symptom of flipping the sign is that sampling
        returns pure noise while shapes and losses all look fine.

        Only the streams active on this step are updated. A prediction supplied
        for an inactive stream raises instead of being ignored: when "drop the
        video stream" silently fails, nothing in the numbers shows it, so this
        check is the only thing that surfaces it.

        Args:
            trajectory: Current trajectory point.
            prediction: Stream name -> predicted velocity.
            gate: The gate deciding per-stream stepping.

        Returns:
            The next trajectory point.

        Raises:
            KeyError: If ``prediction`` covers an inactive stream, or misses an
                active one.
        """
        active = trajectory.active_streams(gate)
        extra = sorted(set(prediction) - set(active))
        if extra:
            raise KeyError(
                f"EulerSampler.step: step {trajectory.step} got predictions "
                f"for streams {extra}, but they do not advance on this step "
                f"(active={list(active)})"
            )
        missing = sorted(set(active) - set(prediction))
        if missing:
            raise KeyError(
                f"EulerSampler.step: step {trajectory.step} is missing "
                f"predictions for streams {missing}"
            )
        delta = 1.0 / gate.total_steps()
        return trajectory.advance({
            name: trajectory.latents[name] - delta * prediction[name]
            for name in active})
