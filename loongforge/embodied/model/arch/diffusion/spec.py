# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 diffusion spec ``DiffusionSpec`` (Lane C).

One ``DiffusionSpec`` describes how a model draws timesteps, adds noise and
runs iterative inference. Supervision loss terms live on
``WorldModelSpec.losses`` (``arch/losses.py``), decoupled from diffusion so
non-diffusion and hybrid models do not carry a ``DiffusionSpec`` for their
losses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ..streams.spec import StreamSpec
from .schedule import NoiseSchedule, RectifiedFlow
from .stepping import Sampler, SteppingGate
from .timesteps import TimestepSampler

__all__ = ["DiffusionSpec"]


@dataclass(frozen=True)
class DiffusionSpec:
    """The diffusion / flow-matching process configuration.

    The keys of ``timesteps`` are **stream names**: only streams with
    ``StreamSpec.needs_timestep=True`` need an entry, and both a missing and an
    extra entry are construction-time errors (V5).

    Attributes:
        timesteps: Stream name -> timestep sampler; frozen to a read-only
            mapping at construction time.
        schedule: The forward noising process; rectified flow by default.
        sampling_gate: Which stream steps when during iterative sampling, or None.
        solver: The per-step numerical update rule, or None.
        noise_levels: Discrete noise levels, if any.
    """

    timesteps: Mapping[str, TimestepSampler]
    schedule: NoiseSchedule = field(default_factory=RectifiedFlow)
    sampling_gate: SteppingGate | None = None
    solver: Sampler | None = None
    noise_levels: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (self.sampling_gate is None) != (self.solver is None):
            raise ValueError(
                "DiffusionSpec.sampling_gate and solver must both be given or both be "
                f"omitted: got sampling_gate={self.sampling_gate!r}, solver={self.solver!r}"
            )
        if len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError(
                f"DiffusionSpec.noise_levels has duplicate levels: "
                f"{self.noise_levels}"
            )
        if any(level < 0 for level in self.noise_levels):
            raise ValueError(
                f"DiffusionSpec.noise_levels must be non-negative: "
                f"{self.noise_levels}"
            )
        object.__setattr__(self, "timesteps",
                           MappingProxyType(dict(self.timesteps)))

    @property
    def is_iterative(self) -> bool:
        """Whether iterative sampling is used."""
        return self.sampling_gate is not None

    @property
    def supports_noise_gate(self) -> bool:
        """Whether discrete noise levels are declared."""
        return bool(self.noise_levels)

    def sampler_for(self, stream: str) -> TimestepSampler:
        """Return one stream's timestep sampler."""
        try:
            return self.timesteps[stream]
        except KeyError:
            raise KeyError(
                f"DiffusionSpec.timesteps has no entry for stream {stream!r}; "
                f"configured: {sorted(self.timesteps)}"
            ) from None

    def validate_against(self, streams: Sequence[StreamSpec]) -> None:
        """Check that ``timesteps`` matches ``StreamSpec.needs_timestep`` (V5)."""
        needs = {spec.name for spec in streams if spec.needs_timestep}
        configured = set(self.timesteps)
        missing = sorted(needs - configured)
        if missing:
            raise ValueError(
                f"streams {missing} declare needs_timestep=True but have no "
                "entry in DiffusionSpec.timesteps (L3/V5). A missing entry "
                "does not degrade to zero noise, that would be a silent "
                "noise-free training run"
            )
        extra = sorted(configured - needs)
        if extra:
            raise ValueError(
                f"DiffusionSpec.timesteps configures streams {extra}, but "
                "their needs_timestep=False or they are not streams of this "
                "model at all (L3/V5)"
            )
