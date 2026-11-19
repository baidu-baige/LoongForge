# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 diffusion spec ``DiffusionSpec`` (Lane C).

The signature follows ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.1.

One ``DiffusionSpec`` describes how a model adds noise, computes its losses and
runs inference. It is **per stream** (a mapping) rather than one global value:
VPP2's video stream and action stream use different timestep distributions, and
today that shows up as two separately hard-coded sampling blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ..streams.spec import StreamSpec
from .loss import LossSpec
from .schedule import NoiseSchedule, RectifiedFlow
from .stepping import Sampler, SteppingGate
from .timesteps import TimestepSampler

__all__ = ["DiffusionSpec"]


@dataclass(frozen=True)
class DiffusionSpec:
    """The complete diffusion / flow-matching configuration.

    The keys of ``timesteps`` are **stream names**: only streams with
    ``StreamSpec.needs_timestep=True`` need an entry, and both a missing and an
    extra entry are construction-time errors (V5). The behaviour this check
    replaces is "if a stream cannot find a timestep, treat it as 0", which turns
    a forgotten timestep into a silent noise-free training run.

    Attributes:
        timesteps: Stream name -> timestep sampler; frozen to a read-only
            mapping at construction time.
        losses: The loss terms, with unique ``name`` values.
        schedule: The forward noising process; rectified flow by default.
        sampling_gate: Which stream steps when during iterative sampling, or None.
        solver: The per-step numerical update rule, or None.
        noise_levels: Discrete noise levels, if any.
    """

    timesteps: Mapping[str, TimestepSampler]
    losses: tuple[LossSpec, ...]
    schedule: NoiseSchedule = field(default_factory=RectifiedFlow)
    sampling_gate: SteppingGate | None = None        # None => no iterative sampling
    solver: Sampler | None = None                   # None => same as above
    noise_levels: tuple[float, ...] = ()
    # Discrete noise levels. Non-empty => visibility.rules.noise_gate is usable.
    # They must be **discrete levels** rather than a continuous sigma: a
    # continuous value would make the structural mask per-sample (INV-1).
    # Full capability negotiation is tracked as a separate implementation item.

    def __post_init__(self) -> None:
        if not self.losses:
            raise ValueError(
                "DiffusionSpec.losses must not be empty: a model with no loss "
                "cannot train"
            )
        names = [loss.name for loss in self.losses]
        if len(set(names)) != len(names):
            raise ValueError(
                f"DiffusionSpec.losses has duplicate names: {names}")
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

    # ---- queries ----

    @property
    def is_iterative(self) -> bool:
        """Whether iterative sampling is used. False means single-step
        feed-forward (a distilled model)."""
        return self.sampling_gate is not None

    @property
    def supports_noise_gate(self) -> bool:
        """Whether discrete noise levels are declared, making the
        ``noise_gate`` rule usable."""
        return bool(self.noise_levels)

    def sampler_for(self, stream: str) -> TimestepSampler:
        """Return one stream's timestep sampler.

        Args:
            stream: Stream name.

        Returns:
            The timestep sampler configured for that stream.

        Raises:
            KeyError: If the stream has no entry; the message lists the streams
                that are configured.
        """
        try:
            return self.timesteps[stream]
        except KeyError:
            raise KeyError(
                f"DiffusionSpec.timesteps has no entry for stream {stream!r}; "
                f"configured: {sorted(self.timesteps)}"
            ) from None

    def losses_of(self, *, stream: str | None = None,
                  head: str | None = None) -> tuple[LossSpec, ...]:
        """Filter losses by source. Passing neither returns all of them.

        Args:
            stream: Keep only losses computed on this stream.
            head: Keep only losses computed on this head.

        Returns:
            The matching loss specs, in declaration order.

        Raises:
            ValueError: If both ``stream`` and ``head`` are given.
        """
        if stream is not None and head is not None:
            raise ValueError(
                "losses_of: stream and head must not both be given")
        if stream is None and head is None:
            return self.losses
        return tuple(loss for loss in self.losses
                     if (loss.stream == stream if stream is not None
                         else loss.head == head))

    # ---- V5 / V6: reconcile with the stream set. Construction time, not
    # forward time ----

    def validate_against(self, streams: Sequence[StreamSpec]) -> None:
        """Check that the timestep configuration matches the streams'
        ``needs_timestep`` declarations (V5), and that every loss's ``stream``
        actually exists (the stream half of V6).

        The head half is checked by ``WorldModelSpec``, which holds the head
        table; this class cannot see the heads.

        Args:
            streams: The stream specs of this model.

        Raises:
            ValueError: If a stream needing a timestep has no entry, if an entry
                names a stream that does not need one, or if a loss references
                an unknown stream.
        """
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
        known = {spec.name for spec in streams}
        for loss in self.losses:
            if loss.stream is not None and loss.stream not in known:
                raise ValueError(
                    f"LossSpec({loss.name!r}).stream={loss.stream!r} is not a "
                    f"stream of this model; this model's streams are "
                    f"{sorted(known)} (L3/V6)"
                )
