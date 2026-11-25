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
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, NamedTuple

from ..streams.spec import StreamSpec
from .loss import BatchPadding, LossSpec
from .schedule import NoiseSchedule, RectifiedFlow
from .stepping import Sampler, SteppingGate
from .timesteps import YAML_KNOB, TimestepSampler

__all__ = ["DiffusionSpec"]


class _Knob(NamedTuple):
    """Where one YAML knob of a stream lives."""

    path: tuple[str, ...]   # attribute names from the stream's sampler to the field, e.g. ("distribution", "shift")
    owner: str              # class declaring the field, named in error messages
    kind: type              # float, int or bool: what a YAML value is checked against


# Annotations are strings under ``from __future__ import annotations``; a
# model's own class without that import hands over the type itself.
_KNOB_KINDS: dict[Any, type] = {"float": float, "int": int, "bool": bool,
                                float: float, int: int, bool: bool}


def _collect_knobs(obj: Any, *, stream: str, path: tuple[str, ...] = (),
                   into: dict[str, _Knob] | None = None) -> dict[str, _Knob]:
    """Collect the ``yaml_knob`` fields reachable from ``obj`` through dataclass fields.

    Only dataclass instances are walked; mappings, tuples and other objects are
    not, so they contribute no knobs.

    Args:
        obj: Where to start, normally a stream's ``TimestepSampler``.
        stream: Stream name, used in error messages.
        path: Attribute path from the stream's sampler to ``obj``.
        into: Accumulator shared by the recursive calls.

    Returns:
        Knob name (the field name) to its ``_Knob``, in declaration order.

    Raises:
        ValueError: Two fields of one stream share a knob name (L3/V10).
        TypeError: A knob field is not annotated float, int or bool.
    """
    into = {} if into is None else into
    if not is_dataclass(obj) or isinstance(obj, type):
        return into
    for item in fields(obj):
        here = path + (item.name,)
        if not item.metadata.get(YAML_KNOB):
            _collect_knobs(getattr(obj, item.name), stream=stream, path=here, into=into)
            continue
        kind = _KNOB_KINDS.get(item.type)
        if kind is None:
            raise TypeError(
                f"{type(obj).__name__}.{item.name} is a yaml_knob annotated "
                f"{item.type!r}; YAML knobs are float, int or bool")
        if item.name in into:
            raise ValueError(
                f"stream {stream!r}: YAML knob {item.name!r} is declared twice, at "
                f"{'.'.join(into[item.name].path)} and {'.'.join(here)}; one stream's "
                "knob names must be unique (L3/V10)")
        into[item.name] = _Knob(here, type(obj).__name__, kind)
    return into


def _replace_at(obj: Any, path: tuple[str, ...], value: Any) -> Any:
    """Rebuild ``obj`` with the field at ``path`` set to ``value``.

    Every dataclass on the way is rebuilt with ``dataclasses.replace``, so each
    one's ``__post_init__`` checks the new value.

    Args:
        obj: The dataclass instance at the start of ``path``.
        path: Attribute names down to the field.
        value: The new value of that field.
    """
    head, rest = path[0], path[1:]
    new = _replace_at(getattr(obj, head), rest, value) if rest else value
    return replace(obj, **{head: new})


def _coerce(knob: _Knob, value: Any, *, key: str) -> Any:
    """Check a YAML value against the knob's kind; float knobs accept an int.

    Args:
        knob: The knob being written.
        value: The value YAML gave.
        key: The full YAML key, used in the error message.
    """
    if isinstance(value, bool):
        ok = knob.kind is bool
    elif isinstance(value, int):
        ok = knob.kind in (int, float)
    else:
        ok = isinstance(value, float) and knob.kind is float
    if not ok:
        raise TypeError(f"{key} must be {knob.kind.__name__}, got {type(value).__name__} {value!r}")
    return float(value) if knob.kind is float else value


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
        batch_padding: Stream name -> the per-sample padding read from the
            batch; padded units do not count in that stream's flow losses.
            Only streams with a timestep can have one. Frozen to a read-only
            mapping at construction time.
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
    batch_padding: Mapping[str, BatchPadding] = field(default_factory=dict)
    # Provider-only: the keys name batch entries, which YAML does not bind.

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
        unknown = sorted(set(self.batch_padding) - set(self.timesteps))
        if unknown:
            raise ValueError(
                f"DiffusionSpec.batch_padding names streams {unknown} that draw no "
                "timestep; padding masks flow losses, which only such streams have "
                f"(streams with a timestep: {list(self.timesteps)})")
        for stream, padding in self.batch_padding.items():
            if not isinstance(padding, BatchPadding):
                raise TypeError(
                    f"DiffusionSpec.batch_padding[{stream!r}] must be a BatchPadding, "
                    f"got {type(padding).__name__}")
        object.__setattr__(self, "batch_padding",
                           MappingProxyType(dict(self.batch_padding)))
        for stream, sampler in self.timesteps.items():
            _collect_knobs(sampler, stream=stream)          # L3/V10: unique knob names per stream

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

    def yaml_knobs(self, stream: str) -> dict[str, Any]:
        """The YAML knobs of one stream and their current values.

        Args:
            stream: Stream name.

        Returns:
            Knob name to current value, in declaration order; empty when the
            stream's sampler declares no knob.

        Raises:
            KeyError: The stream has no timestep entry; the message lists the
                streams that have one.
        """
        sampler = self.sampler_for(stream)
        values: dict[str, Any] = {}
        for name, knob in _collect_knobs(sampler, stream=stream).items():
            obj = sampler
            for attr in knob.path:
                obj = getattr(obj, attr)
            values[name] = obj
        return values

    def with_yaml_knobs(self, section: Mapping[str, Any]) -> "DiffusionSpec":
        """Write the YAML ``diffusion:`` section onto the declared objects.

        Each ``diffusion.<stream>.<knob>`` lands on the field of that name
        under the stream's sampler (``yaml_knob`` marks which fields qualify).
        Applied after the provider has built the spec, which is why a knob
        must not feed a value the provider derives.

        Args:
            section: ``{stream: {knob: value}}``; empty returns ``self``.

        Returns:
            A new ``DiffusionSpec``. ``timesteps`` keeps its key order, which is
            the draw order.

        Raises:
            ValueError: A stream draws no timestep, a key is not one of the
                stream's knobs, or the owning object rejects the value; the
                message starts with the full YAML key.
            TypeError: A stream block is not a mapping, or a value has the
                wrong type.
        """
        if not section:
            return self
        timesteps = dict(self.timesteps)
        # Replacing a value keeps every path, so the tables stay valid while writing.
        tables = {name: _collect_knobs(sampler, stream=name) for name, sampler in timesteps.items()}
        with_knobs = ", ".join(f"{name} ({', '.join(knobs)})"
                               for name, knobs in tables.items() if knobs) or "none"
        for stream, block in section.items():
            if stream not in timesteps:
                raise ValueError(
                    f"diffusion.{stream}: stream {stream!r} draws no timestep; the streams "
                    f"with diffusion knobs are: {with_knobs}")
            if not block:
                continue
            if not isinstance(block, Mapping):
                raise TypeError(f"diffusion.{stream} must be a mapping, got {type(block).__name__}")
            knobs = tables[stream]
            for key, value in block.items():
                full = f"diffusion.{stream}.{key}"
                if not knobs:
                    raise ValueError(
                        f"{full}: stream {stream!r} has no YAML knobs (its sampler draws "
                        f"{type(getattr(timesteps[stream], 'distribution', timesteps[stream])).__name__}); "
                        f"the streams with diffusion knobs are: {with_knobs}")
                if key not in knobs:
                    raise ValueError(
                        f"{full} is not a knob of stream {stream!r}; its knobs are "
                        f"{', '.join(f'{name} ({knob.owner})' for name, knob in knobs.items())}. "
                        f"Stream sizes go under streams.{stream}")
                knob = knobs[key]
                checked = _coerce(knob, value, key=full)
                try:
                    timesteps[stream] = _replace_at(timesteps[stream], knob.path, checked)
                except ValueError as exc:
                    raise ValueError(f"{full}={value!r}: {exc}") from exc
        return replace(self, timesteps=timesteps)

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
