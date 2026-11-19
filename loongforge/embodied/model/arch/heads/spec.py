# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 output head spec ``HeadSpec`` (Lane C).

The signature follows ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.6, and
which fields are configurable follows ``mot_spec_config.zh-CN.md`` §3.7:
**of the seven fields only ``weight`` may be overridden from YAML**; the rest
describe topology and algorithm, which YAML does not get to touch (R5).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..spec.core import Spec
from ..streams.spec import HEAD_SLOTS, ModulationSlots
from .hidden_ref import HiddenStateRef

__all__ = ["HeadSpec"]


@dataclass(frozen=True)
class HeadSpec:
    """Declaration of one output head.

    ``source`` is a ``HiddenStateRef`` rather than a stream-name string: a head
    may consume an intermediate layer (several auxiliary heads of InternW0-Δ do),
    and a stream name cannot express a layer index.
    """

    name: str                           # Head name; also the metrics/output key
    impl: Spec                          # Implementation; must build an OutputHead
    source: HiddenStateRef              # Which layer of which stream it reads
    output_width: int                   # Last-dim width; checked vs impl (V8)
    modulation_slots: ModulationSlots = HEAD_SLOTS
    # The head's own modulation slots. The two slots (shift_out / scale_out) are
    # the action-head setting. A slot table may not be empty (see
    # ModulationSlots.__post_init__), so "this head does not modulate" is
    # expressed by needs_timestep_embedding=False, not by handing over an empty table
    needs_timestep_embedding: bool = False        # Whether the head modulates on timestep
    weight: float = 1.0                 # [YAML] **the only YAML-overridable field**

    def __post_init__(self) -> None:
        if self.output_width <= 0:
            raise ValueError(
                f"HeadSpec({self.name!r}).output_width must be positive, "
                f"got {self.output_width}"
            )
        if self.weight < 0:
            raise ValueError(
                f"HeadSpec({self.name!r}).weight must be non-negative, "
                f"got {self.weight}"
            )

    @property
    def source_stream(self) -> str:
        """Name of the stream this head reads. V8 checks that stream exists."""
        return self.source.stream

    @property
    def capture_key(self) -> tuple[str, int] | None:
        """Key for asking the engine to capture an intermediate layer.

        Returns:
            ``(stream, layer)`` when an intermediate layer is wanted, or None
            when the head consumes the stream's final output.
        """
        return self.source.capture_key
