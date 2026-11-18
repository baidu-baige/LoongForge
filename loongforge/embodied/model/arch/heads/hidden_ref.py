# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 hidden-state reference ``HiddenStateRef`` (Lane C).

The signature follows ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.6.

This module fixes the second of the two gaps shared by all three surveyed
frameworks: **"the N-th conditioning input" and "the N-th output head" are
conflated**. Conditions go through ``streams/conditions.py``; a head reaching
for a hidden state goes through here.

``HiddenStateRef`` turns "which layer of which stream does this head read" into
a checkable declaration, instead of today's hooks plus globals plus layer
indices scattered around.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..streams.engine import EngineOutput

__all__ = ["HiddenStateRef", "captures_from_refs"]


@dataclass(frozen=True)
class HiddenStateRef:
    """Pointer to the hidden state of one layer of one stream.

    ``layer`` = None means the stream's final output, after ``finalize`` (the
    common case). An integer means the intermediate hidden state at that
    **joint** layer index, in which case the engine must have been asked to
    capture it (``captures``); V8 reconciles that at construction time.

    Why the index is a joint layer index and not a per-stream local one: the
    person writing this reference (the author of L4) thinks in terms of
    "layer 12" at the model level, while local indices are L2's internal
    numbering. ``StreamSpec.resolve_layer_index_map`` does the conversion.

    Attributes:
        stream: Name of the stream to read.
        layer: Joint layer index, or None for the stream's final output.
    """

    stream: str
    layer: int | None = None

    def __post_init__(self) -> None:
        if self.layer is not None and self.layer < 0:
            raise ValueError(
                f"HiddenStateRef({self.stream!r}).layer must be non-negative "
                f"or None, got {self.layer}"
            )

    @property
    def is_final(self) -> bool:
        """Whether this points at the final output rather than some mid layer."""
        return self.layer is None

    @property
    def capture_key(self) -> tuple[str, int] | None:
        """Key the engine needs in order to capture this state.

        The key is built in this one place only: it defines the key format of
        ``EngineOutput.captured``.

        Returns:
            ``(stream, layer)``, or None when the final output is wanted.
        """
        if self.layer is None:
            return None
        return (self.stream, self.layer)

    def resolve(self, output: EngineOutput) -> Tensor:
        """Pull this hidden state out of the engine's output.

        Intermediate layers go through ``require_capture``, which raises on a
        miss and lists the keys that were captured. "Requested but not
        captured" is something V8 should have blocked at construction time;
        this is the second net.

        Args:
            output: The engine output to read from.

        Returns:
            The referenced hidden-state tensor.

        Raises:
            KeyError: If the stream or the requested capture is missing.
        """
        if self.layer is None:
            return output.require(self.stream).hidden
        return output.require_capture(self.stream, self.layer)


def captures_from_refs(
        refs: tuple[HiddenStateRef, ...]) -> dict[str, tuple[int, ...]]:
    """Turn a set of refs into the engine's ``captures`` request (request side of I3).

    Step 4 of ``WorldModel.__init__`` calls this: the head table declares which
    hidden states it wants, and the engine's capture request is **derived from
    that declaration** instead of being written out a second time.

    Args:
        refs: The hidden-state references collected from the head table.

    Returns:
        Mapping from stream name to the sorted, deduplicated joint layer
        indices to capture. Refs pointing at final outputs are skipped.
    """
    wanted: dict[str, list[int]] = {}
    for ref in refs:
        if ref.layer is None:
            continue
        wanted.setdefault(ref.stream, []).append(ref.layer)
    return {stream: tuple(sorted(set(layers)))
            for stream, layers in wanted.items()}
