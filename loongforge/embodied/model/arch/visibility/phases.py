# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The ``phase`` label: which role in the video schedule a token plays.

``frame`` says *when* a token is; ``phase`` says *what it is for*. The two are
not interchangeable, because the frame index at which "condition" turns into
"future" is a configured scalar, not a property of the grid.

Why a new label instead of widening ``role``: ``history(policy)`` already owns
``role`` and already fixed it as two-valued (``rules.py``: ``0`` = history,
``1`` = current/future). The video schedule needs four values for real frames
plus two more for the non-video streams, and overloading one label with two
arities is how a rule ends up reading a number it did not mean.

Design doc: ``mot_l0l1_layout_visibility.zh-CN.md`` §2.2.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import IntEnum

import torch
from torch import Tensor

from ..layout import TokenLayout

__all__ = [
    "Phase", "CONDITION", "REAL_VIDEO", "condition_phases", "phase_labels",
]


class Phase(IntEnum):
    """Phase codes carried by the ``phase`` label.

    The first four partition the video stream; ``DELTA`` and ``ACTION`` name
    the other streams so one label covers the whole joint sequence. ``DELTA``
    is declared but unused today: the architecture has no delta tokens yet
    (ruling 2 of §2.2.2, segment-versus-stream, is still open), and reserving
    the code now keeps the value set stable when they land.
    """

    ANCHOR = 0
    RECENT = 1
    CURRENT = 2
    FUTURE = 3
    DELTA = 4
    ACTION = 5


CONDITION: tuple[Phase, ...] = (Phase.ANCHOR, Phase.RECENT, Phase.CURRENT)
"""The real frames that are given, in frame order."""

REAL_VIDEO: tuple[Phase, ...] = CONDITION + (Phase.FUTURE,)
"""Every video phase, i.e. all keys a future row is allowed to read."""


def condition_phases(layout: TokenLayout, *, condition_frames: int,
                     anchor_frames: int = 0) -> Tensor:
    """Build the ``[total]`` phase label for a video-plus-streams layout.

    Segment 0 is the stream the geometry describes (``from_geometry``'s own
    convention), so it is the one that gets frame phases; every other segment
    is labelled ``ACTION``.

    **Padded video slots get ``FUTURE``.** ``build_labels`` gives them
    ``frame = -1``, and the legacy mask builders size themselves by the padded
    sequence length and leave those rows in the non-condition branch, so
    ``FUTURE`` is what reproduces them bit for bit.

    Args:
        layout: Layout to label; only segment bounds and the ``frame`` label
            are read.
        condition_frames: Number of leading real frames that are given. Frames
            at or after this index are ``FUTURE``.
        anchor_frames: How many of those leading frames are anchors. Zero means
            the model draws no anchor/recent distinction, which collapses
            ``ANCHOR`` out of the label without changing any rule.

    Returns:
        A ``[layout.total]`` long tensor of ``Phase`` codes.

    Raises:
        ValueError: If the frame counts are out of range, or if the layout has
            no ``frame`` label to derive the video phases from.
    """
    if condition_frames < 1:
        raise ValueError(
            f"condition_phases: condition_frames must be >= 1, got "
            f"{condition_frames}"
        )
    if not 0 <= anchor_frames < condition_frames:
        raise ValueError(
            f"condition_phases: anchor_frames must be in "
            f"[0, condition_frames), got anchor_frames={anchor_frames} "
            f"condition_frames={condition_frames}"
        )
    if not layout.has_label("frame"):
        raise ValueError(
            "condition_phases: layout has no 'frame' label, so video phases "
            f"cannot be derived; existing labels: {sorted(layout.labels)}"
        )

    video = layout.segments[0]
    frame = layout.labels_of("frame")[video.start:video.stop]
    phase = torch.full((layout.total,), int(Phase.ACTION), dtype=torch.long)
    video_phase = torch.full_like(frame, int(Phase.FUTURE))
    video_phase[(frame >= 0) & (frame < condition_frames - 1)] = Phase.RECENT
    video_phase[frame == condition_frames - 1] = Phase.CURRENT
    if anchor_frames:
        video_phase[(frame >= 0) & (frame < anchor_frames)] = Phase.ANCHOR
    phase[video.start:video.stop] = video_phase
    return phase


def phase_labels(*, condition_frames: int, anchor_frames: int = 0
                 ) -> Callable[[TokenLayout], Mapping[str, Tensor]]:
    """Bind the phase parameters into a ``WorldModelSpec.token_labels`` hook.

    A provider cannot build the label itself: the label is per token, so it
    needs the layout, which only exists after ``from_geometry``. Returning the
    producer keeps the parameters next to the visibility expression that reads
    them.

    Args:
        condition_frames: Forwarded to ``condition_phases``.
        anchor_frames: Forwarded to ``condition_phases``.

    Returns:
        A callable mapping a layout to ``{"phase": ...}``.
    """

    def build(layout: TokenLayout) -> Mapping[str, Tensor]:
        return {"phase": condition_phases(layout,
                                          condition_frames=condition_frames,
                                          anchor_frames=anchor_frames)}

    return build
