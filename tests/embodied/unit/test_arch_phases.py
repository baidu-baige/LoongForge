# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The ``phase`` label: the schedule role each token plays.

``frame`` says *when* a token is, ``phase`` says *what it is for*, and the
index where "condition" turns into "future" is a configured scalar rather than
a property of the grid. That is the whole reason this label exists: a
visibility expression that reads ``phase`` carries no token count, so the same
expression holds at one condition frame and at three, and the configuration
enters only here.

Two properties are load bearing beyond the truth table. Padded video slots are
labelled ``FUTURE``, which is what makes the bit-exact diffs against the
legacy mask builders come out equal rather than merely close; and both frame
counts are validated at construction, because a phase label that is wrong is
not a crash, it is a mask that silently gives or withholds a rectangle.

Design doc: ``mot_l0l1_layout_visibility.zh-CN.md`` §2.2.
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.visibility.phases import (
    CONDITION, REAL_VIDEO, Phase, condition_phases, phase_labels)

# One token per frame, four real frames and one padded slot, then two action
# tokens. One token per frame keeps the expected label a readable list; the
# padded slot is the point of the fixture rather than decoration.
VIDEO_FRAMES = (0, 1, 2, 3, -1)
ACTION_TOKENS = 2


def layout() -> TokenLayout:
    """The fixture layout: video segment first, as ``condition_phases`` assumes.

    Segment 0 being the video stream is ``from_geometry``'s convention, and
    this function is the only place in the module that depends on it.

    Returns:
        A layout with a ``frame`` label and nothing else.
    """
    video = len(VIDEO_FRAMES)
    total = video + ACTION_TOKENS
    frame = torch.tensor(VIDEO_FRAMES + (-1,) * ACTION_TOKENS,
                         dtype=torch.long)
    return TokenLayout(
        segments=(SegmentRange("video", 0, video),
                  SegmentRange("action", video, total)),
        labels={"frame": frame})


def test_the_condition_boundary_is_the_frame_count_and_nothing_else():
    """Two given frames, so frame 1 is current and everything after is future.

    The last given frame is ``CURRENT`` and the earlier ones are ``RECENT``,
    which is the distinction a rule needs to say "the newest observation" apart
    from "the ones before it" without naming a frame index.
    """
    phase = condition_phases(layout(), condition_frames=2)
    assert phase.tolist() == [
        Phase.RECENT, Phase.CURRENT, Phase.FUTURE, Phase.FUTURE,
        Phase.FUTURE, Phase.ACTION, Phase.ACTION]


def test_anchor_frames_split_the_given_prefix_further():
    """``anchor_frames`` renames the leading given frames, leaving the rest.

    Three given frames and one anchor puts all four real-frame codes on the
    sequence at once: the anchor, the one given frame between it and the
    newest, the newest, and the frame to be generated.
    """
    phase = condition_phases(layout(), condition_frames=3, anchor_frames=1)
    assert phase.tolist() == [
        Phase.ANCHOR, Phase.RECENT, Phase.CURRENT, Phase.FUTURE,
        Phase.FUTURE, Phase.ACTION, Phase.ACTION]


def test_no_anchor_frames_collapses_the_code_out_entirely():
    """The default draws no anchor distinction, and draws it by omission.

    A model that does not separate anchors gets a label with no ``ANCHOR`` in
    it, rather than one where ``ANCHOR`` means the same as ``RECENT``. Rules
    written against ``CONDITION`` are unaffected either way, which is what
    makes the parameter free to vary per provider.
    """
    phase = condition_phases(layout(), condition_frames=3)
    assert Phase.ANCHOR not in set(phase.tolist())
    assert phase[0] == Phase.RECENT


def test_padded_video_slots_are_future_not_a_phase_of_their_own():
    """``frame = -1`` comes out ``FUTURE``, which is what reproduces the source.

    The legacy mask builders size themselves by the *padded* sequence length
    and leave those rows in their non-condition branch, so ``FUTURE`` is the
    only code that makes the bit-exact diffs in the parity harnesses equal
    rather than close. A sixth code for padding would read better and would
    break them.
    """
    phase = condition_phases(layout(), condition_frames=2)
    video = layout().segments[0]
    frame = layout().labels_of("frame")[video.start:video.stop]
    padded = (frame == -1).nonzero().flatten().tolist()
    assert padded, "the fixture must pad the video segment"
    for index in padded:
        assert phase[video.start + index] == Phase.FUTURE


def test_streams_other_than_the_video_one_are_labelled_action():
    """One label covers the whole joint sequence, so no rule needs a fallback.

    Giving the non-video streams their own code is what lets ``label_edge``
    select on ``phase`` alone; if they were unlabelled, every phase rule would
    have to be intersected with a stream term to avoid reading a missing
    value.
    """
    phase = condition_phases(layout(), condition_frames=2)
    action = layout().segments[1]
    assert set(phase[action.start:action.stop].tolist()) == {Phase.ACTION}


def test_the_value_groups_are_the_ones_rules_actually_read():
    """``CONDITION`` and ``REAL_VIDEO`` are the two groupings, stated once.

    Rules take value sets, and these are the sets they take. Spelling them out
    here means a code added to ``Phase`` has to be placed in or out of each
    group deliberately: a new given-frame code left out of ``CONDITION`` would
    silently stop being visible to the rows that read condition frames.
    ``DELTA`` is in neither on purpose -- it is reserved for tokens that have
    no position in the geometry yet.
    """
    assert CONDITION == (Phase.ANCHOR, Phase.RECENT, Phase.CURRENT)
    assert REAL_VIDEO == CONDITION + (Phase.FUTURE,)
    assert Phase.DELTA not in REAL_VIDEO
    assert Phase.ACTION not in REAL_VIDEO


@pytest.mark.parametrize("kwargs,match", (
    (dict(condition_frames=0), "condition_frames must be >= 1"),
    (dict(condition_frames=-1), "condition_frames must be >= 1"),
    (dict(condition_frames=1, anchor_frames=1),
     r"anchor_frames must be in \[0, condition_frames\)"),
    (dict(condition_frames=2, anchor_frames=-1),
     r"anchor_frames must be in \[0, condition_frames\)"),
))
def test_the_frame_counts_are_validated_at_construction(kwargs, match):
    """Out-of-range counts raise rather than producing a lopsided label.

    None of these would crash on their own. ``condition_frames=0`` would put
    ``CURRENT`` on frame -1, which no token has, so the newest observation
    would simply go unlabelled; ``anchor_frames == condition_frames`` would
    swallow ``CURRENT`` under ``ANCHOR``. Both come out as a mask missing a
    rectangle, which is the failure mode this label exists to rule out.
    """
    with pytest.raises(ValueError, match=match):
        condition_phases(layout(), **kwargs)


def test_a_layout_without_frames_cannot_be_phased():
    """No ``frame`` label means no video schedule to read, so it raises.

    This one does not degrade for the same reason ``label_edge`` does not:
    the alternative is labelling every token ``ACTION``, under which every
    phase rule quietly matches nothing.
    """
    bare = TokenLayout(segments=(SegmentRange("video", 0, 4),), labels={})
    with pytest.raises(ValueError, match="no 'frame' label"):
        condition_phases(bare, condition_frames=1)


def test_phase_labels_defers_the_call_without_changing_the_answer():
    """The ``token_labels`` hook is the same computation, bound a step earlier.

    A provider cannot build this label itself, because it is per token and the
    layout does not exist until ``from_geometry`` has run. ``phase_labels``
    binds the two counts next to the visibility expression that reads them and
    hands back a producer. The only thing worth asserting is that the detour
    is free: same key, same tensor.
    """
    build = phase_labels(condition_frames=3, anchor_frames=1)
    produced = build(layout())
    assert set(produced) == {"phase"}
    assert torch.equal(
        produced["phase"],
        condition_phases(layout(), condition_frames=3, anchor_frames=1))
