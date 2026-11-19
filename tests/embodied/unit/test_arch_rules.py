# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Truth tables of the nine primitive L1 visibility rules (Lane A).

Two things must hold for every rule: the truth table when its labels are
present, and **degrading to all True rather than raising when a label is
missing** (the default semantics of interfaces §3.2). The second one is what
lets a single visibility expression be reused across models with different
label sets.
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.visibility.rules import (
    block_coupling, chunk_causal, frame_causal, history, noise_gate,
    prefix_only, same_sample, same_stream, stream_edge)


def layout_with(**labels: torch.Tensor) -> TokenLayout:
    """A six-token layout of two 3-token segments; the caller supplies labels."""
    return TokenLayout(
        segments=(SegmentRange("video", 0, 3), SegmentRange("action", 3, 6)),
        labels=labels)


def bare() -> TokenLayout:
    """A layout with no labels: every label-reading rule must degrade on it."""
    return layout_with()


def long_(*values: int) -> torch.Tensor:
    """Shorthand for a 1-D int64 label tensor."""
    return torch.tensor(values, dtype=torch.long)


def test_same_stream_truth_table():
    """``same_stream`` keeps the two per-stream blocks and nothing else.

    It is the base term every other rule is intersected with, so its block
    diagonal shape is what makes varlen attention expressible.
    """
    layout = layout_with(stream_id=long_(0, 0, 0, 1, 1, 1))
    out = same_stream().evaluate(layout)
    assert out[0].tolist() == [True, True, True, False, False, False]
    assert out[3].tolist() == [False, False, False, True, True, True]


def test_stream_edge_is_directed():
    """``stream_edge(src, dst)`` opens dst query rows onto src key columns only.

    The reverse edge is not implied: "the VL stream is read-only" is expressed
    exactly by having an in-edge and no out-edge.
    """
    layout = layout_with(stream_id=long_(0, 0, 0, 1, 1, 1))
    out = stream_edge("video", "action").evaluate(layout)
    # Action query rows can see video key columns
    assert out[3].tolist() == [True, True, True, False, False, False]
    # The reverse edge is not implied: "the VL stream is read-only" is exactly
    # an in-edge with no out-edge
    assert out[0].tolist() == [False] * 6


def test_stream_edge_unknown_name_raises():
    """A stream name absent from the layout raises instead of matching nothing."""
    layout = layout_with(stream_id=long_(0, 0, 0, 1, 1, 1))
    with pytest.raises(KeyError, match="layout has no stream"):
        stream_edge("video", "typo").evaluate(layout)


def test_chunk_causal_treats_missing_chunk_as_visible():
    """Tokens tagged ``chunk = -1`` stay mutually visible under chunk causality.

    Video tokens carry -1, and -1 >= -1 holds, so this rule has to be combined
    with ``same_stream()`` to keep the video block from leaking.
    """
    layout = layout_with(chunk=long_(-1, -1, -1, 0, 1, 2))
    out = chunk_causal().evaluate(layout)
    # Video tokens are tagged -1; -1 on both sides is visible, which is why
    # this must be combined with same_stream()
    assert out[0].tolist() == [True, True, True, False, False, False]
    assert out[4].tolist() == [True, True, True, True, True, False]


def test_frame_causal_window_is_inclusive_length():
    """``frame_causal(window)`` tests ``0 <= q_frame - k_frame < window``.

    The window is a length, not an offset, so ``window=1`` means "this frame
    only" and the mask is the identity over frame indices.
    """
    layout = layout_with(frame=long_(0, 1, 2, 3, 4, 5))
    unbounded = frame_causal().evaluate(layout)
    assert unbounded[2].tolist() == [True, True, True, False, False, False]
    # window=1 means "this frame only", i.e. the test is 0 <= q - k < window
    assert frame_causal(1).evaluate(layout).tolist() == \
        torch.eye(6, dtype=torch.bool).tolist()
    assert frame_causal(2).evaluate(layout)[3].tolist() == \
        [False, False, True, True, False, False]


def test_frame_causal_rejects_nonpositive_window():
    """``window=0`` is rejected: as a length it would hide everything."""
    with pytest.raises(ValueError, match="window must be positive"):
        frame_causal(0)


def test_prefix_only_is_on_key_side():
    """``prefix_only(n)`` opens the first n key columns for every query row.

    Being a key-side predicate is the point: "only the first n tokens may be
    read" is a property of the keys, so it masks columns, not rows. ``n=0`` is
    legal and is how a cross-stream edge gets switched off.
    """
    layout = layout_with(frame=long_(0, 0, 0, 1, 1, 1))
    out = prefix_only(2).evaluate(layout)
    assert out.shape == (6, 6)
    for row in range(6):
        assert out[row].tolist() == [True, True, False, False, False, False]
    # n=0 is legal: it is how a cross-stream edge is switched off
    assert not prefix_only(0).evaluate(layout).any()
    with pytest.raises(ValueError, match="must not be negative"):
        prefix_only(-1)


def test_noise_gate_reads_key_side_only():
    """``noise_gate(t)`` hides key columns whose noise level exceeds t.

    "Who may be seen" is a property of the key, so whole **columns** are
    masked, never whole rows.
    """
    layout = layout_with(noise_level=long_(0, 0, 1, 1, 2, 2))
    out = noise_gate(0).evaluate(layout)
    # "Who may be seen" is a key property, so whole columns are masked, not
    # whole rows
    for row in range(6):
        assert out[row].tolist() == [True, True, False, False, False, False]
    assert noise_gate(2).evaluate(layout).all()


def test_same_sample_isolates_packed_samples():
    """``same_sample`` blocks attention across samples packed in one sequence.

    Without it, packing two trajectories into one sequence would let one
    attend to the other, which is invisible in the loss.
    """
    layout = layout_with(sample_id=long_(0, 0, 1, 1, 2, 2))
    out = same_sample().evaluate(layout)
    assert out[0].tolist() == [True, True, False, False, False, False]
    assert out[5].tolist() == [False, False, False, False, True, True]


def test_history_policies():
    """``history`` has three policies over the key columns tagged as history.

    Here ``role`` 0 marks history tokens (frames 0, 1, 1) and role 1 the
    current ones. ``all`` keeps every history column, ``none`` drops them all,
    and ``last_only`` keeps just the columns of the largest history frame
    index, here the two ``frame == 1`` columns.
    """
    layout = layout_with(role=long_(0, 0, 0, 1, 1, 1),
                         frame=long_(0, 1, 1, 2, 2, 2))
    assert history("all").evaluate(layout).all()
    # none: every history column is hidden
    none_out = history("none").evaluate(layout)
    assert none_out[0].tolist() == [False, False, False, True, True, True]
    # last_only: only the largest history frame index survives (the two
    # frame == 1 columns)
    last_out = history("last_only").evaluate(layout)
    assert last_out[0].tolist() == [False, True, True, True, True, True]


def test_history_last_only_needs_frame_label():
    """``history("last_only")`` needs the ``frame`` label and says so.

    This is the one case that does not degrade: "the last frame" is undefined
    without frame indices, so guessing would silently pick the wrong columns.
    """
    layout = layout_with(role=long_(0, 0, 0, 1, 1, 1))
    with pytest.raises(KeyError, match="also reads the frame label"):
        history("last_only").evaluate(layout)


def test_history_rejects_unknown_policy():
    """An unknown policy name fails when the rule is built, not when evaluated."""
    with pytest.raises(ValueError, match="is invalid; legal values are"):
        history("latest")


def test_block_coupling_only_touches_same_block():
    """``block_coupling`` constrains only the columns inside the query's block.

    ``full`` keeps everything, ``diagonal`` keeps the token's own position
    within its block, and ``none`` drops the block's columns entirely. Columns
    of other blocks are left visible, so this rule composes with the others
    instead of overriding them.
    """
    layout = layout_with(block=long_(0, 0, 0, 1, 1, 1))
    assert block_coupling("full").evaluate(layout).all()
    diag = block_coupling("diagonal").evaluate(layout)
    assert diag[0].tolist() == [True, False, False, True, True, True]
    none_out = block_coupling("none").evaluate(layout)
    assert none_out[0].tolist() == [False, False, False, True, True, True]
    with pytest.raises(ValueError, match="is invalid; legal values are"):
        block_coupling("partial")


# A missing label degrades the rule to all True (it does not raise). This is
# what lets one visibility expression be reused across models.
DEGRADING = [
    ("same_stream", same_stream()),
    ("stream_edge", stream_edge("video", "action")),
    ("chunk_causal", chunk_causal()),
    ("frame_causal", frame_causal(2)),
    ("noise_gate", noise_gate(0)),
    ("same_sample", same_sample()),
    ("history_none", history("none")),
    ("history_last_only", history("last_only")),
    ("block_diagonal", block_coupling("diagonal")),
]


@pytest.mark.parametrize("name,vis", DEGRADING, ids=[n for n, _ in DEGRADING])
def test_missing_label_degrades_to_all_true(name, vis):
    """Every label-reading rule returns a ``[6, 6]`` all-True mask on a bare layout.

    Raising instead would force each model to carry labels for rules it does
    not use, so one shared visibility expression could not be reused.
    """
    out = vis.evaluate(bare())
    assert out.dtype == torch.bool, name
    assert out.shape == (6, 6), name
    assert bool(out.all()), name
