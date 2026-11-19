# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 layout: ``TokenLayout`` invariants, labels and ``labels_of`` (interface I2).

``TokenLayout`` is the one description of the concatenated token sequence: a
segment table saying which half-open ``[start, stop)`` range each stream owns,
plus a label table mapping a name to a ``[total]`` tensor. ``labels_of`` is the
whole of interface I2, so visibility rules and broadcast specs align on named
labels instead of guessing an axis out of a tensor shape.
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.geometry import GeometrySpec
from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout


class _Stream:
    """Reads ``.name`` only, which is all ``from_geometry`` asks of a stream."""

    def __init__(self, name: str) -> None:
        self.name = name


def small_geometry(**extra) -> GeometrySpec:
    """A 12-token geometry: 2 latent frames x 4 patches plus 4 action steps."""
    kwargs = dict(frame_height=32, frame_width=32, num_frames=5, num_views=1,
                  vae_spatial_stride=16, patch_size=(1, 1, 1),
                  action_horizon=4, action_dim=7, action_video_freq_ratio=2)
    kwargs.update(extra)
    return GeometrySpec(**kwargs)


def small_layout() -> TokenLayout:
    """The video+action layout built from :func:`small_geometry`."""
    return TokenLayout.from_geometry(small_geometry(),
                                     [_Stream("video"), _Stream("action")])


def test_segments_cover_and_order():
    """Segments tile ``[0, total)`` with no gaps, in declared concat order.

    Every downstream slice (``split``, the engine's per-stream views) is
    derived from these offsets, so an off-by-one here silently mixes streams.
    """
    layout = small_layout()
    assert layout.stream_names == ("video", "action")
    assert layout.total == 12
    assert [(s.stream, s.start, s.stop) for s in layout.segments] == [
        ("video", 0, 8), ("action", 8, 12)]
    assert layout.range_of("action").length == 4


def test_labels_are_batch_free_and_aligned():
    """Every label is a 1-D ``[total]`` tensor whose values match the geometry.

    Labels must stay batch independent so one visibility mask can be compiled
    once and shared across the batch; per-sample validity travels separately on
    ``AttentionPlan.key_valid``.
    """
    layout = small_layout()
    for key in ("stream_id", "frame", "row", "col", "is_padding", "chunk"):
        tensor = layout.labels_of(key)
        assert tensor.ndim == 1, key
        assert tensor.shape[0] == layout.total, key
    assert layout.labels_of("stream_id").tolist() == [0] * 8 + [1] * 4
    # Video tokens take the geometric frame index; an action token takes
    # step // action_video_freq_ratio, the latent frame it aligns to.
    assert layout.labels_of("frame").tolist() == \
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    # Action tokens get row/col = -1, not 0: a 0 would make spatial rules treat
    # them as "row 0" and silently take effect.
    assert layout.labels_of("row")[8:].tolist() == [-1] * 4
    assert layout.labels_of("chunk").tolist() == [-1] * 8 + [0, 0, 1, 1]


def test_noise_level_not_tagged_by_geometry():
    """``noise_level`` is attached by L3 at noising time, not by the geometry.

    Interfaces §3.2 puts the noise tier on the L3 side so a model that never
    adds noise simply has no such label, and noise-dependent rules degrade to
    all-True instead of raising.
    """
    assert not small_layout().has_label("noise_level")


def test_labels_of_missing_raises_and_lists():
    """``labels_of`` on an absent label raises ``KeyError`` naming the label.

    Lane C has to fail loudly on a missing label rather than infer the axis
    from a shape, so the error text is part of the contract.
    """
    with pytest.raises(KeyError, match="noise_level"):
        small_layout().labels_of("noise_level")


def test_with_labels_rejects_clash():
    """``with_labels`` appends new labels but refuses to overwrite one.

    Labels have different owners (geometry, stream, L3 noising). Allowing an
    overwrite would give one label two sources, with the later write winning
    silently.
    """
    layout = small_layout()
    tier = torch.zeros(layout.total, dtype=torch.long)
    assert layout.with_labels(noise_level=tier).has_label("noise_level")
    with pytest.raises(ValueError, match="one label may have only one source"):
        layout.with_labels(frame=tier)


def test_split_matches_segments():
    """``split`` slices a packed tensor back into exactly the segment ranges.

    This replaces the hand-written start/end cursors each model used to carry,
    so concatenation and de-concatenation can no longer drift apart.
    """
    layout = small_layout()
    packed = torch.arange(2 * layout.total * 3, dtype=torch.float32)
    packed = packed.reshape(2, layout.total, 3)
    parts = layout.split(packed, dim=1)
    assert set(parts) == {"video", "action"}
    assert parts["video"].shape[1] == 8
    assert torch.equal(torch.cat([parts["video"], parts["action"]], dim=1),
                       packed)
    assert set(layout.split(packed, dim=1, streams=["action"])) == {"action"}


def test_split_rejects_wrong_length():
    """``split`` checks ``packed.shape[dim] == total`` before slicing.

    A sequence length that disagrees with the layout would otherwise return
    silently truncated views instead of failing.
    """
    layout = small_layout()
    with pytest.raises(ValueError, match="!= total"):
        layout.split(torch.zeros(2, layout.total + 1, 3), dim=1)


def test_segments_must_be_contiguous():
    """Construction rejects gapped, duplicated and empty segment tables.

    These three are the whole of the segment invariant: segments are pairwise
    disjoint, their union is exactly ``[0, total)``, and even a single-stream
    model has one segment.
    """
    with pytest.raises(ValueError, match="does not join the previous stop"):
        TokenLayout(segments=(SegmentRange("a", 0, 4),
                              SegmentRange("b", 5, 9)), labels={})
    with pytest.raises(ValueError, match="duplicate segment name"):
        TokenLayout(segments=(SegmentRange("a", 0, 4),
                              SegmentRange("a", 4, 8)), labels={})
    with pytest.raises(ValueError, match="must not be empty"):
        TokenLayout(segments=(), labels={})


def test_labels_reject_batch_dim():
    """A label tensor with a batch dimension is rejected at construction.

    Per-sample information belongs on the other channel,
    ``AttentionPlan.key_valid``; keeping labels batch free is what lets the
    structural mask be built once and compiled.
    """
    with pytest.raises(ValueError, match="ndim must be 1"):
        TokenLayout(segments=(SegmentRange("a", 0, 4),),
                    labels={"frame": torch.zeros(2, 4, dtype=torch.long)})


def test_explicit_token_counts_must_match_geometry():
    """Explicit ``token_counts`` may size later streams but not stream 0.

    Stream 0 is the video trunk, whose length has exactly one source, the
    geometry's ``padded_seq_len`` (INV-2: single source of static shapes). The
    table must also cover the stream list exactly, with no missing or stray
    names.
    """
    geo = small_geometry()
    streams = [_Stream("video"), _Stream("action")]
    layout = TokenLayout.from_geometry(
        geo, streams, token_counts={"video": geo.padded_seq_len, "action": 7})
    assert layout.total == geo.padded_seq_len + 7
    # Stream 0's length has only one source, the geometry (INV-2)
    with pytest.raises(ValueError, match="padded_seq_len"):
        TokenLayout.from_geometry(geo, streams,
                                  token_counts={"video": 3, "action": 4})
    with pytest.raises(ValueError, match="must cover the stream list exactly"):
        TokenLayout.from_geometry(geo, streams,
                                  token_counts={"video": geo.padded_seq_len})


def test_default_count_needs_action_horizon():
    """Defaulting a non-video stream to ``action_horizon`` needs it to be > 0.

    With ``action_horizon=0`` the geometry has declared it has no action
    stream, so the default convention cannot size one and asks for explicit
    ``token_counts``. A video-only layout stays legal: it is the N=1
    degenerate case, not a special case.
    """
    geo = small_geometry(action_horizon=0, action_dim=0)
    with pytest.raises(ValueError, match="action_horizon=0"):
        TokenLayout.from_geometry(geo, [_Stream("video"), _Stream("action")])
    # A single stream (no action stream) stays legal: N=1 degenerate, not a
    # special case
    single = TokenLayout.from_geometry(geo, [_Stream("video")])
    assert single.stream_names == ("video",)


def test_label_views_broadcast_shapes():
    """``label_views`` hands rules pre-broadcast query ``[L, 1]`` / key ``[1, L]``.

    Rules are written as elementwise expressions such as
    ``q.get("frame") >= k.get("frame")``. If each rule did its own unsqueeze, a
    reversed one would not raise; it would just invert the causal direction.
    """
    layout = small_layout()
    q, k = layout.label_views()
    assert q.get("frame").shape == (layout.total, 1)
    assert k.get("frame").shape == (1, layout.total)
    assert q.positions().shape == (layout.total, 1)
    assert k.positions().shape == (1, layout.total)
    assert q.has("frame") and not q.has("noise_level")
