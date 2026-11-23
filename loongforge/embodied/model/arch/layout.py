# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 layout: label table and segment table of the concatenated sequence (Lane A).

Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.2.
``labels_of`` is the entirety of interface I2: Lane C's per-frame timestep and
``BroadcastSpec.axes`` may only align through the labels defined here, never
by guessing an axis from a tensor shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:  # Annotation only: layout must not depend on L0/L2 at runtime
    from .geometry import GeometrySpec
    from .streams.spec import StreamSpec

__all__ = ["SegmentRange", "TokenLayout", "LabelView"]


@dataclass(frozen=True)
class SegmentRange:
    """The half-open interval [start, stop) a stream occupies after concat.

    The name was changed from the earlier ``Segment`` to keep it distinct from
    the rejected EasyWAM ``AttentionSegment`` interval-algebra design: this
    class **records positions only** and carries no visibility semantics.

    Attributes:
        stream: Name of the owning stream, same namespace as ``StreamSpec.name``.
        start: First index, inclusive.
        stop: End index, exclusive.
    """

    stream: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.stream:
            raise ValueError("SegmentRange.stream must not be empty")
        if self.start < 0 or self.stop < self.start:
            raise ValueError(
                f"SegmentRange({self.stream!r}): invalid interval "
                f"[{self.start}, {self.stop})"
            )

    @property
    def length(self) -> int:
        """``stop - start``. Derived, because a stored field could disagree."""
        return self.stop - self.start


@dataclass(frozen=True)
class TokenLayout:
    """Label table plus segment table of the concatenated sequence; carries I2.

    Why an **open label dict** instead of a closed enum: when a new visibility
    rule arrives, the label it needs is usually not in the existing enum
    (``noise_level`` entered exactly that way). An open dict means "add a
    rule" does not mean "edit this class".

    Default semantics (important): a missing label means **the rules that
    depend on it are unavailable**, not an error. For a model that never
    attaches ``noise_level``, the ``noise_gate`` rule simply returns all True
    rather than raising, which lets one visibility expression be reused across
    models.

    Invariants:
      - every label tensor's first dimension equals total, and carries **no
        batch dimension**;
      - segments are pairwise disjoint and their union is exactly [0, total);
      - segment order equals the actual concat order (this is what Lane B's
        C8 check verifies).

    Attributes:
        segments: Segments in **actual concat order**, same order as
            ``AttentionGroup.streams`` (C8 checks this).
        labels: Label name to a [total] tensor, **batch independent**.
            frame / row / col / is_padding come from L0's
            ``GeometrySpec.build_labels()``; stream_id is appended by each stream
            via ``with_labels``; chunk is attached by the action stream, and
            noise_level by L3 when it adds noise. Batch-dependent validity
            travels on the other channel (``AttentionPlan.key_valid``).
    """

    segments: tuple[SegmentRange, ...]
    labels: Mapping[str, Tensor]
    regions: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(
                "TokenLayout.segments must not be empty (a single-stream "
                "model still has one segment)"
            )
        cursor = 0
        seen: set[str] = set()
        for seg in self.segments:
            if seg.stream in seen:
                raise ValueError(
                    f"TokenLayout: duplicate segment name {seg.stream!r}"
                )
            seen.add(seg.stream)
            if seg.start != cursor:
                raise ValueError(
                    f"TokenLayout: segment {seg.stream!r} has "
                    f"start={seg.start}, which does not join the previous "
                    f"stop={cursor} (segments must cover [0, total) without "
                    f"overlap)"
                )
            cursor = seg.stop
        for key, tensor in self.labels.items():
            if tensor.ndim != 1:
                raise ValueError(
                    f"TokenLayout.labels[{key!r}]: ndim must be 1 (labels "
                    f"carry no batch dimension; per-sample information "
                    f"travels on AttentionPlan.key_valid), got "
                    f"{tuple(tensor.shape)}"
                )
            if tensor.shape[0] != cursor:
                raise ValueError(
                    f"TokenLayout.labels[{key!r}]: length {tensor.shape[0]} "
                    f"!= total {cursor}"
                )
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        for name, membership in self.regions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("TokenLayout region names must be non-empty strings")
            if name in seen:
                raise ValueError(f"TokenLayout region {name!r} conflicts with a stream name")
            if (not isinstance(membership, Tensor) or membership.dtype != torch.bool
                    or membership.ndim != 1 or membership.shape[0] != cursor):
                raise ValueError(f"TokenLayout.regions[{name!r}] must be bool [{cursor}]")
        object.__setattr__(self, "regions", MappingProxyType(dict(self.regions)))

    # ---- Queries ----

    @property
    def total(self) -> int:
        """Total token count after concatenation."""
        return self.segments[-1].stop

    @property
    def stream_names(self) -> tuple[str, ...]:
        """Stream names as a tuple, in concat order."""
        return tuple(seg.stream for seg in self.segments)

    def has_label(self, key: str) -> bool:
        """Whether the label exists; rules use it to degrade when it is absent."""
        return key in self.labels

    def labels_of(self, key: str) -> Tensor:
        """Return the [total] tensor of one label (interface I2).

        Raises ``KeyError`` listing the labels that do exist, so callers are
        expected to check ``has_label`` first. Lane C **must** fail when a
        label is missing instead of guessing the axis from a shape.
        """
        try:
            return self.labels[key]
        except KeyError:
            raise KeyError(
                f"TokenLayout has no label {key!r}; existing labels: "
                f"{sorted(self.labels)}"
            ) from None

    def range_of(self, stream: str) -> SegmentRange:
        """Return one stream's interval; the only basis for Lane B's split_by_layout."""
        for seg in self.segments:
            if seg.stream == stream:
                return seg
        raise KeyError(
            f"TokenLayout has no stream {stream!r}; existing streams: "
            f"{list(self.stream_names)}"
        )

    # ---- Transforms ----

    def split(self, packed: Tensor, *, dim: int = 1,
              streams: Sequence[str] | None = None) -> dict[str, Tensor]:
        """Slice a concatenated tensor back into a per-stream dict.

        This replaces the hand-written start/end cursors each model used to
        carry. When ``streams`` is None every segment is sliced; when given,
        only those segments are (the engine needs this on layers where just a
        subset of streams participates).
        """
        if packed.shape[dim] != self.total:
            raise ValueError(
                f"TokenLayout.split: packed.shape[{dim}]={packed.shape[dim]} "
                f"!= total {self.total}"
            )
        wanted = self.stream_names if streams is None else tuple(streams)
        out: dict[str, Tensor] = {}
        for name in wanted:
            seg = self.range_of(name)
            out[name] = packed.narrow(dim, seg.start, seg.length)
        return out

    def with_labels(self, **extra: Tensor) -> TokenLayout:
        """Return a new object with extra labels appended (this class is frozen).

        Used when a stream attaches its stream label.

        A name collision **raises instead of overwriting**: labels come from
        different owners (geometry / stream / L3 noising), and once overwriting
        is allowed the same label has two sources, with the later write winning
        silently.
        """
        clash = sorted(set(extra) & set(self.labels))
        if clash:
            raise ValueError(
                f"TokenLayout.with_labels: labels already exist {clash}; "
                f"one label may have only one source"
            )
        return TokenLayout(segments=self.segments,
                           labels={**self.labels, **extra}, regions=self.regions)

    def with_regions(self, **extra: Tensor) -> TokenLayout:
        """Attach named bool[total] groups without changing labels or streams.

        Read-only groups may overlap. Query partition validation belongs to the
        topology compiler, which knows the owning stream of each query region.
        """
        clash = sorted(set(extra) & set(self.regions))
        if clash:
            raise ValueError(f"TokenLayout.with_regions: regions already exist {clash}")
        return TokenLayout(self.segments, self.labels, {**self.regions, **extra})

    def label_views(self) -> tuple[LabelView, LabelView]:
        """Return the (query side, key side) label views handed to ``Predicate``."""
        return LabelView(self, "query"), LabelView(self, "key")

    @classmethod
    def from_geometry(cls, geometry: GeometrySpec,
                      streams: Sequence[StreamSpec], *,
                      token_counts: Mapping[str, int] | None = None
                      ) -> TokenLayout:
        """Materialize the layout from L0 geometry and the stream list.

        This is step 2 of ``WorldModel.__init__``.

        Only ``.name`` is read from ``streams``, and their order *is* the
        concat order (same order as ``AttentionGroup.streams`` -- exactly what
        C8 checks).

        ``token_counts`` gives the per-stream token count. When omitted the
        default convention applies: stream 0 is the one the geometry describes
        (the video backbone) and gets ``padded_seq_len``; every other stream
        gets ``action_horizon``. It is deliberately not a ``StreamSpec`` field
        because "how many tokens does this stream have" is decided by the
        geometry, not declared by the stream; two declarations would inevitably
        diverge.

        Labels attached here:
          stream_id  [total] long  numeric segment index, the position in ``stream_names``
          frame   [total] long  video tokens take the geometric frame index;
                                action tokens take
                                ``step // action_video_freq_ratio``, i.e. the
                                latent frame they align to
          row/col [total] long  video tokens take the geometric patch row and
                                column; action tokens get -1
          is_padding [total] bool  video tokens take the geometric alignment padding
                                flag; action tokens get False
          chunk   [total] long  chunk index of action tokens; video tokens -1

        ``noise_level`` is **not** attached here: it is the noise-bucket label
        appended by L3 through ``with_labels`` when it adds noise
        (interfaces §3.2).
        """
        names = [stream.name for stream in streams]
        if not names:
            raise ValueError("TokenLayout.from_geometry: streams must not be empty")
        counts = _resolve_token_counts(geometry, names, token_counts)
        segments: list[SegmentRange] = []
        cursor = 0
        for name in names:
            segments.append(SegmentRange(stream=name, start=cursor,
                                         stop=cursor + counts[name]))
            cursor += counts[name]
        return cls(segments=tuple(segments),
                   labels=_build_joint_labels(geometry, tuple(segments)))


@dataclass(frozen=True)
class LabelView:
    """The **one-sided** label view a predicate sees (query side or key side).

    Why it exists: every rule is written as an elementwise expression such as
    ``q.get("frame") >= k.get("frame")``, which requires both sides to already
    be reshaped into broadcastable shapes. If each rule did its own unsqueeze,
    the one that gets it backwards would not fail -- it would silently reverse
    the causal direction.

    Attributes:
        layout: The layout whose labels are being viewed.
        side: Either ``"query"`` or ``"key"``.
    """

    layout: TokenLayout
    side: Literal["query", "key"]

    def __post_init__(self) -> None:
        if self.side not in ("query", "key"):
            raise ValueError(
                f"LabelView.side must be 'query' or 'key', got {self.side!r}"
            )

    def has(self, key: str) -> bool:
        """Whether the label exists; a rule degrades to all True when it does not."""
        return self.layout.has_label(key)

    def get(self, key: str) -> Tensor:
        """Return the label reshaped to broadcast: [L, 1] on query, [1, L] on key."""
        flat = self.layout.labels_of(key)
        return flat.unsqueeze(1) if self.side == "query" else flat.unsqueeze(0)

    def positions(self) -> Tensor:
        """Token indices on this side, shaped like ``get``.

        Rules such as ``prefix_only`` that read no label and only look at
        position use this instead of their own arange plus unsqueeze (getting
        that backwards does not fail, it just moves the prefix to the other
        side).
        """
        device = next(iter(self.layout.labels.values())).device if self.layout.labels else None
        flat = torch.arange(self.layout.total, dtype=torch.long, device=device)
        return flat.unsqueeze(1) if self.side == "query" else flat.unsqueeze(0)


def _resolve_token_counts(geometry: GeometrySpec, names: Sequence[str],
                          given: Mapping[str, int] | None) -> dict[str, int]:
    """Per-stream token counts.

    An explicit table wins; otherwise the default convention documented on
    ``from_geometry`` applies.
    """
    if given is not None:
        missing = sorted(set(names) - set(given))
        stray = sorted(set(given) - set(names))
        if missing or stray:
            raise ValueError(
                f"TokenLayout.from_geometry: token_counts must cover the "
                f"stream list exactly {list(names)}; missing {missing}, "
                f"extra {stray}"
            )
        for name in names:
            if given[name] <= 0:
                raise ValueError(
                    f"TokenLayout.from_geometry: stream {name!r} token count "
                    f"must be positive, got {given[name]}"
                )
        if given[names[0]] != geometry.padded_seq_len:
            raise ValueError(
                f"TokenLayout.from_geometry: stream 0 {names[0]!r} has token "
                f"count {given[names[0]]} != geometry.padded_seq_len "
                f"{geometry.padded_seq_len}. The video backbone length has "
                "geometry as its only source (INV-2: the single source of "
                "static shape)"
            )
        return dict(given)
    counts = {names[0]: geometry.padded_seq_len}
    for name in names[1:]:
        if geometry.action_horizon <= 0:
            raise ValueError(
                f"TokenLayout.from_geometry: stream {name!r} would take "
                "action_horizon tokens by the default convention, but "
                "geometry.action_horizon=0 (this geometry declares it has no "
                "action stream). Pass token_counts explicitly"
            )
        counts[name] = geometry.action_horizon
    return counts


def _build_joint_labels(geometry: GeometrySpec,
                        segments: tuple[SegmentRange, ...]
                        ) -> dict[str, Tensor]:
    """Spread the geometric labels across the concatenated sequence.

    Also attaches the stream and chunk labels.

    The fill value is deliberately -1 rather than 0: if an action token's
    ``row`` were 0, a spatial-locality rule would treat it as "row 0" and take
    effect silently. -1 matches no real row index.
    """
    total = segments[-1].stop
    frame = torch.full((total,), -1, dtype=torch.long)
    row = torch.full((total,), -1, dtype=torch.long)
    col = torch.full((total,), -1, dtype=torch.long)
    chunk = torch.full((total,), -1, dtype=torch.long)
    pad = torch.zeros(total, dtype=torch.bool)
    stream = torch.empty(total, dtype=torch.long)
    for code, seg in enumerate(segments):
        stream[seg.start:seg.stop] = code

    video = segments[0]
    geo = geometry.build_labels()
    sl = slice(video.start, video.stop)
    frame[sl] = geo["frame"]
    row[sl] = geo["row"]
    col[sl] = geo["col"]
    pad[sl] = geo["is_padding"]

    ratio = geometry.action_video_freq_ratio
    for seg in segments[1:]:
        step = torch.arange(seg.length, dtype=torch.long)
        # An action token's frame and chunk are the same number: the chunk
        # index is the latent frame it aligns to. Two labels rather than one,
        # because frame_causal and chunk_causal are two separate rules, and in
        # a three-stream model (motus) action chunks and latent frames are no
        # longer one-to-one.
        aligned = step // ratio
        chunk[seg.start:seg.stop] = aligned
        frame[seg.start:seg.stop] = aligned
    return {"stream_id": stream, "frame": frame, "row": row, "col": col,
            "is_padding": pad, "chunk": chunk}
