# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The ten basic L1 visibility rules (Lane A).

The signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.3.

Every rule declares which labels it reads; **if a label is absent the rule
degrades to ``ALWAYS_VISIBLE``** rather than raising, which is what lets one
visibility expression be reused across models. ``label_edge`` is the single
exception and says why in its own docstring: a two-sided selector has no
meaningful degraded form.

Three implementation conventions; every property of L1 hangs off them:

1. Predicate bodies **may only be elementwise**. Sort / cumsum / scatter kill
   FlexAttention's ``mask_mod`` path (the same reason given at the top of the
   combine module). Reducing over the **labels** themselves is allowed
   (``history`` needs the largest history frame index): that is a
   construction-time scalar, not an operator over [Q, K].
2. Positions always come from ``LabelView.positions()``; rules do not arange +
   unsqueeze on their own. Swapping the two unsqueeze sides does not change the
   shape, it just silently flips the causal direction or the prefix side.
3. When a rule cannot find its labels it returns all True, **and the dtype must
   be bool**: ``evaluate`` checks both of these.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ..layout import LabelView
from .combine import Visibility, from_predicate

__all__ = [
    "same_stream", "stream_edge", "label_edge", "chunk_causal", "frame_causal",
    "prefix_only", "noise_gate", "same_sample", "history", "block_coupling",
]

_HISTORY_POLICIES = ("all", "last_only", "none")
_BLOCK_MODES = ("full", "diagonal", "none")


def _always(q: LabelView, k: LabelView) -> Tensor:
    """Degenerate value used when a label is absent: all True."""
    return torch.ones((q.layout.total, k.layout.total), dtype=torch.bool)


def _stream_code(view: LabelView, name: str, *, rule: str) -> int:
    """Stream name to its code in the ``stream_id`` label.

    Args:
        view: Label view whose layout defines the stream order.
        name: Stream name to look up.
        rule: Rule name, used in the error message.

    Returns:
        The stream's integer code.

    Raises:
        KeyError: If the layout has no such stream; all stream names are listed.
    """
    names = view.layout.stream_names
    if name not in names:
        raise KeyError(
            f"visibility.rules.{rule}: layout has no stream {name!r}; "
            f"existing streams are {list(names)}"
        )
    return names.index(name)


def _in_values(labels: Tensor, values: tuple[int, ...]) -> Tensor:
    """Elementwise test for membership of ``labels`` in a fixed value set.

    The reduction is over the **value set**, not over [Q, K]: ``values`` is
    known at construction time, so this unrolls to ``len(values)`` broadcast
    comparisons and an ``or``, which keeps the ``mask_mod`` path alive
    (convention 1 in the module docstring).

    Args:
        labels: Label slice shaped [L, 1] (query side) or [1, L] (key side).
        values: Accepted label values.

    Returns:
        A bool tensor shaped like ``labels``.
    """
    probe = torch.as_tensor(values, dtype=labels.dtype, device=labels.device)
    return (labels == probe.reshape(-1, *([1] * labels.ndim))).any(0)


def same_stream() -> Visibility:
    """Reads ``stream_id``. Tokens inside the same stream can see each other.

    **Always True for a single-stream model**, so the whole visibility can then
    be omitted (interfaces §2.3).

    Returns:
        The ``same_stream()`` visibility.
    """

    @from_predicate("same_stream()")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("stream_id"):
            return _always(q, k)
        return q.get("stream_id") == k.get("stream_id")

    return rule


def stream_edge(src: str, dst: str) -> Visibility:
    """Reads ``stream_id``. Directed cross-stream edge: ``dst`` queries see ``src`` keys.

    Directedness is the point: ``stream_edge("video", "action")`` does not imply
    the reverse edge. "The VL stream is read-only" is expressed as "has incoming
    edges, no outgoing edges", not by a readonly boolean. **Single-stream models
    do not use this rule** (there is no peer stream).

    A misspelled stream name raises ``KeyError`` instead of degrading to all
    False; the latter would make the whole cross-stream edge vanish silently
    while the numbers still converged.

    Args:
        src: Name of the stream whose keys become visible.
        dst: Name of the stream whose queries get to see them.

    Returns:
        The directed-edge visibility.
    """

    @from_predicate(f"stream_edge({src!r}, {dst!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("stream_id"):
            return _always(q, k)
        src_code = _stream_code(k, src, rule="stream_edge")
        dst_code = _stream_code(q, dst, rule="stream_edge")
        return (q.get("stream_id") == dst_code) & (k.get("stream_id") == src_code)
    return rule


def label_edge(label: str, src: Sequence[int],
               dst: Sequence[int]) -> Visibility:
    """Reads ``label``. Keys labelled ``src`` are visible to queries ``dst``.

    The generalization of ``stream_edge``, which is this rule specialized to
    ``label="stream"`` with single-element value sets. It is the only rule that
    selects on the **query** side alone: ``prefix_only`` / ``noise_gate`` /
    ``history`` test keys only, and ``same_stream`` / ``frame_causal`` /
    ``chunk_causal`` are relational. Without it, "two groups of rows inside one
    stream obey different intra-stream rules" cannot be written down — which is
    exactly what InternW0-Delta's condition rows and noisy-future rows need
    (``frame_attention.py:27-31`` versus ``:35``).

    **A missing label raises instead of degrading to all True.** The other rules
    can degrade because they express a restriction, so losing them only widens
    the mask by one term. This rule is a selector: a visibility written as a
    disjunction of seven ``label_edge`` terms would degrade to a fully visible
    mask without any error. The narrow form of that failure has already
    happened once with ``noise_level`` never being attached (``layout.py:255``).

    Args:
        label: Label both sides are tested against.
        src: Label values whose keys become visible.
        dst: Label values whose queries get to see them.

    Returns:
        The directed label-edge visibility.

    Raises:
        ValueError: If either value set is empty, which would make the term
            dead rather than merely narrow.
    """
    src_values = tuple(int(value) for value in src)
    dst_values = tuple(int(value) for value in dst)
    if not src_values or not dst_values:
        raise ValueError(
            f"visibility.rules.label_edge({label!r}): src and dst must be "
            f"non-empty; got src={list(src_values)} dst={list(dst_values)}"
        )
    expr = f"label_edge({label!r}, {list(src_values)}, {list(dst_values)})"

    @from_predicate(expr)
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has(label):
            raise KeyError(
                f"visibility.rules.label_edge: layout has no label {label!r}; "
                f"existing labels are {sorted(q.layout.labels)}. Unlike the "
                f"other rules this one does not degrade to all True, because "
                f"a selector that degrades opens the mask silently"
            )
        return (_in_values(q.get(label), dst_values)
                & _in_values(k.get(label), src_values))

    return rule


def chunk_causal() -> Visibility:
    """Reads chunk. Causality between action chunks: chunk i sees chunks <= i.

    Tokens with no chunk label (video tokens are labeled -1) have -1 on both
    sides, so the comparison holds and they come out visible. This rule must
    therefore be combined with ``same_stream()``; on its own it will not block
    cross-stream edges.

    Returns:
        The chunk-causal visibility.
    """

    @from_predicate("chunk_causal()")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("chunk"):
            return _always(q, k)
        return q.get("chunk") >= k.get("chunk")

    return rule


def frame_causal(window: int | None = None) -> Visibility:
    """Reads frame. Frame causality; with ``window`` given, a sliding window.

    ``window=1`` means "only the current frame". The window is the length of a
    closed interval **including** the current frame, so the test is
    ``0 <= q - k < window``, not ``<= window``.

    Args:
        window: Number of most recent frames visible, or None for no limit.

    Returns:
        The frame-causal visibility.

    Raises:
        ValueError: If ``window`` is given and not positive.
    """
    if window is not None and window <= 0:
        raise ValueError(
            f"frame_causal(window={window}) is invalid: window must be "
            "positive, or pass None for no window limit"
        )

    @from_predicate(f"frame_causal({window})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("frame"):
            return _always(q, k)
        delta = q.get("frame") - k.get("frame")
        causal = delta >= 0
        if window is None:
            return causal
        return causal & (delta < window)

    return rule


def prefix_only(n: int) -> Visibility:
    """Reads no label. Sees only the first n tokens (fixed prefix, n constant).

    The prefix is a prefix of the **concatenated joint sequence**, not of some
    stream's own tokens, because both users of ``prefix_only`` (fastwam and
    InternW0-Δ, whose action streams look at video condition frames) rely on the
    layout fact that the video stream is segment 0. To confine it to one stream,
    combine with ``stream_edge(...)``.

    ``n=0`` is legal and means "see nothing"; it is how a cross-stream edge gets
    switched off.

    Args:
        n: Number of leading tokens that stay visible.

    Returns:
        The prefix-only visibility.

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError(
            f"prefix_only(n={n}) is invalid: n must not be negative")

    @from_predicate(f"prefix_only({n})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        del q
        return (k.positions() < n).expand(k.layout.total, k.layout.total)

    return rule


def noise_gate(threshold: int) -> Visibility:
    """Reads noise_level. Sees only keys whose noise step is <= ``threshold``.

    Today's code cannot do this and VPP2 needs it: its clean-conditioning can
    currently only return a ``[B]`` selection vector, which forces a mask with a
    batch dimension (violating INV-1). The level is a **discrete step**, not a
    continuous sigma; a continuous value would make the structural channel
    per-sample.

    This rule is the subject of the 12/04 gate (visibility depending on
    ``noise_ids`` is the one hard constraint there).

    The test looks at the **key side** only: "who can be seen" is a property of
    the key. Writing ``q <= threshold`` would instead mean "which query rows get
    masked out entirely", which is a different thing (and the two agree on the
    diagonal, so a unit test can easily pass under both).

    Args:
        threshold: Highest noise step that remains visible.

    Returns:
        The noise-gate visibility.
    """

    @from_predicate(f"noise_gate({threshold})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not k.has("noise_level"):
            return _always(q, k)
        gate = k.get("noise_level") <= threshold
        return gate.expand(q.layout.total, k.layout.total)

    return rule


def same_sample() -> Visibility:
    """Reads ``sample_id``. Sample isolation for packed variable-length streams.

    No first-batch model uses it, so it can wait. The ``sample_id`` label is
    **independent of** the batch dimension: a packed stream concatenates several
    samples into one sequence, so "which sample" becomes a property of the token
    rather than a tensor dimension. Models without that label, meaning every
    non-packed model, see this rule degrade to all True.

    Returns:
        The same-sample visibility.
    """

    @from_predicate("same_sample()")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("sample_id"):
            return _always(q, k)
        return q.get("sample_id") == k.get("sample_id")

    return rule


def history(policy: str) -> Visibility:
    """Reads frame / role. Visibility policy for history frames.

    Agreed values of ``role``: ``0`` = history (already observed), ``1`` =
    current/future (being predicted). This rule **only constrains edges pointing
    at history keys** and leaves edges to non-history keys alone: "how history is
    seen" and "how the future is seen" are two orthogonal things, and once they
    are merged into one rule, loosening only the future means rewriting the whole
    expression.

    The three policies:
      ``all``        history fully visible (= no constraint added)
      ``last_only``  only the history frame with the largest frame index
      ``none``       history entirely invisible (it enters only via conditions)

    Args:
        policy: One of ``all`` / ``last_only`` / ``none``.

    Returns:
        The history visibility.

    Raises:
        ValueError: If ``policy`` is not one of the three.
    """
    if policy not in _HISTORY_POLICIES:
        raise ValueError(
            f"history(policy={policy!r}) is invalid; legal values are "
            f"{list(_HISTORY_POLICIES)}"
        )

    @from_predicate(f"history({policy!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not k.has("role"):
            return _always(q, k)
        is_history = k.get("role") == 0
        if policy == "all":
            return _always(q, k)
        if policy == "none":
            return (~is_history).expand(q.layout.total, k.layout.total)
        if not k.has("frame"):
            raise KeyError(
                "history('last_only') also reads the frame label, but this "
                "layout only has role; "
                f"existing labels are {sorted(k.layout.labels)}"
            )
        # Taking a max over the **labels** is a construction-time scalar,
        # not a reduction operator over [Q, K]
        frames = k.layout.labels_of("frame")
        roles = k.layout.labels_of("role")
        hist_frames = frames[roles == 0]
        if hist_frames.numel() == 0:
            return _always(q, k)
        last = int(hist_frames.max())
        keep = (~is_history) | (k.get("frame") == last)
        return keep.expand(q.layout.total, k.layout.total)

    return rule


def block_coupling(mode: str) -> Visibility:
    """Reads block. How tokens inside the current block couple to each other.

    Split of work against ``chunk_causal``: that rule governs the direction
    **between** blocks, this one governs the shape **inside** a block. So this
    rule only constrains the region where ``q.block == k.block`` and leaves
    cross-block edges as they are; otherwise the two rules would overwrite each
    other and the winner would depend on the order they were written in.

    The three modes:
      ``full``      fully coupled inside a block (= no constraint added)
      ``diagonal``  inside a block, a token sees only itself
      ``none``      inside a block nothing is visible, not even itself
                    (a purely cross-block model)

    Args:
        mode: One of ``full`` / ``diagonal`` / ``none``.

    Returns:
        The block-coupling visibility.

    Raises:
        ValueError: If ``mode`` is not one of the three.
    """
    if mode not in _BLOCK_MODES:
        raise ValueError(
            f"block_coupling(mode={mode!r}) is invalid; legal values are "
            f"{list(_BLOCK_MODES)}"
        )

    @from_predicate(f"block_coupling({mode!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("block") or mode == "full":
            return _always(q, k)
        same_block = q.get("block") == k.get("block")
        if mode == "none":
            return ~same_block
        return (~same_block) | (q.positions() == k.positions())

    return rule
