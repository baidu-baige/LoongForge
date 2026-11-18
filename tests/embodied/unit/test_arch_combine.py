# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 combination algebra: the semantics of ``& | ~`` and ``restrict`` (Lane A).

The difference between ``restrict`` and ``& where`` is the main subject here:
``restrict`` **leaves the region where ``where`` does not hold unchanged**, while
``&`` turns that region invisible. Confusing the two does not change any shape,
it just silently widens one rule's scope to the whole mask.
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.visibility.combine import (
    ALWAYS_VISIBLE, Visibility, from_predicate)
from loongforge.embodied.model.arch.visibility.rules import (
    frame_causal, prefix_only, same_stream, stream_edge)


def layout() -> TokenLayout:
    """Six tokens: 4 video then 2 action, over two frames."""
    return TokenLayout(
        segments=(SegmentRange("video", 0, 4), SegmentRange("action", 4, 6)),
        labels={"stream": torch.tensor([0, 0, 0, 0, 1, 1]),
                "frame": torch.tensor([0, 0, 1, 1, 0, 1])})


def test_and_or_invert_are_elementwise():
    """``& | ~`` on a ``Visibility`` are the elementwise operators on the
    evaluated masks, so composing rules never reorders or broadcasts."""
    lay = layout()
    a = same_stream().evaluate(lay)
    b = frame_causal().evaluate(lay)
    assert torch.equal((same_stream() & frame_causal()).evaluate(lay), a & b)
    assert torch.equal((same_stream() | frame_causal()).evaluate(lay), a | b)
    assert torch.equal((~same_stream()).evaluate(lay), ~a)


def test_expr_records_the_whole_tree():
    """``expr`` renders the full rule tree, which is what gets logged and
    diffed when a model's visibility changes."""
    vis = same_stream() | (stream_edge("video", "action") & prefix_only(2))
    assert vis.expr == \
        "(same_stream() | (stream_edge('video', 'action') & prefix_only(2)))"
    assert (~same_stream()).expr == "~same_stream()"
    assert same_stream().restrict(where=ALWAYS_VISIBLE).expr == \
        "same_stream().restrict(always_visible)"


def test_restrict_keeps_unmatched_region_unchanged():
    """``rule.restrict(where=w)`` applies ``rule`` only where ``w`` holds and
    leaves the rest fully visible, unlike ``rule & w`` which zeroes it.

    Both forms agree on the region ``w`` selects, so only the complement tells
    them apart.
    """
    lay = layout()
    is_action = _query_is_stream("action")
    rule = frame_causal(1)
    restricted = rule.restrict(where=is_action).evaluate(lay)
    gated = (rule & is_action).evaluate(lay)
    base = rule.evaluate(lay)
    mask = is_action.evaluate(lay)
    # The two agree on the selected region
    assert torch.equal(restricted[mask], base[mask])
    assert torch.equal(gated[mask], base[mask])
    # Outside it: restrict stays fully visible, ``&`` sets the whole area False
    assert bool(restricted[~mask].all())
    assert not bool(gated[~mask].any())


def test_restrict_by_always_visible_is_identity():
    """Restricting by ``ALWAYS_VISIBLE`` is the identity, which is the base case
    a provider relies on when it has no region to restrict to."""
    lay = layout()
    rule = frame_causal(1)
    assert torch.equal(rule.restrict(where=ALWAYS_VISIBLE).evaluate(lay),
                       rule.evaluate(lay))


def test_evaluate_rejects_non_bool():
    """A predicate returning a non-bool mask is rejected.

    A float mask would survive indexing and silently act as "everything except
    exact zeros is visible".
    """
    @from_predicate("broken_dtype")
    def broken(q, k):
        return torch.zeros((q.layout.total, k.layout.total),
                           dtype=torch.float32)

    with pytest.raises(ValueError, match="must be bool"):
        broken.evaluate(layout())


def test_evaluate_rejects_batch_dim():
    """INV-1: an evaluated mask is exactly 2-D.

    Per-sample masks carrying a batch dimension travel on the separate
    ``key_valid`` channel, so a 3-D result here means a rule mixed the two.
    """

    @from_predicate("broken_shape")
    def broken(q, k):
        return torch.ones((2, q.layout.total, k.layout.total),
                          dtype=torch.bool)

    with pytest.raises(ValueError, match="INV-1"):
        broken.evaluate(layout())


def test_compile_carries_layout_and_backend():
    """A compiled mask keeps the layout and backend it was built for, and the
    dense form is ``[1, 1, 6, 6]`` (broadcast over batch and heads)."""
    lay = layout()
    compiled = same_stream().compile(lay, backend="dense")
    assert compiled.backend == "dense"
    assert compiled.layout is lay
    assert compiled.as_dense().shape == (1, 1, 6, 6)


def _query_is_stream(name: str) -> Visibility:
    """Build the predicate "the query token lies on stream ``name``".

    This is the typical value for ``restrict``'s ``where``: not a tenth atomic
    rule, but a local predicate built in place with ``from_predicate``.
    """

    @from_predicate(f"query_is_stream({name!r})")
    def rule(q, k):
        code = q.layout.stream_names.index(name)
        return (q.get("stream") == code).expand(q.layout.total,
                                                k.layout.total)

    return rule
