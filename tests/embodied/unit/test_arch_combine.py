# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 组合代数：``& | ~`` 与 ``restrict`` 的语义（Lane A）。

``restrict`` 与 ``& where`` 的区别是本文件的主要对象：前者在 ``where``
未命中的区域**保持原样**，后者把它置为不可见。写错不改变形状，
只是把一条规则的作用域悄悄扩大到全图。
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
    return TokenLayout(
        segments=(SegmentRange("video", 0, 4), SegmentRange("action", 4, 6)),
        labels={"stream": torch.tensor([0, 0, 0, 0, 1, 1]),
                "frame": torch.tensor([0, 0, 1, 1, 0, 1])})


def test_and_or_invert_are_elementwise():
    lay = layout()
    a = same_stream().evaluate(lay)
    b = frame_causal().evaluate(lay)
    assert torch.equal((same_stream() & frame_causal()).evaluate(lay), a & b)
    assert torch.equal((same_stream() | frame_causal()).evaluate(lay), a | b)
    assert torch.equal((~same_stream()).evaluate(lay), ~a)


def test_expr_records_the_whole_tree():
    vis = same_stream() | (stream_edge("video", "action") & prefix_only(2))
    assert vis.expr == \
        "(same_stream() | (stream_edge('video', 'action') & prefix_only(2)))"
    assert (~same_stream()).expr == "~same_stream()"
    assert same_stream().restrict(where=ALWAYS_VISIBLE).expr == \
        "same_stream().restrict(always_visible)"


def test_restrict_keeps_unmatched_region_unchanged():
    lay = layout()
    is_action = _query_is_stream("action")
    rule = frame_causal(1)
    restricted = rule.restrict(where=is_action).evaluate(lay)
    gated = (rule & is_action).evaluate(lay)
    base = rule.evaluate(lay)
    mask = is_action.evaluate(lay)
    # 命中区域两者一致
    assert torch.equal(restricted[mask], base[mask])
    assert torch.equal(gated[mask], base[mask])
    # 未命中区域：restrict 保持全可见，& 则整片置 False
    assert bool(restricted[~mask].all())
    assert not bool(gated[~mask].any())


def test_restrict_by_always_visible_is_identity():
    lay = layout()
    rule = frame_causal(1)
    assert torch.equal(rule.restrict(where=ALWAYS_VISIBLE).evaluate(lay),
                       rule.evaluate(lay))


def test_evaluate_rejects_non_bool():
    @from_predicate("broken_dtype")
    def broken(q, k):
        return torch.zeros((q.layout.total, k.layout.total),
                           dtype=torch.float32)

    with pytest.raises(ValueError, match="必须是 bool"):
        broken.evaluate(layout())


def test_evaluate_rejects_batch_dim():
    """INV-1：求值结果恰为 2 维。带 batch 维的 mask 走 key_valid 那条通道。"""

    @from_predicate("broken_shape")
    def broken(q, k):
        return torch.ones((2, q.layout.total, k.layout.total),
                          dtype=torch.bool)

    with pytest.raises(ValueError, match="INV-1"):
        broken.evaluate(layout())


def test_compile_carries_layout_and_backend():
    lay = layout()
    compiled = same_stream().compile(lay, backend="dense")
    assert compiled.backend == "dense"
    assert compiled.layout is lay
    assert compiled.as_dense().shape == (1, 1, 6, 6)


def _query_is_stream(name: str) -> Visibility:
    """「query 落在某条流上」。这是 ``restrict`` 的 where 的典型取值 ——
    它不是第十条原子规则，而是由 ``from_predicate`` 就地构造的局部谓词。"""

    @from_predicate(f"query_is_stream({name!r})")
    def rule(q, k):
        code = q.layout.stream_names.index(name)
        return (q.get("stream") == code).expand(q.layout.total,
                                                k.layout.total)

    return rule
