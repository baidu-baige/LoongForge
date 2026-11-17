# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 九条基本规则的真值表（Lane A）。

每条规则两件事必须成立：① 标签齐备时的真值表；
② **标签缺省时退化为全 True 而不是报错**（§3.2 的缺省语义）。
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.visibility.rules import (
    block_coupling, chunk_causal, frame_causal, history, noise_gate,
    prefix_only, same_sample, same_stream, stream_edge)


def layout_with(**labels: torch.Tensor) -> TokenLayout:
    """两段 × 3 token 的六位布局，标签由调用方按需贴。"""
    return TokenLayout(
        segments=(SegmentRange("video", 0, 3), SegmentRange("action", 3, 6)),
        labels=labels)


def bare() -> TokenLayout:
    """不带任何标签的布局 —— 所有读标签的规则都应在它上面退化为全 True。"""
    return layout_with()


def long_(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def test_same_stream_truth_table():
    layout = layout_with(stream=long_(0, 0, 0, 1, 1, 1))
    out = same_stream().evaluate(layout)
    assert out[0].tolist() == [True, True, True, False, False, False]
    assert out[3].tolist() == [False, False, False, True, True, True]


def test_stream_edge_is_directed():
    layout = layout_with(stream=long_(0, 0, 0, 1, 1, 1))
    out = stream_edge("video", "action").evaluate(layout)
    # action 的 query 行看得见 video 的 key 列
    assert out[3].tolist() == [True, True, True, False, False, False]
    # 反向边不被蕴含 —— 「VL 流只读」正是靠有入边无出边表达的
    assert out[0].tolist() == [False] * 6


def test_stream_edge_unknown_name_raises():
    layout = layout_with(stream=long_(0, 0, 0, 1, 1, 1))
    with pytest.raises(KeyError, match="布局里没有流"):
        stream_edge("video", "typo").evaluate(layout)


def test_chunk_causal_treats_missing_chunk_as_visible():
    layout = layout_with(chunk=long_(-1, -1, -1, 0, 1, 2))
    out = chunk_causal().evaluate(layout)
    # 视频 token 贴 -1，两侧都 -1 ⇒ 可见；所以它必须与 same_stream() 合用
    assert out[0].tolist() == [True, True, True, False, False, False]
    assert out[4].tolist() == [True, True, True, True, True, False]


def test_frame_causal_window_is_inclusive_length():
    layout = layout_with(frame=long_(0, 1, 2, 3, 4, 5))
    unbounded = frame_causal().evaluate(layout)
    assert unbounded[2].tolist() == [True, True, True, False, False, False]
    # window=1 即「只看本帧」⇒ 判定是 0 <= q - k < window
    assert frame_causal(1).evaluate(layout).tolist() == \
        torch.eye(6, dtype=torch.bool).tolist()
    assert frame_causal(2).evaluate(layout)[3].tolist() == \
        [False, False, True, True, False, False]


def test_frame_causal_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="窗口必须为正"):
        frame_causal(0)


def test_prefix_only_is_on_key_side():
    layout = layout_with(frame=long_(0, 0, 0, 1, 1, 1))
    out = prefix_only(2).evaluate(layout)
    assert out.shape == (6, 6)
    for row in range(6):
        assert out[row].tolist() == [True, True, False, False, False, False]
    # n=0 合法：它是把一条跨流边关掉的写法
    assert not prefix_only(0).evaluate(layout).any()
    with pytest.raises(ValueError, match="不得为负"):
        prefix_only(-1)


def test_noise_gate_reads_key_side_only():
    layout = layout_with(noise_level=long_(0, 0, 1, 1, 2, 2))
    out = noise_gate(0).evaluate(layout)
    # 「谁能被看见」是 key 的属性 ⇒ 整**列**被遮，不是整行
    for row in range(6):
        assert out[row].tolist() == [True, True, False, False, False, False]
    assert noise_gate(2).evaluate(layout).all()


def test_same_sample_isolates_packed_samples():
    layout = layout_with(sample=long_(0, 0, 1, 1, 2, 2))
    out = same_sample().evaluate(layout)
    assert out[0].tolist() == [True, True, False, False, False, False]
    assert out[5].tolist() == [False, False, False, False, True, True]


def test_history_policies():
    layout = layout_with(role=long_(0, 0, 0, 1, 1, 1),
                         frame=long_(0, 1, 1, 2, 2, 2))
    assert history("all").evaluate(layout).all()
    # none：历史列全部不可见
    none_out = history("none").evaluate(layout)
    assert none_out[0].tolist() == [False, False, False, True, True, True]
    # last_only：历史里只留帧号最大的那一帧（frame==1 的两列）
    last_out = history("last_only").evaluate(layout)
    assert last_out[0].tolist() == [False, True, True, True, True, True]


def test_history_last_only_needs_frame_label():
    layout = layout_with(role=long_(0, 0, 0, 1, 1, 1))
    with pytest.raises(KeyError, match="还要读 frame 标签"):
        history("last_only").evaluate(layout)


def test_history_rejects_unknown_policy():
    with pytest.raises(ValueError, match="非法"):
        history("latest")


def test_block_coupling_only_touches_same_block():
    layout = layout_with(block=long_(0, 0, 0, 1, 1, 1))
    assert block_coupling("full").evaluate(layout).all()
    diag = block_coupling("diagonal").evaluate(layout)
    assert diag[0].tolist() == [True, False, False, True, True, True]
    none_out = block_coupling("none").evaluate(layout)
    assert none_out[0].tolist() == [False, False, False, True, True, True]
    with pytest.raises(ValueError, match="非法"):
        block_coupling("partial")


# 标签缺省 ⇒ 规则退化为全 True（不是报错）。这条让同一份可见性表达式跨模型复用。
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
    out = vis.evaluate(bare())
    assert out.dtype == torch.bool, name
    assert out.shape == (6, 6), name
    assert bool(out.all()), name
