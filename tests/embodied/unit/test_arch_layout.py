# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 布局：``TokenLayout`` 的不变量、标签与 ``labels_of``（交界面 I2）。"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.geometry import GeometrySpec
from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout


class _Stream:
    """只取 ``.name`` —— ``from_geometry`` 的全部要求。"""

    def __init__(self, name: str) -> None:
        self.name = name


def small_geometry(**extra) -> GeometrySpec:
    """12 个 token 的小几何：视频 2 帧 × 4 token + 动作 4 步。"""
    kwargs = dict(frame_height=32, frame_width=32, num_frames=5, num_views=1,
                  vae_spatial_stride=16, patch_size=(1, 1, 1),
                  action_horizon=4, action_video_freq_ratio=2)
    kwargs.update(extra)
    return GeometrySpec(**kwargs)


def small_layout() -> TokenLayout:
    return TokenLayout.from_geometry(small_geometry(),
                                     [_Stream("video"), _Stream("action")])


def test_segments_cover_and_order():
    layout = small_layout()
    assert layout.stream_names == ("video", "action")
    assert layout.total == 12
    assert [(s.stream, s.start, s.stop) for s in layout.segments] == [
        ("video", 0, 8), ("action", 8, 12)]
    assert layout.range_of("action").length == 4


def test_labels_are_batch_free_and_aligned():
    layout = small_layout()
    for key in ("stream", "frame", "row", "col", "pad", "chunk"):
        tensor = layout.labels_of(key)
        assert tensor.ndim == 1, key
        assert tensor.shape[0] == layout.total, key
    assert layout.labels_of("stream").tolist() == [0] * 8 + [1] * 4
    # 视频 token 取几何帧号；动作 token 取 step // ratio，即它对齐到的 latent 帧
    assert layout.labels_of("frame").tolist() == \
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    # 动作 token 的 row/col 是 -1 而不是 0 —— 0 会被空间规则当成「第 0 行」
    assert layout.labels_of("row")[8:].tolist() == [-1] * 4
    assert layout.labels_of("chunk").tolist() == [-1] * 8 + [0, 0, 1, 1]


def test_noise_level_not_tagged_by_geometry():
    """``noise_level`` 归 L3 加噪时贴，不由几何产出（interfaces §3.2）。"""
    assert not small_layout().has_label("noise_level")


def test_labels_of_missing_raises_and_lists():
    with pytest.raises(KeyError, match="noise_level"):
        small_layout().labels_of("noise_level")


def test_with_labels_rejects_clash():
    layout = small_layout()
    tier = torch.zeros(layout.total, dtype=torch.long)
    assert layout.with_labels(noise_level=tier).has_label("noise_level")
    with pytest.raises(ValueError, match="只许有一个源头"):
        layout.with_labels(frame=tier)


def test_split_matches_segments():
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
    layout = small_layout()
    with pytest.raises(ValueError, match="!= total"):
        layout.split(torch.zeros(2, layout.total + 1, 3), dim=1)


def test_segments_must_be_contiguous():
    with pytest.raises(ValueError, match="不衔接"):
        TokenLayout(segments=(SegmentRange("a", 0, 4),
                              SegmentRange("b", 5, 9)), labels={})
    with pytest.raises(ValueError, match="段名重复"):
        TokenLayout(segments=(SegmentRange("a", 0, 4),
                              SegmentRange("a", 4, 8)), labels={})
    with pytest.raises(ValueError, match="不得为空"):
        TokenLayout(segments=(), labels={})


def test_labels_reject_batch_dim():
    """标签带 batch 维即非法 —— 逐样本信息走 ``AttentionPlan.key_valid``。"""
    with pytest.raises(ValueError, match="ndim 必须为 1"):
        TokenLayout(segments=(SegmentRange("a", 0, 4),),
                    labels={"frame": torch.zeros(2, 4, dtype=torch.long)})


def test_explicit_token_counts_must_match_geometry():
    geo = small_geometry()
    streams = [_Stream("video"), _Stream("action")]
    layout = TokenLayout.from_geometry(
        geo, streams, token_counts={"video": geo.padded_seq_len, "action": 7})
    assert layout.total == geo.padded_seq_len + 7
    # 第 0 条流的长度只有几何一个出处（INV-2）
    with pytest.raises(ValueError, match="padded_seq_len"):
        TokenLayout.from_geometry(geo, streams,
                                  token_counts={"video": 3, "action": 4})
    with pytest.raises(ValueError, match="必须恰好覆盖流清单"):
        TokenLayout.from_geometry(geo, streams,
                                  token_counts={"video": geo.padded_seq_len})


def test_default_count_needs_action_horizon():
    geo = small_geometry(action_horizon=0)
    with pytest.raises(ValueError, match="action_horizon=0"):
        TokenLayout.from_geometry(geo, [_Stream("video"), _Stream("action")])
    # 单流（无动作流）仍然合法：它是 N=1 退化，不是特例
    single = TokenLayout.from_geometry(geo, [_Stream("video")])
    assert single.stream_names == ("video",)


def test_label_views_broadcast_shapes():
    layout = small_layout()
    q, k = layout.label_views()
    assert q.get("frame").shape == (layout.total, 1)
    assert k.get("frame").shape == (1, layout.total)
    assert q.positions().shape == (layout.total, 1)
    assert k.positions().shape == (1, layout.total)
    assert q.has("frame") and not q.has("noise_level")
