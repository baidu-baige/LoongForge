# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 存量接入：流实现注册表与存量 mask 的两通道拆分。

``plan_from_legacy_mask`` 的测试重点是**哪些形状被拒绝**：
它的全部价值在于不让一张 ``[B, 1, L, L]`` 的逐样本 mask 悄悄挤进
``structural``。挤进去不报错，只是 BlockMask 的 batch 复用失效、
整条路退回 dense，而 loss 照降。
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.layout import SegmentRange, TokenLayout
from loongforge.embodied.model.arch.streams.registry import (
    StreamRegistry, plan_from_legacy_mask)

TOTAL = 6


def layout() -> TokenLayout:
    return TokenLayout(
        segments=(SegmentRange("video", 0, 4), SegmentRange("action", 4, 6)),
        labels={"stream": torch.tensor([0, 0, 0, 0, 1, 1])})


def test_registry_rejects_duplicate_names():
    registry = StreamRegistry()

    @registry.register("toy")
    def build_toy():  # pragma: no cover - 只看注册行为
        return None

    assert registry.names() == ("toy",)
    assert registry.get("toy") is build_toy
    with pytest.raises(ValueError, match="流实现重名"):
        registry.register("toy")(build_toy)


def test_registry_lists_names_on_miss():
    registry = StreamRegistry()
    registry.register("toy")(lambda: None)
    with pytest.raises(KeyError, match="已注册"):
        registry.get("typo")


@pytest.mark.parametrize("shape", [(TOTAL, TOTAL), (1, TOTAL, TOTAL),
                                   (1, 1, TOTAL, TOTAL)])
def test_batch_shared_shapes_squeeze_into_structural(shape):
    lay = layout()
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[..., 0] = True
    plan = plan_from_legacy_mask(mask, lay)
    assert plan.structural.shape == (TOTAL, TOTAL)
    assert plan.key_valid is None          # 存量 mask 不碰第二通道
    assert plan.structural[:, 0].all()


def test_per_sample_mask_is_rejected_not_degraded():
    lay = layout()
    mask = torch.ones((2, 1, TOTAL, TOTAL), dtype=torch.bool)
    with pytest.raises(ValueError, match="INV-1"):
        plan_from_legacy_mask(mask, lay)


def test_float_bias_is_rejected():
    with pytest.raises(ValueError, match="必须是 bool"):
        plan_from_legacy_mask(torch.zeros((TOTAL, TOTAL)), layout())


def test_shape_must_match_layout_total():
    with pytest.raises(ValueError, match="layout.total=6"):
        plan_from_legacy_mask(torch.ones((8, 8), dtype=torch.bool), layout())
