# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""装配层里不需要整个 ``WorldModel`` 就能查的两件事（builder.py 的自由函数）。

联合可见性的合成在 ``test_arch_providers.py`` 里按三个真 provider 查；
本文件查条件总线的装配，以及两条**刻意抛错**的路 ——
它们抛错是裁定未落地的记账，不是漏实现，所以也要有测试钉住。
"""

from __future__ import annotations

import pytest
import torch

from loongforge.embodied.model.arch.builder import materialize_conditions
from loongforge.embodied.model.arch.spec.core import Spec
from loongforge.embodied.model.arch.streams.conditions import (
    ConditionSpec, InjectionKind)


def text(**extra) -> ConditionSpec:
    kwargs = dict(name="text", kind=InjectionKind.CROSS_ATTENTION,
                  targets=("video", "action"), source_key="context")
    kwargs.update(extra)
    return ConditionSpec(**kwargs)


def batch(**extra) -> dict:
    out = {"context": torch.zeros((2, 7, 16))}
    out.update(extra)
    return out


def test_batch_sourced_condition_lands_on_its_targets():
    bus = materialize_conditions([text()], batch(), training=True)
    assert len(bus) == 1
    condition = bus.require("text")
    assert condition.kind is InjectionKind.CROSS_ATTENTION
    assert torch.equal(condition.value, batch()["context"])
    assert set(bus.for_stream("video")) == {"text"}
    assert set(bus.for_stream("action")) == {"text"}
    # 不在 targets 里的流取到空字典而不是报错 —— 单体模型的常态
    assert bus.for_stream("vl") == {}


def test_disabled_condition_takes_no_name_on_the_bus():
    """关掉的条件整条跳过。留一个零张量会让 V7 的名字对账把它算进去。"""
    bus = materialize_conditions([text(enabled=False)], batch(), training=True)
    assert len(bus) == 0
    assert bus.declared_names() == frozenset()


def test_missing_source_key_lists_the_batch_keys():
    with pytest.raises(KeyError, match="batch 只有"):
        materialize_conditions([text()], {"video": torch.zeros(1)},
                               training=True)


def test_two_conditions_sharing_a_name_are_rejected():
    with pytest.raises(ValueError, match="不许覆盖"):
        materialize_conditions([text(), text()], batch(), training=True)


def test_encoder_bearing_condition_is_an_open_ruling():
    spec = text(encoder=Spec(impl=torch.nn.Identity))
    with pytest.raises(NotImplementedError, match="建造工位"):
        materialize_conditions([spec], batch(), training=True)


def test_dropout_is_an_open_ruling_only_while_training():
    spec = text(dropout_prob=0.1)
    with pytest.raises(NotImplementedError, match="条件丢弃"):
        materialize_conditions([spec], batch(), training=True)
    # eval 期不丢弃 ⇒ 同一条声明照常物化
    assert len(materialize_conditions([spec], batch(), training=False)) == 1
