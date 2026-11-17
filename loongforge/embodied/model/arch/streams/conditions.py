# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 条件注入：``InjectionKind`` / ``Condition`` / ``ConditionBus``（Lane B）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.4 为准，
``ConditionSpec`` 的可配性以 ``mot_spec_config.zh-CN.md`` §3.5 为准。

本模块要修的是三家框架共有的缺口之一：**「第 N 个条件输入」与「第 N 个输出头」
被混为一谈**。条件走这条总线，头走 ``heads/``，两者再无交叉。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from torch import Tensor

from ..spec.core import Spec

__all__ = ["InjectionKind", "Condition", "ConditionBus", "ConditionSpec"]


class InjectionKind(Enum):
    """条件怎么进模型。四种方式**物理上不同**，不是同一件事的参数。"""

    CROSS_ATTENTION = "cross_attention"   # 作为 cross-attn 的 KV
    ADD_TO_MODULATION = "add_to_modulation"   # 加到调制向量上（timestep 式）
    CONCAT_TOKENS = "concat_tokens"       # 拼成额外 token 进联合序列
    CHANNEL_CONCAT = "channel_concat"     # 沿通道维拼到残差上（入口处一次）


@dataclass(frozen=True)
class Condition:
    """一个已物化的条件。``name`` 是它在总线上的键。

    ``targets`` 是**流名集合**：同一个条件可以同时注入多条流，
    也可以只注入一条。空集合是非法的 —— 没有目标的条件是死配置，
    现状里这类死分支靠读代码才能发现。
    """

    name: str
    kind: InjectionKind
    value: Tensor
    targets: frozenset[str]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(
                f"Condition({self.name!r}).targets 不得为空："
                "没有注入目标的条件是死配置"
            )

    def goes_to(self, stream: str) -> bool:
        return stream in self.targets


@dataclass
class ConditionBus:
    """条件总线。``WorldModel.forward`` 第②步装它，流模块按名取。

    为什么是总线而不是 forward 的若干个形参：条件的**条数**逐模型不同
    （VPP2 两条、EWAM 三条、单体模型零条），而形参表是签名的一部分。
    用总线之后，加一条条件不动任何签名。
    """

    conditions: dict[str, Condition] = field(default_factory=dict)

    def add(self, condition: Condition) -> None:
        """登记一条条件。重名直接抛错 —— 同名覆盖是静默的数据丢失。"""
        if condition.name in self.conditions:
            raise ValueError(
                f"ConditionBus 已有条件 {condition.name!r}；"
                "同名条件不许覆盖（一个名字一个源头）"
            )
        self.conditions[condition.name] = condition

    def for_stream(self, stream: str) -> dict[str, Condition]:
        """取注入到 ``stream`` 的全部条件。没有 ⇒ 空字典，不报错。

        空字典是**单体模型的常态**：它一条条件也没有，而流模块里
        「遍历我的条件」这段代码对它恰好是空循环，无需分支。
        """
        return {name: cond for name, cond in self.conditions.items()
                if cond.goes_to(stream)}

    def of_kind(self, stream: str, kind: InjectionKind) -> dict[str, Condition]:
        """按注入方式再筛一层。流模块的四段注入逻辑各自取自己那一类。"""
        return {name: cond for name, cond in self.for_stream(stream).items()
                if cond.kind is kind}

    def declared_names(self) -> frozenset[str]:
        """已登记的条件名集合。V7 用它与各流的 ``cross_kv_slots`` 对账。"""
        return frozenset(self.conditions)

    def require(self, name: str) -> Condition:
        """按名取，缺失即抛错并列出全部已登记的名字。"""
        try:
            return self.conditions[name]
        except KeyError:
            raise KeyError(
                f"ConditionBus 没有条件 {name!r}；"
                f"已登记：{sorted(self.conditions)}"
            ) from None

    def __len__(self) -> int:
        return len(self.conditions)

    def __iter__(self) -> Iterator[Condition]:
        return iter(self.conditions.values())


@dataclass(frozen=True)
class ConditionSpec:
    """一条条件的声明。构造期的东西，``forward`` 期据它产出 ``Condition``。

    字段的可配性（spec_config §3.5）：``enabled`` 与 ``dropout_prob``
    标 [YAML]（调参的人会反复扫），其余是拓扑。
    """

    name: str
    kind: InjectionKind
    targets: tuple[str, ...]            # 注入的目标流名。拓扑，不进 YAML
    encoder: Spec | None = None         # 产出 value 的模块；None ⇒ 由 batch 直接给
    source_key: str | None = None       # 从 batch 的哪个键取原料。
                                        # encoder 为 None 时必填
    enabled: bool = True                # [YAML] 关掉这条条件
    dropout_prob: float = 0.0           # [YAML] CFG 用的条件丢弃概率

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(
                f"ConditionSpec({self.name!r}).targets 不得为空"
            )
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(
                f"ConditionSpec({self.name!r}).targets 有重复：{self.targets}"
            )
        if self.encoder is None and self.source_key is None:
            raise ValueError(
                f"ConditionSpec({self.name!r})：encoder 与 source_key 不得同时为 "
                "None —— 否则这条条件的 value 无处可来"
            )
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError(
                f"ConditionSpec({self.name!r}).dropout_prob 必须落在 [0, 1)，"
                f"得到 {self.dropout_prob}"
            )

    def materialize(self, value: Tensor) -> Condition:
        """把一个已算好的张量包成 ``Condition``。声明 → 实例的唯一通道。"""
        return Condition(name=self.name, kind=self.kind, value=value,
                         targets=frozenset(self.targets))
