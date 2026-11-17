# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 可见性表达式与四个组合运算（Lane A）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.3 为准。

组合运算封闭于四个，且实现**必须是对 bool 张量的逐元素运算** ——
这是 dense 与 FlexAttention score_mod 两种后端能从同一份判定物化出来的前提。
任何非逐元素算子（排序、cumsum、scatter）都会让 score_mod 这条路失效。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..layout import LabelView, TokenLayout

if TYPE_CHECKING:  # 避免 combine ↔ mask 的导入环
    from .mask import CompiledMask

__all__ = ["Predicate", "Visibility", "from_predicate", "ALWAYS_VISIBLE"]


# 判定函数的类型：给定 query 侧与 key 侧的标签视图，返回 [Lq, Lk] 的 bool。
# 关键设计：它是**逐 token 标签上的谓词**，不是区间运算。
# 选标签组合而非区间代数的全部理由：区间代数里「噪声水平」没有槽位可放。
Predicate = Callable[[LabelView, LabelView], Tensor]


@dataclass(frozen=True)
class Visibility:
    """可见性表达式。本身不含张量，只在 ``compile`` 时物化。

    ``expr`` 只用于错误消息与测试可读性：组合出来的表达式要能在
    V9 协商失败时告诉人「是哪一条规则让这个后端表达不了」。
    """

    predicate: Predicate = field(repr=False)
    expr: str = "<anonymous>"

    # ---- 四个组合运算，全部逐元素 ----

    def __and__(self, other: Visibility) -> Visibility:
        """两条规则同时成立才可见。"""
        return Visibility(
            lambda q, k: self.predicate(q, k) & other.predicate(q, k),
            f"({self.expr} & {other.expr})",
        )

    def __or__(self, other: Visibility) -> Visibility:
        """任一条成立即可见。"""
        return Visibility(
            lambda q, k: self.predicate(q, k) | other.predicate(q, k),
            f"({self.expr} | {other.expr})",
        )

    def __invert__(self) -> Visibility:
        """取反。"""
        return Visibility(
            lambda q, k: ~self.predicate(q, k),
            f"~{self.expr}",
        )

    def restrict(self, *, where: Visibility) -> Visibility:
        """只在 ``where`` 命中的 (q, kv) 区域上应用本规则，未命中区域**保持原样**。

        与 ``& where`` 的区别：``&`` 会把未命中区域置为不可见，
        ``restrict`` 不改变它。这条语义是 T7 组合律测试的对象。
        """
        return Visibility(
            lambda q, k: (~where.predicate(q, k)) | self.predicate(q, k),
            f"{self.expr}.restrict({where.expr})",
        )

    # ---- 物化 ----

    def evaluate(self, layout: TokenLayout) -> Tensor:
        """在给定布局上求值，得到 [total, total] 的 bool 张量。

        这里是「组合是闭的」这件事的实现：``&`` / ``|`` / ``~`` / ``restrict``
        都只是把两个谓词合成一个谓词，求值时一次性走到底，
        中间不物化任何张量。

        硬不变量（总纲 INV-1）：结果恰为 2 维、不含 batch 维。
        逐样本有效位走 ``AttentionPlan.key_valid`` 那条通道。
        """
        q_view, k_view = layout.label_views()
        out = self.predicate(q_view, k_view)
        expected = (layout.total, layout.total)
        if out.dtype != torch.bool:
            raise ValueError(
                f"可见性 {self.expr} 求值结果 dtype={out.dtype}，必须是 bool"
            )
        if tuple(out.shape) != expected:
            raise ValueError(
                f"可见性 {self.expr} 求值结果形状 {tuple(out.shape)} "
                f"!= {expected}（structural 不含 batch 维，INV-1）"
            )
        return out

    def compile(self, layout: TokenLayout, *, backend: str) -> CompiledMask:
        """在给定布局上物化。``backend`` 决定走哪条物化路径。"""
        from .mask import CompiledMask

        return CompiledMask(structural=self.evaluate(layout),
                            layout=layout, backend=backend)


def from_predicate(expr: str) -> Callable[[Predicate], Visibility]:
    """把一个判定函数包成 ``Visibility``，供 ``rules.py`` 的九条规则使用。"""

    def wrap(predicate: Predicate) -> Visibility:
        return Visibility(predicate, expr)

    return wrap


ALWAYS_VISIBLE = Visibility(
    lambda q, k: torch.ones((q.layout.total, k.layout.total), dtype=torch.bool),
    "always_visible",
)
"""全可见。标签缺省时规则退化成它，而**不是**抛异常（§3.2 的缺省语义）。"""
