# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 隐状态引用 ``HiddenStateRef``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.6 为准。

本模块修的是三家框架共有缺口之二：**「第 N 个条件输入」与「第 N 个输出头」
被混为一谈**。条件走 ``streams/conditions.py``，头取隐状态走这里。

``HiddenStateRef`` 让「这个头吃哪条流的哪一层」成为一个可校验的声明，
而不是现状的 hook + 全局变量 + 散落的层号。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..streams.engine import EngineOutput

__all__ = ["HiddenStateRef", "captures_from_refs"]


@dataclass(frozen=True)
class HiddenStateRef:
    """指向某条流某一层的隐状态。

    ``layer`` 为 None ⇒ 取该流 ``finalize`` 之后的最终输出（最常见）。
    给定整数 ⇒ 取那个**联合层号**的中间隐状态，此时引擎必须被请求抓它
    （``captures``），V8 在构造期对账这件事。

    为什么层号是联合层号而不是局部层号：写这个引用的人（L4 的作者）
    看到的是「第 12 层」这个模型层面的概念，而局部层号是 L2 的内部编号。
    换算由 ``StreamSpec.resolve_layer_index_map`` 负责。
    """

    stream: str
    layer: int | None = None

    def __post_init__(self) -> None:
        if self.layer is not None and self.layer < 0:
            raise ValueError(
                f"HiddenStateRef({self.stream!r}).layer 必须非负或 None，"
                f"得到 {self.layer}"
            )

    @property
    def is_final(self) -> bool:
        """是否取最终输出（而不是某个中间层）。"""
        return self.layer is None

    @property
    def capture_key(self) -> tuple[str, int] | None:
        """需要引擎抓取时的键，取最终输出时为 None。

        键的构造只在这一处 —— ``EngineOutput.captured`` 的键格式由它定义。
        """
        if self.layer is None:
            return None
        return (self.stream, self.layer)

    def resolve(self, output: EngineOutput) -> Tensor:
        """从引擎产出里取出这个隐状态。

        取中间层时走 ``require_capture``，缺失即抛错并列出已抓的键 ——
        「请求了但没抓」是 V8 应该在构造期拦住的，这里是第二道网。
        """
        if self.layer is None:
            return output.require(self.stream).hidden
        return output.require_capture(self.stream, self.layer)


def captures_from_refs(
        refs: tuple[HiddenStateRef, ...]) -> dict[str, tuple[int, ...]]:
    """把一组引用翻成引擎的 ``captures`` 请求（I3 的请求侧）。

    ``WorldModel.__init__`` 第④步调它：头表声明了要哪些隐状态，
    引擎的抓取请求就**从声明推出来**，而不是两处各写一遍。
    """
    wanted: dict[str, list[int]] = {}
    for ref in refs:
        if ref.layer is None:
            continue
        wanted.setdefault(ref.stream, []).append(ref.layer)
    return {stream: tuple(sorted(set(layers)))
            for stream, layers in wanted.items()}
