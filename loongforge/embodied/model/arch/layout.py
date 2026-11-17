# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 布局：拼接后序列的标签表与段表（Lane A）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.2 为准。
``labels_of`` 是交界面 I2 的全部内容：Lane C 的逐帧 timestep 与
``BroadcastSpec.axes`` 只许由这里的标签对齐，不许按 shape 猜轴。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from torch import Tensor

__all__ = ["SegmentRange", "TokenLayout", "LabelView"]


@dataclass(frozen=True)
class SegmentRange:
    """一条流在拼接后序列里占的半开区间 [start, stop)。

    名字从早先的 ``Segment`` 改过来，是为了和被否决的 EasyWAM
    ``AttentionSegment`` 区间代数方案区分：本类**只记位置**，
    不承载任何可见性语义。
    """

    stream: str             # 段所属的流名，与 StreamSpec.name 同一套名字
    start: int              # 起点，含
    stop: int               # 终点，不含

    def __post_init__(self) -> None:
        if not self.stream:
            raise ValueError("SegmentRange.stream 不得为空")
        if self.start < 0 or self.stop < self.start:
            raise ValueError(
                f"SegmentRange({self.stream!r}): 区间非法 "
                f"[{self.start}, {self.stop})"
            )

    @property
    def length(self) -> int:
        """``stop - start``。派生量 —— 存成字段就能配出 length 与区间不符的三元组。"""
        return self.stop - self.start


@dataclass(frozen=True)
class TokenLayout:
    """拼接后序列的标签表 + 段表。I2 的载体。

    为什么是**开放标签字典**而不是封闭枚举：新增一条可见性规则时，
    它需要的标签往往是现有枚举里没有的（``noise_level`` 就是这样进来的）。
    开放字典让「加一条规则」不必改本类。

    缺省语义（重要）：某个标签不存在，表示**依赖它的规则不可用**，
    不是报错。例如没贴 ``noise_level`` 的模型，``noise_gate`` 规则直接返回
    全 True，而不是抛异常 —— 这让同一份可见性表达式可以跨模型复用。

    不变量：
      - 所有标签张量第一维长度 == total，且**不含 batch 维**；
      - segments 两两不重叠、并集恰好覆盖 [0, total)；
      - segments 的顺序 == 实际 concat 顺序（这是 Lane B 的 C8 校验对象）。
    """

    segments: tuple[SegmentRange, ...]  # 按**实际 concat 顺序**，与
                                        # AttentionGroup.members 同序（C8 校验这件事）
    labels: Mapping[str, Tensor]        # 标签名 → [total] 张量，**batch 无关**。
                                        # frame / row / col / pad 由 L0 的
                                        # GeometrySpec.build_labels() 产出；
                                        # stream 由各流 with_labels 追加，
                                        # chunk 由动作流贴，noise_level 由 L3 加噪时贴。
                                        # batch 相关的有效性走另一条通道
                                        # （AttentionPlan.key_valid）

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("TokenLayout.segments 不得为空（单流模型也有一个段）")
        cursor = 0
        seen: set[str] = set()
        for seg in self.segments:
            if seg.stream in seen:
                raise ValueError(f"TokenLayout: 段名重复 {seg.stream!r}")
            seen.add(seg.stream)
            if seg.start != cursor:
                raise ValueError(
                    f"TokenLayout: 段 {seg.stream!r} 的 start={seg.start} "
                    f"与上一段的 stop={cursor} 不衔接（段必须覆盖 [0, total) 且不重叠）"
                )
            cursor = seg.stop
        for key, tensor in self.labels.items():
            if tensor.ndim != 1:
                raise ValueError(
                    f"TokenLayout.labels[{key!r}]: ndim 必须为 1（标签不含 batch 维，"
                    f"逐样本信息走 AttentionPlan.key_valid），得到 {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != cursor:
                raise ValueError(
                    f"TokenLayout.labels[{key!r}]: 长度 {tensor.shape[0]} "
                    f"!= total {cursor}"
                )
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))

    # ---- 查询 ----

    @property
    def total(self) -> int:
        """拼接后的总 token 数。"""
        return self.segments[-1].stop

    @property
    def stream_names(self) -> tuple[str, ...]:
        """按 concat 顺序给出的流名元组。"""
        return tuple(seg.stream for seg in self.segments)

    def has_label(self, key: str) -> bool:
        """标签是否存在。规则库用它实现「缺省则规则不可用」。"""
        return key in self.labels

    def labels_of(self, key: str) -> Tensor:
        """取某个标签的 [total] 张量（交界面 I2）。

        不存在时抛 ``KeyError`` 并列出已有标签 —— 调用方应先 ``has_label``。
        Lane C 在标签缺省时**必须报错**，不许按 shape 猜轴。
        """
        try:
            return self.labels[key]
        except KeyError:
            raise KeyError(
                f"TokenLayout 没有标签 {key!r}；已有标签："
                f"{sorted(self.labels)}"
            ) from None

    def range_of(self, stream: str) -> SegmentRange:
        """取某条流的区间。Lane B 的 split_by_layout 的唯一依据。"""
        for seg in self.segments:
            if seg.stream == stream:
                return seg
        raise KeyError(
            f"TokenLayout 没有流 {stream!r}；已有流：{list(self.stream_names)}"
        )

    # ---- 变换 ----

    def split(self, packed: Tensor, *, dim: int = 1,
              streams: Sequence[str] | None = None) -> dict[str, Tensor]:
        """按段把拼接张量切回逐流字典。

        替代各模型手写的 start/end 游标。``streams`` 为 None 时切全部段；
        给定时只切这些段（引擎在某些层只有部分流参与时用）。
        """
        if packed.shape[dim] != self.total:
            raise ValueError(
                f"TokenLayout.split: packed.shape[{dim}]={packed.shape[dim]} "
                f"!= total {self.total}"
            )
        wanted = self.stream_names if streams is None else tuple(streams)
        out: dict[str, Tensor] = {}
        for name in wanted:
            seg = self.range_of(name)
            out[name] = packed.narrow(dim, seg.start, seg.length)
        return out

    def with_labels(self, **extra: Tensor) -> TokenLayout:
        """返回追加了标签的新对象（本类不可变）。各流贴 stream 标签时用。

        同名标签**抛错而不是覆盖**：标签的来源不同（几何 / 流 / L3 加噪），
        一旦允许覆盖，同一个标签就会有两个源头，而后写的那个会静默赢。
        """
        clash = sorted(set(extra) & set(self.labels))
        if clash:
            raise ValueError(
                f"TokenLayout.with_labels: 标签已存在 {clash}；"
                f"同一个标签只许有一个源头"
            )
        return TokenLayout(segments=self.segments,
                           labels={**self.labels, **extra})

    def label_views(self) -> tuple[LabelView, LabelView]:
        """产出 (query 侧, key 侧) 两个标签视图，交给 ``Predicate``。"""
        return LabelView(self, "query"), LabelView(self, "key")

    @classmethod
    def from_geometry(cls, geometry, streams) -> TokenLayout:
        """由 L0 几何与流清单物化布局（``WorldModel.__init__`` 第②步）。

        Lane A 实现：按 ``streams`` 的顺序给每条流算 token 数、拼 segments、
        把 ``geometry.build_labels()`` 的四个标签按段铺开、给动作流贴 chunk。
        """
        raise NotImplementedError(
            "TokenLayout.from_geometry 属 Lane A（L1），见 "
            "docs/design/mot_l0l1_geometry_visibility.detailed.zh-CN.md"
        )


@dataclass(frozen=True)
class LabelView:
    """判定函数看到的**单侧**标签视图（query 侧或 key 侧）。

    存在的理由：每条规则都要写成
    ``q.get("frame") >= k.get("frame")`` 这样的逐元素表达式，
    两侧必须已经 reshape 成可广播的形状。让每条规则自己写 unsqueeze，
    写反的那一条不报错 —— 它只是把因果方向弄反。
    """

    layout: TokenLayout
    side: Literal["query", "key"]

    def __post_init__(self) -> None:
        if self.side not in ("query", "key"):
            raise ValueError(
                f"LabelView.side 只能是 'query' 或 'key'，得到 {self.side!r}"
            )

    def has(self, key: str) -> bool:
        """标签是否存在。规则据此退化为全 True。"""
        return self.layout.has_label(key)

    def get(self, key: str) -> Tensor:
        """取标签并 reshape 成可广播形状：query 侧 [L, 1]，key 侧 [1, L]。"""
        flat = self.layout.labels_of(key)
        return flat.unsqueeze(1) if self.side == "query" else flat.unsqueeze(0)
