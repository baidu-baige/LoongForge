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
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:  # 只用于标注：layout 不许在运行期依赖 L0/L2 的具体类
    from .geometry import GeometrySpec
    from .streams.spec import StreamSpec

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
    def from_geometry(cls, geometry: GeometrySpec,
                      streams: Sequence[StreamSpec], *,
                      token_counts: Mapping[str, int] | None = None
                      ) -> TokenLayout:
        """由 L0 几何与流清单物化布局（``WorldModel.__init__`` 第②步）。

        ``streams`` 只读 ``.name``，顺序即 concat 顺序（与
        ``AttentionGroup.members`` 同序 —— C8 校验的正是这件事）。

        ``token_counts`` 给出逐流 token 数。省略时按缺省约定推：第 0 条流是
        几何所描述的那条（视频主干）⇒ ``padded_seq_len``；其余每条流 ⇒
        ``action_horizon``。不把它做成 ``StreamSpec`` 的字段，是因为
        「这条流有多少 token」由几何决定而不由流自己声明 —— 两处声明必然分叉。

        贴出的标签：
          stream  [total] long  段下标，值 = 在 ``stream_names`` 里的位置
          frame   [total] long  视频 token 取几何帧号；动作 token 取
                                ``step // action_video_freq_ratio``，
                                即它对齐到的那个 latent 帧
          row/col [total] long  视频 token 取几何 patch 行列；动作 token 为 -1
          pad     [total] bool  视频 token 取几何对齐填充位；动作 token 为 False
          chunk   [total] long  动作 token 的块号；视频 token 为 -1

        ``noise_level`` **不在这里贴** —— 它是 L3 加噪时经 ``with_labels``
        追加的档位标签（interfaces §3.2）。
        """
        names = [stream.name for stream in streams]
        if not names:
            raise ValueError("TokenLayout.from_geometry: streams 不得为空")
        counts = _resolve_token_counts(geometry, names, token_counts)
        segments: list[SegmentRange] = []
        cursor = 0
        for name in names:
            segments.append(SegmentRange(stream=name, start=cursor,
                                         stop=cursor + counts[name]))
            cursor += counts[name]
        return cls(segments=tuple(segments),
                   labels=_build_joint_labels(geometry, tuple(segments)))


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

    def positions(self) -> Tensor:
        """本侧的 token 下标，形状同 ``get``。``prefix_only`` 这类
        「不读标签、只看位置」的规则用它，而不是自己 arange + unsqueeze
        （写反了不报错，只是把前缀挪到另一侧）。"""
        flat = torch.arange(self.layout.total, dtype=torch.long)
        return flat.unsqueeze(1) if self.side == "query" else flat.unsqueeze(0)


def _resolve_token_counts(geometry: GeometrySpec, names: Sequence[str],
                          given: Mapping[str, int] | None) -> dict[str, int]:
    """逐流 token 数。显式表优先，否则走 ``from_geometry`` 文档里的缺省约定。"""
    if given is not None:
        missing = sorted(set(names) - set(given))
        stray = sorted(set(given) - set(names))
        if missing or stray:
            raise ValueError(
                f"TokenLayout.from_geometry: token_counts 必须恰好覆盖流清单 "
                f"{list(names)}；缺 {missing}、多 {stray}"
            )
        for name in names:
            if given[name] <= 0:
                raise ValueError(
                    f"TokenLayout.from_geometry: 流 {name!r} 的 token 数必须为正，"
                    f"得到 {given[name]}"
                )
        if given[names[0]] != geometry.padded_seq_len:
            raise ValueError(
                f"TokenLayout.from_geometry: 第 0 条流 {names[0]!r} 的 token 数 "
                f"{given[names[0]]} != geometry.padded_seq_len "
                f"{geometry.padded_seq_len}。视频主干的长度只有几何一个出处"
                "（INV-2：静态形状的唯一来源）"
            )
        return dict(given)
    counts = {names[0]: geometry.padded_seq_len}
    for name in names[1:]:
        if geometry.action_horizon <= 0:
            raise ValueError(
                f"TokenLayout.from_geometry: 流 {name!r} 要按缺省约定取 "
                "action_horizon 个 token，但 geometry.action_horizon=0"
                "（该几何声明了自己没有动作流）。请显式传 token_counts"
            )
        counts[name] = geometry.action_horizon
    return counts


def _build_joint_labels(geometry: GeometrySpec,
                        segments: tuple[SegmentRange, ...]
                        ) -> dict[str, Tensor]:
    """把几何标签按段铺到拼接后序列上，并贴 stream / chunk。

    缺省值刻意取 -1 而不是 0：动作 token 的 ``row`` 若填 0，空间局部性规则
    会把它当成「第 0 行」而静默生效。-1 不等于任何真实行号。
    """
    total = segments[-1].stop
    frame = torch.full((total,), -1, dtype=torch.long)
    row = torch.full((total,), -1, dtype=torch.long)
    col = torch.full((total,), -1, dtype=torch.long)
    chunk = torch.full((total,), -1, dtype=torch.long)
    pad = torch.zeros(total, dtype=torch.bool)
    stream = torch.empty(total, dtype=torch.long)
    for code, seg in enumerate(segments):
        stream[seg.start:seg.stop] = code

    video = segments[0]
    geo = geometry.build_labels()
    sl = slice(video.start, video.stop)
    frame[sl] = geo["frame"]
    row[sl] = geo["row"]
    col[sl] = geo["col"]
    pad[sl] = geo["pad"]

    ratio = geometry.action_video_freq_ratio
    for seg in segments[1:]:
        step = torch.arange(seg.length, dtype=torch.long)
        # 动作 token 的 frame 与 chunk 是同一个数：块号就是它对齐到的 latent 帧号。
        # 两个标签而不是一个，是因为 frame_causal 与 chunk_causal 是两条规则，
        # 而三流模型（motus）里动作块与 latent 帧不再一一对应
        aligned = step // ratio
        chunk[seg.start:seg.stop] = aligned
        frame[seg.start:seg.stop] = aligned
    return {"stream": stream, "frame": frame, "row": row, "col": col,
            "pad": pad, "chunk": chunk}
