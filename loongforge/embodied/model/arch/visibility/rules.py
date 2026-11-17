# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 九条基本可见性规则（Lane A）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.3 为准。

每条规则声明它读哪些标签；**标签缺省 ⇒ 该规则退化为 ``ALWAYS_VISIBLE``**，
不是报错 —— 这让同一份可见性表达式可以跨模型复用。

三条实现约定，整个 L1 的性质都挂在它们上面：

1. 判定体**只许逐元素**。排序 / cumsum / scatter 会让 FlexAttention 的
   ``mask_mod`` 这条路失效（combine 模块开头那段的同一条理由）。
   对**标签**本身做归约是允许的（``history`` 要取历史帧的最大帧号）——
   那是构造期的一个标量，不是 [Q, K] 上的算子。
2. 位置一律经 ``LabelView.positions()`` 取，不在规则里自己 arange + unsqueeze：
   两侧 unsqueeze 写反不改变形状，只把因果方向或前缀侧悄悄换掉。
3. 规则读不到标签时返回全 True，**且 dtype 必须是 bool** ——
   ``evaluate`` 会查这两件事。
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..layout import LabelView
from .combine import Visibility, from_predicate

__all__ = [
    "same_stream", "stream_edge", "chunk_causal", "frame_causal",
    "prefix_only", "noise_gate", "same_sample", "history", "block_coupling",
]

_HISTORY_POLICIES = ("all", "last_only", "none")
_BLOCK_MODES = ("full", "diagonal", "none")


def _always(q: LabelView, k: LabelView) -> Tensor:
    """标签缺省时的退化值：全 True。"""
    return torch.ones((q.layout.total, k.layout.total), dtype=torch.bool)


def _stream_code(view: LabelView, name: str, *, rule: str) -> int:
    """流名 → ``stream`` 标签里的编码。不存在即抛错并列出全部流名。"""
    names = view.layout.stream_names
    if name not in names:
        raise KeyError(
            f"visibility.rules.{rule}: 布局里没有流 {name!r}；"
            f"已有流 {list(names)}"
        )
    return names.index(name)


def same_stream() -> Visibility:
    """读 stream。同一条流内部可见。

    **单流模型下恒为 True** ⇒ 整个可见性可以省略（interfaces §2.3）。
    """

    @from_predicate("same_stream()")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("stream"):
            return _always(q, k)
        return q.get("stream") == k.get("stream")

    return rule


def stream_edge(src: str, dst: str) -> Visibility:
    """读 stream。有向跨流边：``dst`` 的 query 可以看见 ``src`` 的 key。

    有向是关键：``stream_edge("video", "action")`` 不蕴含反向边。
    「VL 流只读」正是用「有入边无出边」表达的，而不是靠一个 readonly 布尔。
    **单流模型不使用这条规则**（没有对端流）。

    流名写错时抛 ``KeyError`` 而不是退化为全 False —— 后者会让整条跨流边
    静默消失，而数值上仍然收敛。
    """

    @from_predicate(f"stream_edge({src!r}, {dst!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("stream"):
            return _always(q, k)
        src_code = _stream_code(k, src, rule="stream_edge")
        dst_code = _stream_code(q, dst, rule="stream_edge")
        return (q.get("stream") == dst_code) & (k.get("stream") == src_code)

    return rule


def chunk_causal() -> Visibility:
    """读 chunk。动作块之间的因果：第 i 块只看 ≤ i 的块。

    没有 chunk 标签的 token（视频 token 贴 -1）两侧都取 -1 ⇒ 判定为可见。
    这条规则因此必须与 ``same_stream()`` 合用；单独用它不会把跨流边挡住。
    """

    @from_predicate("chunk_causal()")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("chunk"):
            return _always(q, k)
        return q.get("chunk") >= k.get("chunk")

    return rule


def frame_causal(window: int | None = None) -> Visibility:
    """读 frame。帧因果；``window`` 给定时为滑动窗口（只看最近 window 帧）。

    ``window=1`` 即「只看本帧」。窗口是**含当前帧**的闭区间长度，
    所以判定是 ``0 <= q - k < window``，不是 ``<= window``。
    """
    if window is not None and window <= 0:
        raise ValueError(
            f"frame_causal(window={window}) 非法：窗口必须为正，"
            "或传 None 表示不限窗"
        )

    @from_predicate(f"frame_causal({window})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("frame"):
            return _always(q, k)
        delta = q.get("frame") - k.get("frame")
        causal = delta >= 0
        if window is None:
            return causal
        return causal & (delta < window)

    return rule


def prefix_only(n: int) -> Visibility:
    """不读标签。只看序列最前 n 个 token（固定前缀，n 必须是常量）。

    前缀是**拼接后联合序列**的前缀，不是某条流内部的前缀 —— 因为
    ``prefix_only`` 的两个使用者（fastwam / InternW0-Δ 的动作流看视频条件帧）
    都依赖「视频流是第 0 段」这条布局事实。要限定在某条流内，与
    ``stream_edge(...)`` 合用。

    ``n=0`` 合法，表示「什么都不看」—— 它是把一条跨流边关掉的写法。
    """
    if n < 0:
        raise ValueError(f"prefix_only(n={n}) 非法：n 不得为负")

    @from_predicate(f"prefix_only({n})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        del q
        return (k.positions() < n).expand(k.layout.total, k.layout.total)

    return rule


def noise_gate(threshold: int) -> Visibility:
    """读 noise_level。只看噪声档位 ≤ ``threshold`` 的 key。

    现状无此能力，VPP2 需要它：它的 clean-conditioning 今天只能返回一个
    [B] 选择向量，于是不得不造带 batch 维的 mask（违反 INV-1）。
    档位是**离散档**而不是连续 sigma —— 连续值会让 structural 变成逐样本的。

    这条规则是 12/04 闸门的对象（可见性依赖 ``noise_ids``，唯一硬约束）。

    判定只看 **key 侧**档位：「谁能被看见」是 key 的属性。写成
    ``q <= threshold`` 会变成「哪些 query 行被整行遮掉」，那是另一件事
    （而且两者在对角线上一致，单元测试很容易同时通过）。
    """

    @from_predicate(f"noise_gate({threshold})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not k.has("noise_level"):
            return _always(q, k)
        gate = k.get("noise_level") <= threshold
        return gate.expand(q.layout.total, k.layout.total)

    return rule


def same_sample() -> Visibility:
    """读 sample。packed 变长流的样本隔离。首批无人用，可延后。

    ``sample`` 标签是 batch **无关**的（packed 流把多条样本拼进同一条序列，
    于是「第几条样本」成了 token 的属性而不是张量的一个维度）。
    没有这个标签的模型 —— 也就是所有非 packed 模型 —— 规则退化为全 True。
    """

    @from_predicate("same_sample()")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("sample"):
            return _always(q, k)
        return q.get("sample") == k.get("sample")

    return rule


def history(policy: str) -> Visibility:
    """读 frame / role。历史帧的可见性策略。

    ``role`` 的约定值：``0`` = 历史（已观测），``1`` = 当前/未来（被预测）。
    本规则**只约束指向历史 key 的边**，不碰指向非历史 key 的边 ——
    「历史怎么看」与「未来怎么看」是两条正交的事，合在一条规则里之后，
    想单独放宽未来就只能重写整条表达式。

    三个策略：
      ``all``        历史全可见（= 不加约束）
      ``last_only``  只看历史里帧号最大的那一帧
      ``none``       历史完全不可见（只靠条件通路进模型）
    """
    if policy not in _HISTORY_POLICIES:
        raise ValueError(
            f"history(policy={policy!r}) 非法；合法取值为 "
            f"{list(_HISTORY_POLICIES)}"
        )

    @from_predicate(f"history({policy!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not k.has("role"):
            return _always(q, k)
        is_history = k.get("role") == 0
        if policy == "all":
            return _always(q, k)
        if policy == "none":
            return (~is_history).expand(q.layout.total, k.layout.total)
        if not k.has("frame"):
            raise KeyError(
                "history('last_only') 还要读 frame 标签，而本布局只有 role；"
                f"已有标签 {sorted(k.layout.labels)}"
            )
        # 对**标签**取最大值是构造期的一个标量，不是 [Q, K] 上的归约算子
        frames = k.layout.labels_of("frame")
        roles = k.layout.labels_of("role")
        hist_frames = frames[roles == 0]
        if hist_frames.numel() == 0:
            return _always(q, k)
        last = int(hist_frames.max())
        keep = (~is_history) | (k.get("frame") == last)
        return keep.expand(q.layout.total, k.layout.total)

    return rule


def block_coupling(mode: str) -> Visibility:
    """读 block。当前块内部的耦合方式（全耦合 / 对角 / 不耦合）。

    与 ``chunk_causal`` 的分工：那条管**块之间**的方向，这条管**块内部**
    的形状。因此本规则只约束 ``q.block == k.block`` 的那一片，
    跨块的边保持原样（否则两条规则会互相覆盖，而谁赢取决于书写顺序）。

    三个模式：
      ``full``      块内全耦合（= 不加约束）
      ``diagonal``  块内只看自己
      ``none``      块内互不可见，连自己也不看（纯跨块模型）
    """
    if mode not in _BLOCK_MODES:
        raise ValueError(
            f"block_coupling(mode={mode!r}) 非法；合法取值为 {list(_BLOCK_MODES)}"
        )

    @from_predicate(f"block_coupling({mode!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        if not q.has("block") or mode == "full":
            return _always(q, k)
        same_block = q.get("block") == k.get("block")
        if mode == "none":
            return ~same_block
        return (~same_block) | (q.positions() == k.positions())

    return rule
