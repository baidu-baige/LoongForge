# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 九条基本可见性规则（Lane A）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.3 为准。

每条规则声明它读哪些标签；**标签缺省 ⇒ 该规则退化为 ``ALWAYS_VISIBLE``**，
不是报错 —— 这让同一份可见性表达式可以跨模型复用。
实现属 Lane A，判据见 A 组 10 条等价性测试（T1–T10）。
"""

from __future__ import annotations

from .combine import Visibility

__all__ = [
    "same_stream", "stream_edge", "chunk_causal", "frame_causal",
    "prefix_only", "noise_gate", "same_sample", "history", "block_coupling",
]


def _todo(rule: str) -> Visibility:
    raise NotImplementedError(
        f"visibility.rules.{rule} 属 Lane A（L1 规则库，S2 交付），见 "
        "docs/design/mot_l0l1_geometry_visibility.detailed.zh-CN.md"
    )


def same_stream() -> Visibility:
    """读 stream。同一条流内部可见。

    **单流模型下恒为 True** ⇒ 整个可见性可以省略（interfaces §2.3）。
    """
    return _todo("same_stream")


def stream_edge(src: str, dst: str) -> Visibility:
    """读 stream。有向跨流边：``dst`` 的 query 可以看见 ``src`` 的 key。

    有向是关键：``stream_edge("video", "action")`` 不蕴含反向边。
    「VL 流只读」正是用「有入边无出边」表达的，而不是靠一个 readonly 布尔。
    **单流模型不使用这条规则**（没有对端流）。
    """
    return _todo("stream_edge")


def chunk_causal() -> Visibility:
    """读 chunk。动作块之间的因果：第 i 块只看 ≤ i 的块。"""
    return _todo("chunk_causal")


def frame_causal(window: int | None = None) -> Visibility:
    """读 frame。帧因果；``window`` 给定时为滑动窗口（只看最近 window 帧）。"""
    return _todo("frame_causal")


def prefix_only(n: int) -> Visibility:
    """不读标签。只看序列最前 n 个 token（固定前缀，n 必须是常量）。"""
    return _todo("prefix_only")


def noise_gate(threshold: int) -> Visibility:
    """读 noise_level。只看噪声档位 ≤ ``threshold`` 的 key。

    现状无此能力，VPP2 需要它：它的 clean-conditioning 今天只能返回一个
    [B] 选择向量，于是不得不造带 batch 维的 mask（违反 INV-1）。
    档位是**离散档**而不是连续 sigma —— 连续值会让 structural 变成逐样本的。

    这条规则是 12/04 闸门的对象（可见性依赖 ``noise_ids``，唯一硬约束）。
    """
    return _todo("noise_gate")


def same_sample() -> Visibility:
    """读 sample。packed 变长流的样本隔离。首批无人用，可延后。"""
    return _todo("same_sample")


def history(policy: str) -> Visibility:
    """读 frame / role。历史帧的可见性策略（全可见 / 只看最后一帧 / 不可见）。"""
    return _todo("history")


def block_coupling(mode: str) -> Visibility:
    """读 block。当前块内部的耦合方式（全耦合 / 对角 / 不耦合）。"""
    return _todo("block_coupling")
