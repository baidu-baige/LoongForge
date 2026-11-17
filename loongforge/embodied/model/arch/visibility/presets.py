# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 常用规则组合（Lane A）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.6 为准。

可见性是算法，**整条链路上没有一个环节经过 YAML**。规则与组合运算是 Python
表达式，预设只是「这几条组合经常一起出现」的复用函数，调用方只有
``spec/providers/``。三条约定：

1. **预设名不是对外契约**，改名是一次纯重构，不走总纲 §9。
2. **预设不是必须的**。provider 可以直接写 ``same_stream() & frame_causal(w)``；
   只有一个组合被两个以上模型复用时才值得进本文件。
3. **单体模型的 provider 可以整段不设 visibility**，缺省即 ``full_bidirectional``。
"""

from __future__ import annotations

from collections.abc import Callable

from .combine import ALWAYS_VISIBLE, Visibility
from .rules import chunk_causal, frame_causal, noise_gate, prefix_only

__all__ = ["PRESETS", "preset", "get_preset"]

PRESETS: dict[str, Callable[..., Visibility]] = {}


def preset(name: str) -> Callable[[Callable[..., Visibility]],
                                  Callable[..., Visibility]]:
    """注册一个常用组合。重名直接抛错，不静默覆盖。"""

    def wrap(fn: Callable[..., Visibility]) -> Callable[..., Visibility]:
        if name in PRESETS:
            raise ValueError(
                f"可见性预设重名：{name!r} 已由 {PRESETS[name].__module__}."
                f"{PRESETS[name].__qualname__} 注册"
            )
        PRESETS[name] = fn
        return fn

    return wrap


def get_preset(name: str) -> Callable[..., Visibility]:
    """按名取预设；未注册时抛错并列出全部已注册的名字。"""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(
            f"没有可见性预设 {name!r}；已注册：{sorted(PRESETS)}"
        ) from None


# ---- 五个预设。函数体是**真实的组合表达式**，只有被组合的九条规则是 Lane A 的实现 ----
# 形参比 interfaces §3.6 多出来的几个（prefix_tokens / threshold）是组合本身要的：
# 预设看不到几何，首帧有几个 token 只能由 provider 传进来。
# 预设名与形参都不是对外契约，改它们是一次纯重构（§3.6 约定 1）。


@preset("full_bidirectional")
def _full_bidirectional() -> Visibility:
    """全双向。**单体模型与 motus 的默认值**，编译结果是 structural=None。"""
    return ALWAYS_VISIBLE


@preset("first_frame_causal")
def _first_frame_causal(prefix_tokens: int) -> Visibility:
    """帧因果 + 首帧对所有帧可见。"""
    return frame_causal() | prefix_only(prefix_tokens)


@preset("sliding_window_chunk")
def _sliding_window_chunk(window: int) -> Visibility:
    """块因果 + 帧侧滑动窗口。"""
    return chunk_causal() & frame_causal(window)


@preset("clean_conditioned")
def _clean_conditioned(threshold: int) -> Visibility:
    """动作流只看噪声档位 ≤ threshold 的视频 key（VPP2 的用例）。"""
    return noise_gate(threshold)


@preset("noise_gated_history")
def _noise_gated_history(window: int, threshold: int = 0) -> Visibility:
    """最近 window 帧、且噪声档位 ≤ threshold 的历史可见。

    VPP2 provider 写的 ``noise_gated_history(3)`` 就是它：
    动作 query 只读最近 3 帧的 clean 视频 key。
    """
    return frame_causal(window) & noise_gate(threshold)
