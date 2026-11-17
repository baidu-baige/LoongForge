# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 存量接入：``StreamRegistry`` / ``StreamAdapter`` / ``plan_from_legacy_mask``。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.7 为准。

本模块是**迁移期的脚手架**，不是长期契约。它让存量模型（现成的 MoT block、
现成的 [B, 1, L, L] mask）先跑起来，再逐个换成原生实现。
三个入口各自对应一条迁移路径，全部走完之后本文件可以删掉。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ..layout import TokenLayout
from ..visibility.mask import AttentionPlan
from .spec import StreamSpec
from .state import QKV, StreamState

__all__ = ["StreamRegistry", "STREAMS", "StreamAdapter", "plan_from_legacy_mask"]


class StreamRegistry:
    """流实现的注册表。键是实现名，值是产出 ``StreamModule`` 的工厂。

    与 ``spec/core.py`` 的 provider 注册表是两件事：
    那个注册的是**整模型**（键 = ``model_type``），这个注册的是**一条流的 block**。
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]],
                                              Callable[..., Any]]:
        """注册一个流实现。重名直接抛错，不静默覆盖。"""

        def wrap(factory: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._factories:
                prior = self._factories[name]
                raise ValueError(
                    f"流实现重名：{name!r} 已由 {prior.__module__}."
                    f"{prior.__qualname__} 注册"
                )
            self._factories[name] = factory
            return factory

        return wrap

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._factories[name]
        except KeyError:
            raise KeyError(
                f"没有流实现 {name!r}；已注册：{sorted(self._factories)}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


STREAMS = StreamRegistry()
"""进程级流实现注册表。provider 从这里取 block 工厂。"""


class StreamAdapter:
    """把一个**存量** block 包成 ``StreamModule`` 的四段协议。

    存量 block 的形态是「一个 forward 干完整层」，而协议要求
    ``pre_attn`` / ``post_attn`` 之间能插入联合注意力。适配器因此必须
    知道怎么把存量 forward 切成两半 —— 这件事**不能通用地做**，
    所以本类只定签名，每个存量模型给一个自己的子类。

    迁移期的唯一用途：让存量权重零改动地跑在新引擎上，
    用它与原实现做数值对齐，再把 block 换成原生实现。
    """

    def __init__(self, legacy: Any, spec: StreamSpec) -> None:
        self.legacy = legacy
        self.spec = spec

    def prepare(self, state: StreamState, *,
                timestep: Tensor | None) -> StreamState:
        raise NotImplementedError(
            f"StreamAdapter.prepare 需由 {type(self).__name__} 的存量模型子类实现："
            "存量 block 的调制计算位置逐模型不同，无法通用切分"
        )

    def pre_attn(self, state: StreamState, *, local_layer_id: int) -> QKV:
        raise NotImplementedError(
            "StreamAdapter.pre_attn 需由存量模型子类实现"
        )

    def post_attn(self, state: StreamState, attn_out: Tensor, *,
                  local_layer_id: int) -> StreamState:
        raise NotImplementedError(
            "StreamAdapter.post_attn 需由存量模型子类实现"
        )

    def finalize(self, state: StreamState) -> StreamState:
        """缺省不做事。存量 block 的最终 norm 多在模型顶层而不在 block 里。"""
        return state


def plan_from_legacy_mask(mask: Tensor, layout: TokenLayout) -> AttentionPlan:
    """把存量的布尔 mask 拆成 ``AttentionPlan`` 的两个通道。

    接受 ``[L, L]`` / ``[1, L, L]`` / ``[1, 1, L, L]``：这三种都是
    batch 共享的，直接 squeeze 成 structural。

    ``[B, ...]`` 且 B > 1 的 mask **拒绝**而不是降级：它违反 INV-1，
    而静默塞进 structural 会让 BlockMask 复用失效、整条路退 dense。
    现状里 VPP2 的 clean-conditioning 正是这种 mask，它的正解是
    ``visibility.rules.noise_gate``，不是搬进来。
    """
    if mask.dtype != torch.bool:
        raise ValueError(
            f"plan_from_legacy_mask: mask.dtype={mask.dtype}，必须是 bool。"
            "加性 float bias 请先转成布尔可见性"
        )
    squeezed = mask
    while squeezed.ndim > 2 and squeezed.shape[0] == 1:
        squeezed = squeezed.squeeze(0)
    if squeezed.ndim != 2:
        raise ValueError(
            f"plan_from_legacy_mask: 形状 {tuple(mask.shape)} 带 batch 维，"
            "违反 INV-1。逐样本可见性请走 AttentionPlan.key_valid，"
            "依赖噪声水平的请走 visibility.rules.noise_gate"
        )
    expected = (layout.total, layout.total)
    if tuple(squeezed.shape) != expected:
        raise ValueError(
            f"plan_from_legacy_mask: squeeze 后形状 {tuple(squeezed.shape)} "
            f"!= {expected}（layout.total={layout.total}）"
        )
    return AttentionPlan(structural=squeezed, layout=layout)
