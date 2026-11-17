# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 逐流状态与 ``StreamModule`` 协议（Lane B）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.3 为准。

``StreamModule`` 的四个方法把「一层做什么」切成四段，引擎只按顺序调用它们，
**不碰流内部的张量语义**。这是单流退化成 N=1 而不是走另一条代码路径的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor

__all__ = ["StreamState", "QKV", "StreamModule"]


@dataclass
class QKV:
    """一条流在某一层算出来的 Q/K/V。**未拼接**，形状逐流可以不同。

    形状约定：``[B, L, num_heads, head_dim]``。最后两维分开而不是摊平成
    attn_width，是因为联合注意力要求各流 ``num_heads`` 可以不同、
    ``head_dim`` 必须相同（V3）—— 摊平后这条校验就看不见了。
    """

    q: Tensor
    k: Tensor
    v: Tensor

    def __post_init__(self) -> None:
        for name in ("q", "k", "v"):
            tensor = getattr(self, name)
            if tensor.ndim != 4:
                raise ValueError(
                    f"QKV.{name} 必须是 [B, L, num_heads, head_dim] 四维，"
                    f"得到 {tuple(tensor.shape)}"
                )
        if self.k.shape[1] != self.v.shape[1]:
            raise ValueError(
                f"QKV: k 的 token 数 {self.k.shape[1]} != v 的 "
                f"{self.v.shape[1]}"
            )
        if self.q.shape[-1] != self.k.shape[-1]:
            raise ValueError(
                f"QKV: q 的 head_dim {self.q.shape[-1]} != k 的 "
                f"{self.k.shape[-1]}（V3 要求联合成员 head_dim 相等）"
            )

    @property
    def head_dim(self) -> int:
        return int(self.q.shape[-1])

    @property
    def num_query_tokens(self) -> int:
        return int(self.q.shape[1])


@dataclass
class StreamState:
    """一条流在层间传递的全部东西。引擎持有它，流模块读写它。

    ``hidden`` 的宽度是 ``StreamSpec.residual_width``，**不是** attn_width。
    ``extras`` 存流自己要跨层带的东西（例如 RoPE 频率、per-layer 调制缓存），
    引擎不解释它的内容，只负责原样传递。
    """

    stream: str                                  # 流名，与 StreamSpec.name 相等
    hidden: Tensor                               # [B, L, residual_width]
    modulation: Tensor | None = None             # 打包的调制张量，槽序见 spec
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3:
            raise ValueError(
                f"StreamState({self.stream!r}).hidden 必须是 "
                f"[B, L, residual_width] 三维，得到 {tuple(self.hidden.shape)}"
            )

    @property
    def num_tokens(self) -> int:
        """本流的 token 数。C8/C9 用它断言「层前后 token 数不变」。"""
        return int(self.hidden.shape[1])

    @property
    def width(self) -> int:
        return int(self.hidden.shape[-1])

    def replace_hidden(self, hidden: Tensor) -> StreamState:
        """换残差、保留其余字段。要求 token 数不变（C9）。

        引擎每层用它写回输出 —— 用 ``dataclasses.replace`` 不做这条校验，
        所以这里是一个显式方法而不是 replace 的别名。
        """
        if hidden.shape[1] != self.num_tokens:
            raise ValueError(
                f"StreamState({self.stream!r}).replace_hidden：token 数从 "
                f"{self.num_tokens} 变成了 {hidden.shape[1]}；"
                "层不得改变流的 token 数（C9）"
            )
        return StreamState(stream=self.stream, hidden=hidden,
                           modulation=self.modulation, extras=dict(self.extras))


class StreamModule(Protocol):
    """一条流的 block 必须满足的协议。``build(StreamSpec.impl, ...)`` 的产出物。

    四段切分的理由：联合注意力必须在**所有流算完 QKV 之后、
    任何流做 post-attn 之前**发生。把一层写成一个 ``forward`` 就做不到这件事，
    现状正是因此在 MoT 的 forward 里把两条流的代码交织在一起。

    ``local_layer_id`` 是**本流的局部层号**，由
    ``StreamSpec.resolve_layer_index_map`` 在构造期解析。
    流模块永远看不到联合层号 —— 看见了就会有流自己判断「我参不参与」的实现。
    """

    def prepare(self, state: StreamState, *,
                timestep: Tensor | None) -> StreamState:
        """进入第一层之前做一次。调制量的计算、RoPE 频率的准备都在这里。

        ``timestep is None`` 对应 ``StreamSpec.needs_timestep=False``
        （VL 这类纯条件流）。
        """
        ...

    def pre_attn(self, state: StreamState, *, local_layer_id: int) -> QKV:
        """算本层的 Q/K/V。**不做注意力**。"""
        ...

    def post_attn(self, state: StreamState, attn_out: Tensor, *,
                  local_layer_id: int) -> StreamState:
        """吃联合注意力切回来的本流输出，做投影、残差、FFN。

        ``attn_out`` 形状 ``[B, L_self, num_heads, head_dim]``，
        token 数必须等于 ``pre_attn`` 返回的 ``q`` 的 token 数。
        """
        ...

    def finalize(self, state: StreamState) -> StreamState:
        """走完所有层之后做一次。最终 norm 之类。

        **不含输出头** —— 头是 L4 的事（``heads/``），通过
        ``HiddenStateRef`` 取这里的产出物。
        """
        ...
