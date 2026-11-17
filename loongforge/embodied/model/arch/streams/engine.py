# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 联合注意力引擎：``AttentionGroup`` / ``EngineOutput`` / ``StreamEngine``。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.6 为准。

两条硬约定，整个 L2 的性质都挂在它们上面：

1. **引擎不是 ``nn.Module``。** 它不持有参数、不进 state_dict（INV-3）。
   参数全在流模块里。引擎一旦读参数，FSDP2 / ZeRO-3 的 all-gather
   就会在引擎这一层被触发，而引擎是不被 FSDP 包裹的。
2. **单流是 N=1 退化，不是特例。** 引擎里不得出现
   ``if len(members) == 1``。退化只体现在两处数据上：
   ``AttentionPlan.structural is None`` 与 ``routes == ()``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor

from ..layout import TokenLayout
from ..visibility.mask import AttentionPlan, LayerAttentionOverride
from .conditions import ConditionBus
from .routing import KVRoute
from .spec import StreamSpec
from .state import StreamState

__all__ = [
    "AttentionGroup", "EngineOutput", "StreamEngine", "assert_shape_signature",
]


@dataclass(frozen=True)
class AttentionGroup:
    """参与同一张联合注意力的一组流 + 联合层数。

    ``num_layers`` 是**联合层数**，与任何单条流的 ``num_layers`` 都无关 ——
    这正是放宽「三者全等」校验之后需要一个独立数字的地方。
    各流用 ``StreamSpec.resolve_layer_index_map(num_layers)`` 对上号。
    """

    members: tuple[StreamSpec, ...]
    num_layers: int
    routes: tuple[KVRoute, ...] = ()

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("AttentionGroup.members 不得为空")
        if self.num_layers <= 0:
            raise ValueError(
                f"AttentionGroup.num_layers 必须为正，得到 {self.num_layers}"
            )
        names = [spec.name for spec in self.members]
        if len(set(names)) != len(names):
            raise ValueError(f"AttentionGroup.members 有重名流：{names}")
        # V3：联合成员的 head_dim 必须相等。num_heads 与 residual_width 可以不等 ——
        # 这是「放宽 num_layers 三者全等」之后剩下的唯一宽度约束。
        head_dims = {spec.name: spec.head_dim for spec in self.members}
        if len(set(head_dims.values())) != 1:
            raise ValueError(
                f"AttentionGroup 成员的 head_dim 必须相等（V3），得到 {head_dims}"
            )
        known = set(names)
        for route in self.routes:
            missing = {route.src, route.dst} - known
            if missing:
                raise ValueError(
                    f"KVRoute({route.src!r} → {route.dst!r}) 指向了不在本组里的流："
                    f"{sorted(missing)}；本组成员为 {names}"
                )
            if max(route.dst_layers) >= self.spec_of(route.dst).num_layers:
                raise ValueError(
                    f"KVRoute({route.src!r} → {route.dst!r}).dst_layers 最大层号 "
                    f"{max(route.dst_layers)} 超出流 {route.dst!r} 的 num_layers="
                    f"{self.spec_of(route.dst).num_layers}。"
                    "路由的层号是**局部**层号，不是联合层号"
                )

    @property
    def head_dim(self) -> int:
        """联合 head_dim。__post_init__ 已保证全组相等。"""
        return self.members[0].head_dim

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.members)

    def spec_of(self, name: str) -> StreamSpec:
        for spec in self.members:
            if spec.name == name:
                return spec
        raise KeyError(
            f"AttentionGroup 没有流 {name!r}；成员为 {list(self.stream_names)}"
        )

    def layer_index_maps(self) -> dict[str, dict[int, int]]:
        """全组的「联合层号 → 局部层号」表。构造期算一次，forward 期只查。"""
        return {spec.name: spec.resolve_layer_index_map(self.num_layers)
                for spec in self.members}

    def routes_into(self, dst: str) -> tuple[KVRoute, ...]:
        """取以 ``dst`` 为终点的路由。空元组是常态（单流模型恒为空）。"""
        return tuple(route for route in self.routes if route.dst == dst)


@dataclass
class EngineOutput:
    """引擎跑完一组之后的产出。交界面 **I3**（B → C）的载体。

    ``captured`` 的键是 ``(流名, 联合层号)``，供 L3/L4 取中间层隐状态
    （``HiddenStateRef``）。请求侧是 ``captures: Mapping[str, tuple[int, ...]]``，
    即「这条流要抓哪几个联合层」。

    为什么键里带层号：现状取中间层靠的是 hook + 全局变量，
    层号散落在调用点。显式键让「抓了哪层」在类型里可见，
    V8 能在构造期校验请求的层号确实存在。
    """

    states: dict[str, StreamState]
    captured: dict[tuple[str, int], Tensor] = field(default_factory=dict)

    def require(self, stream: str) -> StreamState:
        try:
            return self.states[stream]
        except KeyError:
            raise KeyError(
                f"EngineOutput 没有流 {stream!r} 的状态；"
                f"有的是 {sorted(self.states)}"
            ) from None

    def require_capture(self, stream: str, layer_id: int) -> Tensor:
        """取一个抓取点。缺失时报错并列出全部已抓的键。"""
        key = (stream, layer_id)
        try:
            return self.captured[key]
        except KeyError:
            raise KeyError(
                f"EngineOutput 没有抓取点 {key}；"
                f"已抓：{sorted(self.captured)}"
            ) from None


class StreamEngine(Protocol):
    """联合注意力引擎的协议。``run`` 是 L2 的全部入口。

    调用顺序（interfaces §7，三条不可重排）：
      1. 全部流 ``prepare``；
      2. 逐联合层：全部参与流 ``pre_attn`` → **一次**联合注意力 → 各流
         ``post_attn``；
      3. 全部流 ``finalize``。
    「所有流算完 QKV 再做注意力」是第 2 步里不可重排的那一条。
    """

    def run(self, group: AttentionGroup,
            states: Mapping[str, StreamState],
            modules: Mapping[str, Any],
            plan: AttentionPlan,
            layout: TokenLayout,
            *,
            conditions: ConditionBus,
            timesteps: Mapping[str, Tensor | None],
            captures: Mapping[str, Sequence[int]] | None = None,
            override: LayerAttentionOverride | None = None) -> EngineOutput:
        """跑完一个注意力组的所有联合层。

        ``modules`` 是流名 → ``StreamModule``；``timesteps`` 是流名 →
        timestep（``needs_timestep=False`` 的流给 None）。
        ``override`` 非 None 时，被它接管的层不走拼接 + 一次 attention
        这条路（也因此不能进 CUDA graph 捕获）。
        """
        ...


def assert_shape_signature(before: Mapping[str, StreamState],
                           after: Mapping[str, StreamState]) -> None:
    """C8 / C9：一层前后，流集合与各流 token 数必须完全一致。

    放在模块级而不是引擎内部，是为了让 Lane B 的实现与测试共用同一份判定。
    """
    if set(before) != set(after):
        raise ValueError(
            f"层前后流集合变了：前 {sorted(before)}，后 {sorted(after)}（C8）"
        )
    for name, state in before.items():
        got = after[name].num_tokens
        if got != state.num_tokens:
            raise ValueError(
                f"流 {name!r} 的 token 数从 {state.num_tokens} 变成了 {got}；"
                "层不得改变流的 token 数（C9）"
            )
