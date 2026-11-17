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

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from ..layout import SegmentRange, TokenLayout
from ..visibility.mask import AttentionPlan, LayerAttentionOverride
from .conditions import ConditionBus
from .routing import KVRoute
from .spec import StreamSpec
from .state import QKV, StreamState

__all__ = [
    "AttentionGroup", "EngineOutput", "StreamEngine", "JointAttentionEngine",
    "assert_shape_signature",
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


def restrict_to_streams(layout: TokenLayout, plan: AttentionPlan,
                        names: tuple[str, ...]
                        ) -> tuple[TokenLayout, AttentionPlan]:
    """把布局与 plan 限制到参与某一层的那几条流上。

    不参与的流被整段抠掉，剩下的段**重新连续编号** —— ``TokenLayout``
    要求段覆盖 ``[0, total)``。两个通道按同一个下标表 index_select，
    于是 structural 的行列、key_valid 的列、以及 frame/chunk/stream 三个标签
    在抠掉之后仍然对得上同一个 token。

    ``names == layout.stream_names`` 时原样返回：这是**首批两个模型的常态**
    （两条流等深、逐层都参与），不该为它付一次 index_select。
    """
    if names == layout.stream_names:
        return layout, plan
    stray = sorted(set(names) - set(layout.stream_names))
    if stray:
        raise KeyError(
            f"restrict_to_streams: 布局里没有流 {stray}；"
            f"已有 {list(layout.stream_names)}"
        )
    keep = [seg for seg in layout.segments if seg.stream in names]
    index = torch.cat([torch.arange(seg.start, seg.stop, dtype=torch.long)
                       for seg in keep])
    segments: list[SegmentRange] = []
    cursor = 0
    for seg in keep:
        segments.append(SegmentRange(stream=seg.stream, start=cursor,
                                     stop=cursor + seg.length))
        cursor += seg.length
    sub_layout = TokenLayout(
        segments=tuple(segments),
        labels={key: value[index] for key, value in layout.labels.items()},
    )
    structural = plan.structural
    if structural is not None:
        idx = index.to(structural.device)
        structural = structural.index_select(0, idx).index_select(1, idx)
    key_valid = plan.key_valid
    if key_valid is not None:
        key_valid = key_valid.index_select(1, index.to(key_valid.device))
    return sub_layout, AttentionPlan(structural=structural,
                                     key_valid=key_valid, layout=sub_layout)


def _joint_attention(packed: QKV, mask: Any, *, backend: str) -> Tensor:
    """一次联合注意力。进出都是 ``[B, L, num_heads, head_dim]``。

    ``QKV`` 把 heads 与 head_dim 分开存，而两条 kernel 都要
    ``[B, num_heads, L, head_dim]`` ⇒ transpose 只在这一个函数里出现。
    散在调用点的 transpose 写漏一次不报错，只是把 head 当成 token 来算。
    """
    q = packed.q.transpose(1, 2)
    k = packed.k.transpose(1, 2)
    v = packed.v.transpose(1, 2)
    if backend == "flex":
        from torch.nn.attention.flex_attention import flex_attention
        out = flex_attention(q, k, v, block_mask=mask)
    elif backend in ("dense", "sdpa"):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        raise ValueError(
            f"_joint_attention: 后端 {backend!r} 没有实现。varlen 要先把 "
            "CompiledMask.as_segments 的段表接上，属 R3（cosmos3）"
        )
    return out.transpose(1, 2)


class JointAttentionEngine:
    """``StreamEngine`` 的实现：cat → 一次联合注意力 → split。

    **不是 nn.Module**（engine 模块开头的约定 1）：不持参数、不进 state_dict。
    **没有 ``if len(members) == 1``**（约定 2）：单流走的是同一条路，
    只是 cat/split 各只有一项、``plan.structural`` 为 None。

    mask 的物化是**逐次 run 一张**，按「本层参与的流集合」缓存在 run 内部的
    局部字典里。不跨 run 缓存的理由：``plan.key_valid`` 是逐 batch 的，
    跨 run 复用会把上一个 batch 的 padding 带进来 —— 这类缺陷在 loss 上
    看不出来。flex 后端下 ``create_block_mask`` 因此每次 forward 跑一遍；
    把它提到构造期要先结清 key_valid 的归属（总纲 §10-5）。
    """

    def __init__(self, *, backend: str = "dense",
                 block_size: int | None = None) -> None:
        if backend not in ("dense", "sdpa", "flex"):
            raise ValueError(
                f"JointAttentionEngine: 后端 {backend!r} 不支持；"
                "可选 'dense' / 'sdpa' / 'flex'"
            )
        if backend == "flex" and block_size is None:
            raise ValueError(
                "JointAttentionEngine(backend='flex') 必须给 block_size —— "
                "它改变 BlockMask 的形状，而 BlockMask 会被 torch.compile 捕获"
            )
        self.backend = backend
        self.block_size = block_size

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
        """跑完一个注意力组的所有联合层。调用顺序见 ``StreamEngine``。"""
        del conditions      # 条件由流模块自己按名从总线取（§4.4）；引擎不解释它
        names = group.stream_names
        for label, got in (("states", set(states)), ("modules", set(modules))):
            if got != set(names):
                raise ValueError(
                    f"JointAttentionEngine.run: {label} 的流集合 {sorted(got)} "
                    f"!= 组成员 {list(names)}"
                )
        if layout.stream_names != names:
            raise ValueError(
                f"JointAttentionEngine.run: 布局的段顺序 "
                f"{list(layout.stream_names)} != 组成员顺序 {list(names)}（C8）"
            )
        index_maps = group.layer_index_maps()
        wanted = _resolve_captures(captures, index_maps)

        # ① 全部流 prepare
        live = {name: modules[name].prepare(states[name],
                                            timestep=timesteps.get(name))
                for name in names}
        assert_shape_signature(states, live)

        # ② 逐联合层：全部参与流 pre_attn → 一次联合注意力 → 各流 post_attn
        masks: dict[tuple[str, ...], tuple[TokenLayout, Any]] = {}
        captured: dict[tuple[str, int], Tensor] = {}
        for layer_id in range(group.num_layers):
            active = tuple(name for name in names
                           if layer_id in index_maps[name])
            if not active:
                raise ValueError(
                    f"联合层 {layer_id} 没有任何流参与 —— 这是一个死层。"
                    f"各流的参与层为 "
                    f"{ {n: sorted(m) for n, m in index_maps.items()} }"
                )
            qkvs = {
                name: modules[name].pre_attn(
                    live[name], local_layer_id=index_maps[name][layer_id])
                for name in active
            }
            if override is not None and override.handles_layer(layer_id):
                outs = override.run_attention(layer_id, qkvs, plan, layout)
                if set(outs) != set(active):
                    raise ValueError(
                        f"LayerAttentionOverride 在层 {layer_id} 返回的流集合 "
                        f"{sorted(outs)} != 参与流 {list(active)}"
                    )
            else:
                outs = self._attend(active, qkvs, plan, layout, masks)
            before = dict(live)
            for name in active:
                live[name] = modules[name].post_attn(
                    live[name], outs[name],
                    local_layer_id=index_maps[name][layer_id])
            assert_shape_signature(before, live)
            for name in active:
                if layer_id in wanted.get(name, ()):
                    captured[(name, layer_id)] = live[name].hidden

        # ③ 全部流 finalize
        final = {name: modules[name].finalize(live[name]) for name in names}
        assert_shape_signature(live, final)
        return EngineOutput(states=final, captured=captured)

    def _attend(self, active: tuple[str, ...],
                qkvs: Mapping[str, QKV],
                plan: AttentionPlan,
                layout: TokenLayout,
                masks: dict[tuple[str, ...], tuple[TokenLayout, Any]]
                ) -> dict[str, Tensor]:
        """拼 QKV → 一次 masked attention → 按段切回逐流输出。"""
        heads = {name: int(qkvs[name].q.shape[2]) for name in active}
        if len(set(heads.values())) != 1:
            raise ValueError(
                f"联合注意力要求参与流的 num_heads 相等（否则沿 token 维拼不起来），"
                f"得到 {heads}。V3 只管住了 head_dim —— num_heads 不等的拓扑"
                "需要逐流独立的注意力，不是一张联合 mask"
            )
        q = torch.cat([qkvs[name].q for name in active], dim=1)
        k = torch.cat([qkvs[name].k for name in active], dim=1)
        v = torch.cat([qkvs[name].v for name in active], dim=1)
        if active not in masks:
            sub_layout, sub_plan = restrict_to_streams(layout, plan, active)
            if sub_plan.structural is not None \
                    and sub_plan.structural.device != q.device:
                sub_plan = AttentionPlan(
                    structural=sub_plan.structural.to(q.device),
                    key_valid=None if sub_plan.key_valid is None
                    else sub_plan.key_valid.to(q.device),
                    layout=sub_layout)
            masks[active] = (
                sub_layout,
                sub_plan.materialize(self.backend, block_size=self.block_size),
            )
        sub_layout, mask = masks[active]
        if q.shape[1] != sub_layout.total:
            raise ValueError(
                f"拼接后 token 数 {q.shape[1]} != 布局 total {sub_layout.total}；"
                f"参与流 {list(active)} 的段长为 "
                f"{[(s.stream, s.length) for s in sub_layout.segments]}"
            )
        out = _joint_attention(QKV(q=q, k=k, v=v), mask, backend=self.backend)
        return sub_layout.split(out, dim=1)


def _resolve_captures(captures: Mapping[str, Sequence[int]] | None,
                      index_maps: Mapping[str, Mapping[int, int]]
                      ) -> dict[str, tuple[int, ...]]:
    """校验抓取请求并归一成 {流名: 联合层号元组}。

    请求了一条流不参与的联合层**抛错**而不是静默跳过：静默跳过的症状是
    ``EngineOutput.require_capture`` 在 L4 那边报 KeyError，
    而那时已经离开了声明这件事的地方（V8 要在构造期拦住它）。
    """
    if not captures:
        return {}
    stray = sorted(set(captures) - set(index_maps))
    if stray:
        raise KeyError(
            f"captures 请求了不在本组里的流 {stray}；"
            f"本组成员为 {sorted(index_maps)}"
        )
    resolved: dict[str, tuple[int, ...]] = {}
    for name, layer_ids in captures.items():
        for layer_id in layer_ids:
            if layer_id not in index_maps[name]:
                raise ValueError(
                    f"captures 请求抓流 {name!r} 的联合层 {layer_id}，"
                    f"但该流不参与这一层；它参与的联合层为 "
                    f"{sorted(index_maps[name])}"
                )
        resolved[name] = tuple(layer_ids)
    return resolved
