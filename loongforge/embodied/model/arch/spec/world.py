# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``WorldModelSpec`` 与九条构造期校验 V1–V9，以及 ``build_world_model``。

签名以 ``docs/design/mot_spec_config.zh-CN.md`` §3.2 / §3.8 为准。

本文件是整条配置链路的收口：一个 ``WorldModelSpec`` 构造成功，
就意味着 V1–V9 全过，``WorldModel.__init__`` 不需要再校验任何形状。

**全部校验抛 ``ValueError``，一个 ``assert`` 都没有** ——
``python -O`` 会删掉 assert，而这些校验恰恰是生产环境最需要的那些。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..diffusion.spec import DiffusionSpec
from ..geometry import GeometrySpec
from ..heads.spec import HeadSpec
from ..streams.conditions import ConditionSpec, InjectionKind
from ..streams.engine import AttentionGroup
from ..streams.spec import StreamSpec
from ..visibility.combine import Visibility
from .core import get_provider

__all__ = ["WorldModelSpec", "build_world_model"]

_KNOWN_BACKENDS = frozenset({"dense", "flex", "sdpa", "flash"})
"""已知注意力后端。V9 在 ``CompiledMask.supports`` 交付前按这一条校验。"""


@dataclass(frozen=True)
class WorldModelSpec:
    """一个世界模型的完整声明。provider 的返回类型，``WorldModel`` 的唯一输入。

    ``groups`` 是**注意力组的元组**而不是一个流列表：EWAM 的 VL 流与
    视频流不在同一张联合注意力里，这件事必须在类型里可见。
    单体模型 ⇒ 一个组、组里一条流（N=1 退化，不是特例）。
    """

    model_type: str
    geometry: GeometrySpec
    groups: tuple[AttentionGroup, ...]
    diffusion: DiffusionSpec
    heads: tuple[HeadSpec, ...] = ()
    conditions: tuple[ConditionSpec, ...] = ()
    visibility: Mapping[str, Visibility] = field(default_factory=dict)
    # 键是**流名**，值是该流 query 侧的可见性。缺键 ⇒ 全双向
    # （structural=None，单体模型与存量 motus 的常态）
    attention_backend: str = "dense"    # [YAML] 解析后的具体后端，不接受 "auto"
    attention_block_size: int | None = None
    # 仅 backend="flex" 有意义；它必须与协商时 Scenario.block_size 同值

    def __post_init__(self) -> None:
        self._v1_stream_names()
        self._v2_groups()
        self._v3_joint_widths()
        self._v4_routes()
        self._v5_v6_diffusion()
        self._v7_conditions()
        self._v8_heads()
        self._v9_backend()
        object.__setattr__(self, "visibility",
                           MappingProxyType(dict(self.visibility)))

    # ---- 查询 ----

    @property
    def streams(self) -> tuple[StreamSpec, ...]:
        """全部流，按组序 + 组内序。顺序稳定，供 optimizer 分组使用。"""
        return tuple(spec for group in self.groups for spec in group.members)

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.streams)

    def stream(self, name: str) -> StreamSpec:
        for spec in self.streams:
            if spec.name == name:
                return spec
        raise KeyError(
            f"模型 {self.model_type!r} 没有流 {name!r}；"
            f"有的是 {list(self.stream_names)}"
        )

    def group_of(self, stream: str) -> AttentionGroup:
        """这条流在哪个注意力组里。跨组的流不存在（V1 保证流名全局唯一）。"""
        for group in self.groups:
            if stream in group.stream_names:
                return group
        raise KeyError(
            f"模型 {self.model_type!r} 没有流 {stream!r}；"
            f"有的是 {list(self.stream_names)}"
        )

    def head(self, name: str) -> HeadSpec:
        for spec in self.heads:
            if spec.name == name:
                return spec
        raise KeyError(
            f"模型 {self.model_type!r} 没有输出头 {name!r}；"
            f"有的是 {[h.name for h in self.heads]}"
        )

    def captures(self) -> dict[str, tuple[int, ...]]:
        """由头表推出的引擎抓取请求（I3 请求侧）。声明 → 请求，只推一次。"""
        wanted: dict[str, set[int]] = {}
        for spec in self.heads:
            key = spec.capture_key
            if key is not None:
                wanted.setdefault(key[0], set()).add(key[1])
        return {stream: tuple(sorted(layers))
                for stream, layers in wanted.items()}

    # ---- V1–V9。全部 ValueError，一个 assert 都没有 ----

    def _v1_stream_names(self) -> None:
        """V1：流名非空、全局唯一。

        全局唯一而不是组内唯一：流名同时是 ``TokenLayout`` 的标签值、
        capture hook 的键、optimizer 分组的键。三处共用一个名字空间。
        """
        if not self.groups:
            raise ValueError(
                f"WorldModelSpec({self.model_type!r}).groups 不得为空（V1）"
            )
        names = [spec.name for spec in self.streams]
        if any(not name for name in names):
            raise ValueError(f"流名不得为空字符串（V1）：{names}")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"流名必须全局唯一（V1），重复的是 {duplicates}。"
                "流名同时是布局标签值、capture 键与 optimizer 分组键"
            )

    def _v2_groups(self) -> None:
        """V2：每条流的层对应在其组的联合层数下可解析。

        ``resolve_layer_index_map`` 自带「解析出的参与层数 == num_layers」
        这条断言，所以这里只要调一遍就等于把 STRIDED / AT_LAYERS 的
        边界条件全查了。
        """
        for group in self.groups:
            for spec in group.members:
                spec.resolve_layer_index_map(group.num_layers)

    def _v3_joint_widths(self) -> None:
        """V3：联合成员的 head_dim 相等；``num_heads`` 与 residual_width 可以不等。

        ``AttentionGroup.__post_init__`` 已查过本条，这里重查是**刻意的**：
        provider 可以手搓 ``AttentionGroup``，但也可以从别处拿到一个；
        在 spec 层再查一次让错误消息带上 model_type。
        """
        for group in self.groups:
            head_dims = {spec.name: spec.head_dim for spec in group.members}
            if len(set(head_dims.values())) != 1:
                raise ValueError(
                    f"模型 {self.model_type!r} 的注意力组 "
                    f"{list(group.stream_names)} 内 head_dim 不等（V3）："
                    f"{head_dims}"
                )

    def _v4_routes(self) -> None:
        """V4：KV 路由的端点在**同一组**内，且两侧层号都落在各自流的局部层范围里。

        跨组路由被拒：两个组各有自己的联合注意力，跨组取 KV 等于
        把两张注意力耦合起来，而可见性（L1）看不到这件事。
        ``AttentionGroup.__post_init__`` 已查过端点归属，这里重查层号两侧 ——
        路由的层号是**局部**层号（``KVRoute`` 的约定），
        与联合层号混用是这一处最容易犯的错。
        """
        for group in self.groups:
            for route in group.routes:
                src_spec = group.spec_of(route.src)
                dst_spec = group.spec_of(route.dst)
                for side, layers, spec in (
                        ("src_layers", route.src_layers, src_spec),
                        ("dst_layers", route.dst_layers, dst_spec)):
                    bad = [layer for layer in layers
                           if layer >= spec.num_layers]
                    if bad:
                        raise ValueError(
                            f"KVRoute({route.src!r} → {route.dst!r}).{side} 含 "
                            f"{bad}，超出流 {spec.name!r} 的 num_layers="
                            f"{spec.num_layers}（V4）"
                        )

    def _v5_v6_diffusion(self) -> None:
        """V5：timestep 配置与 ``needs_timestep`` 一一对应。
        V6：每条 loss 的来源（流或头）确实存在。

        V5 替掉的现状行为是「流里拿不到 timestep 就当成 0」——
        那让一次忘配变成静默的无噪声训练，数值上看不出来。
        """
        self.diffusion.validate_against(self.streams)
        head_names = {spec.name for spec in self.heads}
        for loss in self.diffusion.losses:
            if loss.head is not None and loss.head not in head_names:
                raise ValueError(
                    f"LossSpec({loss.name!r}).head={loss.head!r} 不是本模型的头；"
                    f"本模型的头为 {sorted(head_names)}（V6）"
                )

    def _v7_conditions(self) -> None:
        """V7：条件声明与各流的 ``cross_kv_slots`` 双向对账。

        两个方向都查：
          - 条件的 ``targets`` 必须是真实的流名；
          - 走 CROSS_ATTENTION 的条件，其名字必须出现在目标流的
            ``cross_kv_slots`` 里，反之目标流声明的每个槽都必须有条件填。
        这一条是「第 N 个条件输入」与「第 N 个输出头」分家之后的对账点：
        现状靠下标对齐，加一条条件会悄悄串到别的槽上。
        """
        known = set(self.stream_names)
        names = [spec.name for spec in self.conditions]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"ConditionSpec 名字重复（V7）：{duplicates}"
            )
        cross_by_stream: dict[str, set[str]] = {
            name: set() for name in known
        }
        for cond in self.conditions:
            missing = sorted(set(cond.targets) - known)
            if missing:
                raise ValueError(
                    f"ConditionSpec({cond.name!r}).targets 含不存在的流 "
                    f"{missing}；本模型的流为 {sorted(known)}（V7）"
                )
            if cond.kind is InjectionKind.CROSS_ATTENTION and cond.enabled:
                for target in cond.targets:
                    cross_by_stream[target].add(cond.name)
        for spec in self.streams:
            declared = set(spec.cross_kv_slots)
            filled = cross_by_stream[spec.name]
            unfilled = sorted(declared - filled)
            if unfilled:
                raise ValueError(
                    f"流 {spec.name!r} 声明了 cross_kv_slots {unfilled}，"
                    "但没有 CROSS_ATTENTION 条件注入到它们（V7）"
                )
            unused = sorted(filled - declared)
            if unused:
                raise ValueError(
                    f"条件 {unused} 以 CROSS_ATTENTION 注入流 {spec.name!r}，"
                    f"但该流的 cross_kv_slots={list(spec.cross_kv_slots)} "
                    "里没有这些槽（V7）"
                )

    def _v8_heads(self) -> None:
        """V8：头名唯一；头吃的流存在；吃中间层时该层号在那条流的参与层里。

        最后一条是现状最容易出错的地方：取中间层靠 hook，层号写在调用点，
        而「这条流到底参不参与那一层」要读 ``layer_mapping`` 才知道。
        在这里查完之后，``captures()`` 推出来的请求必然是可满足的。
        """
        names = [spec.name for spec in self.heads]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"输出头名字重复（V8）：{duplicates}")
        known = set(self.stream_names)
        for spec in self.heads:
            if spec.source_stream not in known:
                raise ValueError(
                    f"HeadSpec({spec.name!r}).source 指向流 "
                    f"{spec.source_stream!r}，它不是本模型的流；"
                    f"本模型的流为 {sorted(known)}（V8）"
                )
            key = spec.capture_key
            if key is None:
                continue
            stream_name, layer_id = key
            group = self.group_of(stream_name)
            index_map = group.layer_index_maps()[stream_name]
            if layer_id not in index_map:
                raise ValueError(
                    f"HeadSpec({spec.name!r}) 要抓流 {stream_name!r} 的联合层 "
                    f"{layer_id}，但该流在这一层不参与联合注意力；"
                    f"可抓的层为 {sorted(index_map)}（V8）"
                )

    def _v9_backend(self) -> None:
        """V9：注意力后端已注册，且 ``flex`` 必须给 ``attention_block_size``。

        完整形态是 ``CompiledMask.supports(vis, scenario)`` 的三档协商
        （训练 / 推理各协商一次）。它属 Lane A、S4 交付；在那之前
        V9 退化成本方法这两条单项校验 —— 退化的是**严格度**，不是时机：
        校验仍然发生在构造期，而不是跑到推理才 raise。
        """
        if self.attention_backend == "auto":
            raise ValueError(
                f"模型 {self.model_type!r} 的 attention_backend 不接受 'auto'："
                "build 期必须已解析成具体后端（V9）"
            )
        if self.attention_backend not in _KNOWN_BACKENDS:
            raise ValueError(
                f"未知注意力后端 {self.attention_backend!r}；"
                f"已知的是 {sorted(_KNOWN_BACKENDS)}（V9）"
            )
        if self.attention_backend == "flex":
            if self.attention_block_size is None \
                    or self.attention_block_size <= 0:
                raise ValueError(
                    "attention_backend='flex' 必须给出正的 attention_block_size"
                    f"（V9），得到 {self.attention_block_size}。"
                    "它必须与协商时 Scenario.block_size 同值 —— "
                    "换值等于换一张 mask，而 BlockMask 会被 torch.compile 捕获"
                )
        elif self.attention_block_size is not None:
            raise ValueError(
                f"attention_block_size 只在 backend='flex' 时有意义，"
                f"当前 backend={self.attention_backend!r}（V9）"
            )


def build_world_model(config: Any, *,
                      overrides: Mapping[str, Any] | None = None
                      ) -> WorldModelSpec:
    """配置 → ``WorldModelSpec``。四步**有序**，顺序不可换。

    1. 从 ``config.model_type`` 取 provider（``get_provider``）。
       模型身份的唯一出处是启动参数 ``--model-name``，它映射到一份配置，
       配置里的 ``model_type`` 就是这里的键。
    2. 调 provider 产出 ``WorldModelSpec``，拓扑与算法全在 provider 的
       Python 代码里（R5：它们从不进 YAML）。
    3. 施加 YAML 覆写 —— 只允许标 [YAML] 的那些字段。
    4. 构造 ``WorldModelSpec``，此时 V1–V9 一次全跑（在 ``__post_init__`` 里）。

    为什么覆写在 provider **之后**而不是之前：provider 要看到完整的拓扑
    才能算出派生量（联合层数、各流 token 数）。先覆写会让 provider
    对着半成品算。

    为什么第 4 步的校验在 dataclass 的 ``__post_init__`` 而不是这里：
    手搓 ``WorldModelSpec``（测试与 provider 自己）也必须过同一套校验。
    校验写在 build 里就会被绕过。
    """
    model_type = getattr(config, "model_type", None)
    if not model_type:
        raise ValueError(
            "配置里没有 model_type；它是 provider 注册表的键。"
            "模型身份的唯一出处是启动参数 --model-name"
        )
    provider = get_provider(model_type)
    spec = provider(config)
    if not isinstance(spec, WorldModelSpec):
        raise TypeError(
            f"provider(model_type={model_type!r}) 返回了 {type(spec).__name__}，"
            "必须是 WorldModelSpec"
        )
    if overrides:
        raise NotImplementedError(
            "build_world_model 的第③步（YAML 覆写施加）属配置层，"
            "判据见 mot_spec_config.zh-CN.md §3.8 的 R1–R5；"
            f"收到待施加的覆写 {sorted(overrides)}"
        )
    return spec
