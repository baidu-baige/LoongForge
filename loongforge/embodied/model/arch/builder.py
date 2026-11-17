# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``WorldModel``：各 spec 之间的协作实现。

签名与步骤以 ``docs/design/mot_spec_model_matrix.zh-CN.md`` §4.5 为准
（``__init__`` 四步 / 训练七步 / 推理三步 / 六个扩展点）。

本类**只做装配与编排**，不含任何模型算法：
  - 形状事实在 ``WorldModelSpec``（构造期已过 V1–V9）；
  - 一层怎么算在 ``StreamModule``（Lane B）；
  - 加噪与 loss 在 ``DiffusionSpec``（Lane C）；
  - 可见性在 ``Visibility``（Lane A）。
因此本文件里的 ``NotImplementedError`` 都指向某一条 lane，而不是某个算法。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .diffusion.spec import DiffusionSpec
from .layout import LabelView, TokenLayout
from .spec.core import build
from .spec.world import WorldModelSpec
from .streams.conditions import ConditionBus, ConditionSpec
from .streams.engine import AttentionGroup, EngineOutput, StreamEngine
from .streams.state import StreamState
from .visibility.combine import Visibility, from_predicate
from .visibility.mask import AttentionPlan

__all__ = ["DiffusionRuntime", "WorldModel", "join_stream_visibility",
           "materialize_conditions", "query_in_stream"]


def materialize_conditions(specs: Sequence[ConditionSpec],
                           batch: Mapping[str, Any], *,
                           training: bool) -> ConditionBus:
    """逐条 ``ConditionSpec`` 物化成 ``Condition``，装进一条总线。

    ``enabled=False`` 的条目**整条跳过**而不是物化一个零张量：
    关掉的条件不应该在总线上占一个名字，否则 V7 的
    「总线上的名字 ↔ 各流的 cross_kv_slots」对账会把它算进去。

    两条路还没有落点，命中即抛错而不是悄悄给个空值：
      - ``encoder`` 非 None —— 条件编码器在 ``__init__`` 四步里没有建造工位；
      - ``dropout_prob > 0`` 且 ``training`` —— CFG 丢弃是置零还是换成
        一个可学的 null 向量，两种实现的权重形状不同，要先裁定。
    """
    bus = ConditionBus()
    for spec in specs:
        if not spec.enabled:
            continue
        if spec.encoder is not None:
            raise NotImplementedError(
                f"条件 {spec.name!r} 带 encoder；条件编码器的建造工位尚未裁定"
                "（model_matrix §4.5.1 的四步里没有它），待该裁定落地"
            )
        if training and spec.dropout_prob > 0.0:
            raise NotImplementedError(
                f"条件 {spec.name!r} 的 dropout_prob={spec.dropout_prob}；"
                "CFG 条件丢弃的语义（置零 / 换可学 null 向量）待裁定"
            )
        if spec.source_key not in batch:
            raise KeyError(
                f"条件 {spec.name!r} 要从 batch[{spec.source_key!r}] 取原料，"
                f"但 batch 只有 {sorted(batch)}"
            )
        bus.add(spec.materialize(batch[spec.source_key]))
    return bus


def query_in_stream(name: str) -> Visibility:
    """「query 落在流 ``name`` 上」。``Visibility.restrict`` 的 where 的取值。

    刻意不进 ``visibility/rules.py`` 的九条原子规则：那九条描述的是两个
    token 之间的关系，这一条只看 query 一侧 —— 它是个**作用域选择器**，
    用来把一条规则限制到它的归属流的那些行上。
    """

    @from_predicate(f"query_in_stream({name!r})")
    def rule(q: LabelView, k: LabelView) -> Tensor:
        names = q.layout.stream_names
        if name not in names:
            raise KeyError(
                f"query_in_stream: 布局里没有流 {name!r}；已有流 {list(names)}"
            )
        if not q.has("stream"):  # 单流布局不贴 stream 标签 ⇒ 全是这条流
            return torch.ones((q.layout.total, k.layout.total),
                              dtype=torch.bool)
        return (q.get("stream") == names.index(name)).expand(
            q.layout.total, k.layout.total)

    return rule


def join_stream_visibility(declared: Mapping[str, Visibility]
                           ) -> Visibility | None:
    """把「流名 → 该流 query 侧的规则」合成一张联合 mask。

    每条规则先 ``restrict`` 到自己流的 query 行上，再把各条 **与** 起来。
    ``restrict`` 在未命中区域保持全可见 ⇒ 与起来之后每条流的行恰好只受
    它自己那条规则管，没声明规则的流的行全可见。

    不能直接 ``&``：那会把别的流的 query 行也按这条规则遮掉。
    症状是跨流边静默消失，形状不变、loss 照降。
    """
    if not declared:
        return None
    joint: Visibility | None = None
    for name in sorted(declared):
        scoped = declared[name].restrict(where=query_in_stream(name))
        joint = scoped if joint is None else joint & scoped
    return joint


@dataclass
class DiffusionRuntime:
    """一次 forward 里扩散相关的中间量。把七步之间要传的东西收成一个对象。

    存在的理由：现状 ``training_loss`` 的七个步骤之间靠十余个局部变量传值，
    其中 ``t`` / ``t_expanded`` / ``timestep_input`` 三者形状不同、含义相近，
    调用点很容易取错一个。收进一个类之后，取错变成 AttributeError。
    """

    timesteps: dict[str, Tensor]            # 流名 → [B, *axes]，原始 t ∈ [0, 1]
    noised: dict[str, Tensor]               # 流名 → x_t
    noise: dict[str, Tensor]                # 流名 → ε，算 target 要用
    targets: dict[str, Tensor]              # 流名 → 回归目标

    def require(self, stream: str) -> tuple[Tensor, Tensor, Tensor]:
        """取一条流的 (x_t, 目标, timestep)。缺失即抛错并列出已有的流。"""
        try:
            return (self.noised[stream], self.targets[stream],
                    self.timesteps[stream])
        except KeyError:
            raise KeyError(
                f"DiffusionRuntime 没有流 {stream!r}；"
                f"已有 {sorted(self.noised)}"
            ) from None


class WorldModel(nn.Module):
    """世界模型。``spec`` 进、``(loss, metrics)`` 出（交界面 I4）。

    ``__init__`` 四步（model_matrix §4.5.1，顺序不可换）：
      ① 按 spec 建各流的 block（``build(stream.impl, ...)``）；
      ② 物化结构可见性 → ``AttentionPlan``（构造期一次，不是每步一次）；
      ③ 建输出头，并由头表推出引擎的 ``captures`` 请求；
      ④ 解析各流的「联合层号 → 局部层号」表。

    第②步在构造期的理由：结构通道是 batch 共享的常量，
    每步重算等于把一张 [L, L] 的张量构造塞进训练循环。
    """

    def __init__(self, spec: WorldModelSpec, *,
                 engine: StreamEngine,
                 layout: TokenLayout) -> None:
        super().__init__()
        self.spec = spec
        self.engine = engine
        self.layout = layout

        # ① 建流 block。参数全在这里，引擎不持有任何参数
        self.blocks = nn.ModuleDict({
            stream.name: build(stream.impl, spec=stream,
                               geometry=spec.geometry)
            for stream in spec.streams
        })

        # ② 物化结构可见性。一个注意力组一张 plan（而不是一条流一张）：
        #    联合注意力是整组共用的那一张 [total, total]
        self.plans: dict[int, AttentionPlan] = {
            index: AttentionPlan.from_visibility(
                self._group_visibility(group), layout)
            for index, group in enumerate(spec.groups)
        }

        # ③ 建头 + 推出抓取请求。请求从头表推出来，不在两处各写一遍
        self.heads = nn.ModuleDict({
            head.name: build(head.impl, spec=head, geometry=spec.geometry)
            for head in spec.heads
        })
        self.captures: dict[str, tuple[int, ...]] = spec.captures()

        # ④ 层号表。构造期解析一次，forward 期只查
        self.layer_index_maps: dict[str, dict[int, int]] = {}
        for group in spec.groups:
            self.layer_index_maps.update(group.layer_index_maps())

    def _group_visibility(self, group: AttentionGroup) -> Visibility | None:
        """一个组的联合可见性。无人声明 ⇒ None（全双向，单体模型的常态）。

        ``spec.visibility`` 的键是**流名**，值是该流 query 侧的规则；
        合成规则见 ``join_stream_visibility``。只取本组成员的声明 ——
        别的组的流不在本组的 ``[total, total]`` 里。
        """
        return join_stream_visibility({
            name: self.spec.visibility[name]
            for name in group.stream_names
            if name in self.spec.visibility})

    # ---- 训练路径。七步，顺序不可换（model_matrix §4.5.2）----

    def forward(self, batch: Mapping[str, Any]
                ) -> tuple[Tensor, dict[str, Any]]:
        """交界面 **I4**：``(loss, metrics)``。这条签名不动。

        七步：
          ① 采 timestep（逐流，``DiffusionSpec.sampler_for``）；
          ② 装 ``ConditionBus``（逐条 ``ConditionSpec`` 物化）；
          ③ 加噪、算回归目标 → ``DiffusionRuntime``；
          ④ 建各流初始 ``StreamState``；
          ⑤ 逐组跑引擎（``StreamEngine.run``）；
          ⑥ 跑输出头（``HiddenStateRef.resolve`` 取隐状态）；
          ⑦ 按 ``LossSpec`` 逐条算 loss 并加权求和。

        第③步必须在第⑤步之前、第①步之后：噪声要用 t，而引擎要吃 x_t。
        第⑥步必须在第⑤步之后：头吃的是引擎的产出。
        这两条是 interfaces §7 里不可重排的约束。
        """
        raise NotImplementedError(
            "WorldModel.forward 的七步编排待实现（装配层）；"
            "各步的被调方分属 Lane A/B/C，判据见 "
            "docs/design/mot_spec_model_matrix.zh-CN.md §4.5.2 的逐行对照表"
        )

    # ---- 六个扩展点（model_matrix §4.5.4）。子类覆写它们，不覆写 forward ----

    def sample_timesteps(self, batch: Mapping[str, Any]
                         ) -> dict[str, Tensor]:
        """扩展点 1：第①步。缺省按 ``DiffusionSpec.timesteps`` 逐流采。"""
        raise NotImplementedError(
            "WorldModel.sample_timesteps 属 Lane C（L3 timestep，S3 交付）"
        )

    def build_conditions(self, batch: Mapping[str, Any]) -> ConditionBus:
        """扩展点 2：第②步。缺省按 ``ConditionSpec`` 逐条物化。"""
        return materialize_conditions(self.spec.conditions, batch,
                                      training=self.training)

    def add_noise(self, batch: Mapping[str, Any],
                  timesteps: Mapping[str, Tensor]) -> DiffusionRuntime:
        """扩展点 3：第③步。缺省走 ``DiffusionSpec.schedule``。"""
        raise NotImplementedError(
            "WorldModel.add_noise 属 Lane C（L3 噪声调度，S3 交付）"
        )

    def initial_states(self, runtime: DiffusionRuntime
                       ) -> dict[str, StreamState]:
        """扩展点 4：第④步。把 x_t 包成各流的初始 ``StreamState``。

        **卡在一条未裁定的交界上，所以这里抛错而不是给个能跑的版本**：
        L2 详设 §370 把 patchify 划给 ``StreamModule.prepare``，而
        ``StreamState.hidden`` 要求恰好三维 ``[B, L, W]``。视频流的 x_t 是
        ``[B, C, F, H, W]`` 五维 —— 两条约定不能同时成立。两个出口：
          ① patchify 留在 ``prepare``，则 ``hidden`` 的三维约束要放宽；
          ② ``hidden`` 的三维约束不动，则空间 → token 的 rearrange 归本方法
             （它要读 ``geometry.patch_size`` 与 ``padded_seq_len``）。
        选②会把一段几何相关的算法搬进装配层，选①会让 C8/C9 失去落点。
        """
        raise NotImplementedError(
            "WorldModel.initial_states：patchify 的归属未裁定 —— "
            "L2 详设 §370 给 StreamModule.prepare，但 StreamState.hidden "
            "要求三维 [B, L, W]，而视频流的 x_t 是五维"
        )

    def run_group(self, group: AttentionGroup,
                  states: Mapping[str, StreamState],
                  conditions: ConditionBus,
                  runtime: DiffusionRuntime) -> EngineOutput:
        """扩展点 5：第⑤步。按组取 plan 与 captures，转调 ``self.engine.run``。

        ``captures`` 只传本组成员那几条 —— 别的组的流不在本次 run 的参与者里，
        传过去会被引擎当成未知流名拒掉（那条校验是对的，不要绕）。
        """
        index = self.spec.groups.index(group)
        captures = {name: layers for name, layers in self.captures.items()
                    if name in group.stream_names}
        return self.engine.run(
            group, states,
            {name: self.blocks[name] for name in group.stream_names},
            self.plans[index], self.layout, conditions=conditions,
            timesteps={name: runtime.timesteps.get(name)
                       for name in group.stream_names},
            captures=captures)

    def compute_losses(self, output: EngineOutput,
                       runtime: DiffusionRuntime,
                       outputs: Mapping[str, Tensor], *,
                       step: int) -> tuple[Tensor, dict[str, Any]]:
        """扩展点 6：第⑦步。缺省按 ``LossSpec.weight_at(step)`` 加权求和。"""
        raise NotImplementedError(
            "WorldModel.compute_losses 属 Lane C（L3 损失，S3 交付）"
        )

    # ---- 推理路径。三步（model_matrix §4.5.3）----

    @torch.no_grad()
    def sample(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        """推理三步：① 建初始 ``Trajectory``；② 按 ``SteppingGate`` 迭代；
        ③ 跑输出头。

        **逐流步数不同就落在第②步**：``Trajectory.active_streams(gate)``
        决定这一步谁动。Fast-WAM 推理期砍掉视频流是
        ``UniformSteps(skip={"video"})``，而不是另一条代码路径。
        """
        diffusion: DiffusionSpec = self.spec.diffusion
        if not diffusion.is_iterative:
            raise NotImplementedError(
                f"模型 {self.spec.model_type!r} 的 DiffusionSpec 没有配 "
                "gate/sampler（单步前馈模型）；单步推理路径属装配层，待实现"
            )
        raise NotImplementedError(
            "WorldModel.sample 的三步编排待实现（装配层）；"
            "判据见 mot_spec_model_matrix.zh-CN.md §4.5.3"
        )
