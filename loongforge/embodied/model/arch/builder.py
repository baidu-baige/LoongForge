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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .diffusion.spec import DiffusionSpec
from .layout import TokenLayout
from .spec.core import build
from .spec.world import WorldModelSpec
from .streams.conditions import ConditionBus
from .streams.engine import AttentionGroup, EngineOutput, StreamEngine
from .streams.state import StreamState
from .visibility.combine import Visibility
from .visibility.mask import AttentionPlan

__all__ = ["DiffusionRuntime", "WorldModel"]


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

        ``spec.visibility`` 的键是**流名**，值是该流 query 侧的规则。
        把多条流各自的规则合成一张联合 mask，需要用
        ``Visibility.restrict(where=same_stream_query(...))`` 把每条规则
        限制到它自己的 query 行上 —— 而 ``same_stream`` 等九条规则属 Lane A，
        尚未实现。因此这里：零条或一条时直接给出，多条时明确抛错，
        **不**用 ``&`` 凑一个看起来能跑的结果（那会把别的流的 query 行也遮掉）。
        """
        declared = {name: self.spec.visibility[name]
                    for name in group.stream_names
                    if name in self.spec.visibility}
        if not declared:
            return None
        if len(declared) == 1:
            return next(iter(declared.values()))
        raise NotImplementedError(
            f"注意力组 {list(group.stream_names)} 里有 {len(declared)} 条流各自"
            f"声明了可见性（{sorted(declared)}）；把它们合成一张联合 mask 需要 "
            "visibility.rules.same_stream + Visibility.restrict，属 Lane A（S2 交付）"
        )

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
        raise NotImplementedError(
            "WorldModel.build_conditions 属 Lane B（L2 条件注入，S2 交付）"
        )

    def add_noise(self, batch: Mapping[str, Any],
                  timesteps: Mapping[str, Tensor]) -> DiffusionRuntime:
        """扩展点 3：第③步。缺省走 ``DiffusionSpec.schedule``。"""
        raise NotImplementedError(
            "WorldModel.add_noise 属 Lane C（L3 噪声调度，S3 交付）"
        )

    def initial_states(self, runtime: DiffusionRuntime
                       ) -> dict[str, StreamState]:
        """扩展点 4：第④步。缺省把 x_t 作为各流的初始残差。"""
        raise NotImplementedError(
            "WorldModel.initial_states 属 Lane B（L2 入口，S2 交付）"
        )

    def run_group(self, group: AttentionGroup,
                  states: Mapping[str, StreamState],
                  conditions: ConditionBus,
                  runtime: DiffusionRuntime) -> EngineOutput:
        """扩展点 5：第⑤步。缺省直接转调 ``self.engine.run``。"""
        raise NotImplementedError(
            "WorldModel.run_group 属 Lane B（L2 引擎，S2 交付）"
        )

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
