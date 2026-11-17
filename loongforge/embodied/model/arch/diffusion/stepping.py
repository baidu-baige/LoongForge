# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 推理步进：``Sampler`` / ``SteppingGate`` / ``Trajectory``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.5 为准。
``Trajectory`` / ``SteppingGate`` 的**配置期归属**是总纲 §10-7 的未决项，
这里只定运行期签名。

``SteppingGate`` 要表达的事情，现状没有抽象：**不同流的步数可以不同**。
Fast-WAM 推理期砍掉视频流（视频流步数 0、动作流 10 步）就是这一档，
现状里它表现为 ``infer_action`` 里一段与训练路径完全分叉的代码。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from torch import Tensor

__all__ = ["Trajectory", "SteppingGate", "UniformSteps", "Sampler",
           "EulerSampler"]


@dataclass
class Trajectory:
    """一次采样过程的状态。逐流一个 latent + 一个当前步号。

    ``step`` 是**全局步号**；某条流这一步要不要动，由 ``SteppingGate`` 答。
    """

    latents: dict[str, Tensor]
    step: int = 0
    extras: dict[str, object] = field(default_factory=dict)

    def active_streams(self, gate: SteppingGate) -> tuple[str, ...]:
        """本步要前进的流名。空元组合法 —— 意味着这一步谁都不动。"""
        return tuple(name for name in sorted(self.latents)
                     if gate.should_step(name, self.step))

    def advance(self, updates: Mapping[str, Tensor]) -> Trajectory:
        """写回被更新的流并把步号 +1。未出现在 ``updates`` 里的流原样保留。"""
        unknown = set(updates) - set(self.latents)
        if unknown:
            raise KeyError(
                f"Trajectory.advance 收到未知流 {sorted(unknown)}；"
                f"本轨迹的流为 {sorted(self.latents)}"
            )
        latents = dict(self.latents)
        latents.update(updates)
        return Trajectory(latents=latents, step=self.step + 1,
                          extras=dict(self.extras))


class SteppingGate(Protocol):
    """哪条流在第几步前进。推理期**逐流步数不同**的唯一表达处。"""

    def total_steps(self) -> int:
        """全局步数上限。构造期常量（INV-2：推理图也要静态）。"""
        ...

    def should_step(self, stream: str, step: int) -> bool:
        """``stream`` 在第 ``step`` 步是否前进。"""
        ...


@dataclass(frozen=True)
class UniformSteps:
    """所有流同步步进 ``num_steps`` 步。**多数模型的档位**，也是缺省。

    ``skip`` 列出推理期整条砍掉的流（Fast-WAM 的视频流）。
    砍掉表达为「这条流 should_step 恒 False」，而不是另一条代码路径。
    """

    num_steps: int
    skip: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(
                f"UniformSteps.num_steps 必须为正，得到 {self.num_steps}"
            )

    def total_steps(self) -> int:
        return self.num_steps

    def should_step(self, stream: str, step: int) -> bool:
        if step < 0:
            raise ValueError(f"step 必须非负，得到 {step}")
        return stream not in self.skip and step < self.num_steps


class Sampler(Protocol):
    """一步 ODE / SDE 更新。``WorldModel`` 推理第②步循环调它。"""

    def step(self, trajectory: Trajectory,
             prediction: Mapping[str, Tensor], *,
             gate: SteppingGate) -> Trajectory:
        """吃模型预测，产出下一个轨迹点。

        只更新 ``trajectory.active_streams(gate)`` 里的流 ——
        更新了不该动的流会让「砍掉视频流」失效，而且数值上看不出来。
        """
        ...


@dataclass(frozen=True)
class EulerSampler:
    """一阶 Euler ODE 更新。rectified flow 系的缺省采样器。

    **本类不带步数**：步数归 ``SteppingGate``。现状把两者挤在一个 scheduler
    对象里（VPP2 的四个 scheduler 实例彼此只差一个 shift，
    ``vpp2.py:122-137``），于是「改步数」与「改分布」会互相牵连。
    """

    def step(self, trajectory: Trajectory,
             prediction: Mapping[str, Tensor], *,
             gate: SteppingGate) -> Trajectory:
        raise NotImplementedError(
            "EulerSampler.step 属 Lane C（L3 推理步进，S3 交付）"
        )
