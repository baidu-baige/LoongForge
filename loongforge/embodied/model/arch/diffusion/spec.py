# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 扩散规格 ``DiffusionSpec``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.1 为准。

一个 ``DiffusionSpec`` 描述「这个模型怎么加噪、怎么算 loss、怎么推理」。
它是**逐流**的字典而不是全局单值 —— VPP2 的视频流与动作流用不同的
timestep 分布，现状里这件事表现为两段分别硬编码的采样代码。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ..streams.spec import StreamSpec
from .loss import LossSpec
from .schedule import NoiseSchedule, RectifiedFlow
from .stepping import Sampler, SteppingGate
from .timesteps import TimestepSampler

__all__ = ["DiffusionSpec"]


@dataclass(frozen=True)
class DiffusionSpec:
    """扩散 / flow-matching 的全部配置。

    ``timesteps`` 的键是**流名**：只有 ``StreamSpec.needs_timestep=True``
    的流需要条目，多给或少给都是构造期错误（V5）。
    这条校验替掉的现状行为是「流里拿不到 timestep 就当成 0」——
    它让「忘配 timestep」变成一次静默的无噪声训练。
    """

    timesteps: Mapping[str, TimestepSampler]
    losses: tuple[LossSpec, ...]
    schedule: NoiseSchedule = field(default_factory=RectifiedFlow)
    gate: SteppingGate | None = None        # None ⇒ 本模型不走迭代采样
    sampler: Sampler | None = None          # None ⇒ 同上
    noise_levels: tuple[float, ...] = ()
    # 离散噪声档位。非空 ⇒ visibility.rules.noise_gate 可用。
    # 必须是**离散档**而不是连续 sigma：连续值会让 structural 变成逐样本的
    # （INV-1）。这是 12/04 闸门的对象

    def __post_init__(self) -> None:
        if not self.losses:
            raise ValueError("DiffusionSpec.losses 不得为空：没有 loss 的模型训不动")
        names = [loss.name for loss in self.losses]
        if len(set(names)) != len(names):
            raise ValueError(f"DiffusionSpec.losses 有重名项：{names}")
        if (self.gate is None) != (self.sampler is None):
            raise ValueError(
                "DiffusionSpec.gate 与 sampler 必须同时给或同时不给："
                f"当前 gate={self.gate!r}, sampler={self.sampler!r}"
            )
        if len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError(
                f"DiffusionSpec.noise_levels 有重复档位：{self.noise_levels}"
            )
        if any(level < 0 for level in self.noise_levels):
            raise ValueError(
                f"DiffusionSpec.noise_levels 必须非负：{self.noise_levels}"
            )
        object.__setattr__(self, "timesteps",
                           MappingProxyType(dict(self.timesteps)))

    # ---- 查询 ----

    @property
    def is_iterative(self) -> bool:
        """是否走迭代采样。False ⇒ 单步前馈（蒸馏后的模型）。"""
        return self.gate is not None

    @property
    def supports_noise_gate(self) -> bool:
        """是否声明了离散噪声档位，从而 ``noise_gate`` 规则可用。"""
        return bool(self.noise_levels)

    def sampler_for(self, stream: str) -> TimestepSampler:
        """取一条流的 timestep 采样器。缺失即抛错并列出已配的流。"""
        try:
            return self.timesteps[stream]
        except KeyError:
            raise KeyError(
                f"DiffusionSpec.timesteps 没有流 {stream!r} 的条目；"
                f"已配：{sorted(self.timesteps)}"
            ) from None

    def losses_of(self, *, stream: str | None = None,
                  head: str | None = None) -> tuple[LossSpec, ...]:
        """按来源筛 loss。两个都不给 ⇒ 返回全部。"""
        if stream is not None and head is not None:
            raise ValueError("losses_of：stream 与 head 不得同时给")
        if stream is None and head is None:
            return self.losses
        return tuple(loss for loss in self.losses
                     if (loss.stream == stream if stream is not None
                         else loss.head == head))

    # ---- V5 / V6：与流集合对账。构造期调用，不是 forward 期 ----

    def validate_against(self, streams: Sequence[StreamSpec]) -> None:
        """校验 timestep 配置与流的 ``needs_timestep`` 声明一致（V5），
        且每条 loss 的 ``stream`` 确实存在（V6 的流侧部分）。

        head 侧由 ``WorldModelSpec`` 持有头表时一并校验 —— 本类看不到头。
        """
        needs = {spec.name for spec in streams if spec.needs_timestep}
        configured = set(self.timesteps)
        missing = sorted(needs - configured)
        if missing:
            raise ValueError(
                f"流 {missing} 声明了 needs_timestep=True，但 "
                "DiffusionSpec.timesteps 里没有条目（V5）。"
                "缺条目不退化成零噪声 —— 那是一次静默的无噪声训练"
            )
        extra = sorted(configured - needs)
        if extra:
            raise ValueError(
                f"DiffusionSpec.timesteps 配了流 {extra}，但它们的 "
                "needs_timestep=False 或根本不是本模型的流（V5）"
            )
        known = {spec.name for spec in streams}
        for loss in self.losses:
            if loss.stream is not None and loss.stream not in known:
                raise ValueError(
                    f"LossSpec({loss.name!r}).stream={loss.stream!r} 不是本模型的流；"
                    f"本模型的流为 {sorted(known)}（V6）"
                )
