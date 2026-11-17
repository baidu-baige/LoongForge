# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 噪声调度 ``NoiseSchedule`` / ``ConstantSigma``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.3 为准。

与 ``timesteps.py`` 的分工：分布决定**采到哪个 t**，调度决定**t 对应多少噪声**。
两者分开的直接理由是 rectified flow 下 ``x_t = (1-t)·x_0 + t·ε`` 这条式子
里 t 既是分布的样本又是插值系数，混写之后换分布会顺带换掉插值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

__all__ = ["NoiseSchedule", "ConstantSigma", "RectifiedFlow"]


class NoiseSchedule(Protocol):
    """t → (加噪的系数, 回归目标)。三个方法就是一个前向过程的全部。"""

    def add_noise(self, clean: Tensor, noise: Tensor,
                  t: Tensor) -> Tensor:
        """产出 ``x_t``。``t`` 已由 ``BroadcastSpec.expand`` 广播到可乘的形状。"""
        ...

    def target(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """回归目标。rectified flow 下是 ``noise - clean``（速度）。

        它是一个**方法**而不是常量字符串："v_prediction" 这类枚举
        在现状里要在三处分别 if 一遍。
        """
        ...

    def sigma(self, t: Tensor) -> Tensor:
        """t 对应的噪声尺度。``visibility.rules.noise_gate`` 的离散档位由它换算。"""
        ...


@dataclass(frozen=True)
class RectifiedFlow:
    """rectified flow。**13 个模型里 12 个是这一档**，所以它是缺省。

    ``x_t = (1-t)·clean + t·noise``，目标 ``noise - clean``。
    """

    def add_noise(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        return (1.0 - t) * clean + t * noise

    def target(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        del t                      # 目标与 t 无关，这是 rectified flow 的性质
        return noise - clean

    def sigma(self, t: Tensor) -> Tensor:
        return t


@dataclass(frozen=True)
class ConstantSigma:
    """固定噪声尺度。配 ``timesteps.Constant`` 表达「这条流不加噪」。

    ``sigma=0.0`` 时 ``add_noise`` 恒等返回 clean —— 这让 clean 条件流
    与噪声流走**同一条代码路径**，而不是靠 ``if is_condition`` 分支。
    """

    sigma_value: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_value < 0:
            raise ValueError(
                f"ConstantSigma.sigma_value 必须非负，得到 {self.sigma_value}"
            )

    def add_noise(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        del t
        if self.sigma_value == 0.0:
            return clean
        return clean + self.sigma_value * noise

    def target(self, clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        del clean, t
        return noise

    def sigma(self, t: Tensor) -> Tensor:
        return t.new_full(t.shape, self.sigma_value)
