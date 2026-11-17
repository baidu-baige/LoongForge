# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 timestep 分布与采样器（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.2 为准，
四个分布的键名以 ``config_layer_facts.md`` 里的四套 Beta 键名为准。

分布是**算法**（R5）：换 logit_normal 为 beta 就是换一个训练目标，
因此分布的**种类**不进 YAML，只有它的**参数**进（调参的人会反复扫）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .broadcast import DEFAULT_PER_SAMPLE, BroadcastSpec

__all__ = [
    "TimestepDistribution", "UniformShift", "LogitNormal", "BetaDist",
    "Constant", "TimestepSampler",
]


class TimestepDistribution(Protocol):
    """timestep 分布的协议。只有一个方法 —— 采一批出来。

    返回值必须落在 **[0, 1]**：σ 的换算、1000 倍缩放这类事情归
    ``NoiseSchedule``，不归分布。现状把两者混在一个函数里，
    于是「改采样分布」会顺带改掉噪声尺度。
    """

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        """采 ``shape`` 形状的 timestep，值域 [0, 1]。"""
        ...


@dataclass(frozen=True)
class UniformShift:
    """均匀分布 + shift 重参数化。WAN 系的缺省。

    ``shift=1.0`` 即纯均匀。shift > 1 把质量往高噪声端推。
    """

    shift: float = 1.0          # [YAML]

    def __post_init__(self) -> None:
        if self.shift <= 0:
            raise ValueError(f"UniformShift.shift 必须为正，得到 {self.shift}")

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        raise NotImplementedError(
            "UniformShift.sample 属 Lane C（L3 timestep，S3 交付）"
        )


@dataclass(frozen=True)
class LogitNormal:
    """logit-normal 分布。SD3 / pi05 系的缺省。"""

    mean: float = 0.0           # [YAML]
    std: float = 1.0            # [YAML]

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError(f"LogitNormal.std 必须为正，得到 {self.std}")

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        raise NotImplementedError(
            "LogitNormal.sample 属 Lane C（L3 timestep，S3 交付）"
        )


@dataclass(frozen=True)
class BetaDist:
    """Beta 分布。存量四套配置用了四组不同的键名拼同一件事，这里统一成两个。

    键名统一的代价是**配置迁移表**，收益是「这到底是哪个 Beta」
    不再需要读四份 YAML 才能回答。
    """

    alpha: float = 1.0          # [YAML]
    beta: float = 1.0           # [YAML]

    def __post_init__(self) -> None:
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"BetaDist.{name} 必须为正，得到 {value}"
                )

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        raise NotImplementedError(
            "BetaDist.sample 属 Lane C（L3 timestep，S3 交付）"
        )


@dataclass(frozen=True)
class Constant:
    """常量 timestep。**不是调试工具** —— 它是「这条流不加噪」的正式表达。

    ``value=0.0`` 即 clean 条件流（VPP2 的 clean-conditioning 视频流）。
    用它而不是在流里写 ``if not add_noise``，是为了让「这条流的噪声水平」
    在 spec 里可见，从而 ``visibility.rules.noise_gate`` 能读到它。
    """

    value: float = 0.0          # [YAML]

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"Constant.value 必须落在 [0, 1]，得到 {self.value}"
            )

    def sample(self, shape: tuple[int, ...], *,
               device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.full(shape, self.value, device=device, dtype=dtype)


@dataclass(frozen=True)
class TimestepSampler:
    """分布 + 广播规格。``WorldModel.forward`` 第①步调它。

    把 ``broadcast`` 绑在采样器上而不是分布上：同一个分布可以逐样本采、
    也可以逐帧采，而「采多少个」正是 ``broadcast.axes`` 决定的。
    """

    distribution: TimestepDistribution
    broadcast: BroadcastSpec = DEFAULT_PER_SAMPLE

    def sample(self, batch_size: int, *, device: torch.device,
               dtype: torch.dtype = torch.float32,
               axis_sizes: tuple[int, ...] = ()) -> Tensor:
        """采一批 timestep。形状 = ``(batch_size, *axis_sizes)``。

        ``axis_sizes`` 必须与 ``broadcast.axes`` 一一对应 ——
        长度不等就是配置错误，而不是靠广播规则兜住。
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，得到 {batch_size}")
        if len(axis_sizes) != len(self.broadcast.axes):
            raise ValueError(
                f"TimestepSampler.sample：axis_sizes 有 {len(axis_sizes)} 项，"
                f"broadcast.axes={self.broadcast.axes} 要求 "
                f"{len(self.broadcast.axes)} 项"
            )
        return self.distribution.sample((batch_size, *axis_sizes),
                                        device=device, dtype=dtype)
