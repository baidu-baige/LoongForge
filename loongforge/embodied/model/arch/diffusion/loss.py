# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 损失规格 ``LossSpec`` / ``WeightRamp``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.4 为准，
``WeightRamp`` 与新交界面 **I4′** 的裁定见总纲 §9 的 11/13 结论。

损失**条数**逐模型不同（2 / 3 / 5 条），因此是一个列表而不是若干形参 ——
这正是 OpenWAM-TS 的二元组 forward 筛掉 InternW0-Δ 与 AW-0 的那条约束，
本层刻意不复制它。
"""

from __future__ import annotations

from dataclasses import dataclass

from .broadcast import DEFAULT_PER_SAMPLE, BroadcastSpec

__all__ = ["WeightRamp", "LossSpec"]


@dataclass(frozen=True)
class WeightRamp:
    """损失权重随训练步数的线性爬升 / 衰减。

    为什么它必须进 spec 而不是写在 trainer 里：权重曲线会改变收敛结果，
    而「这一条 loss 在第几步开始生效」现状散落在训练脚本的 if 里，
    换一次实验要改代码。进 spec 之后它是配置，I4′ 让 trainer 只读不算。
    """

    start_step: int = 0         # [YAML] 开始爬升的步数
    end_step: int = 0           # [YAML] 爬到终值的步数。== start_step ⇒ 阶跃
    start_value: float = 1.0    # [YAML]
    end_value: float = 1.0      # [YAML]

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step < 0:
            raise ValueError(
                f"WeightRamp 的步数必须非负，得到 "
                f"start_step={self.start_step}, end_step={self.end_step}"
            )
        if self.end_step < self.start_step:
            raise ValueError(
                f"WeightRamp.end_step({self.end_step}) 不得小于 "
                f"start_step({self.start_step})"
            )

    @property
    def is_constant(self) -> bool:
        """恒定权重。缺省构造即这一档，爬升逻辑整段不执行。"""
        return self.start_value == self.end_value

    def scale(self, step: int) -> float:
        """第 ``step`` 步的权重。step < start_step ⇒ start_value，
        step ≥ end_step ⇒ end_value，中间线性插值。
        """
        if step < 0:
            raise ValueError(f"step 必须非负，得到 {step}")
        if self.is_constant:
            return self.start_value
        if step <= self.start_step:
            return self.start_value
        if step >= self.end_step:
            return self.end_value
        span = self.end_step - self.start_step
        ratio = (step - self.start_step) / span
        return self.start_value + ratio * (self.end_value - self.start_value)


@dataclass(frozen=True)
class LossSpec:
    """一条损失项。``name`` 既是日志键也是 ``metrics`` 字典的键（I4）。

    ``stream`` + ``head`` 两者**只许填一个**：损失要么对一条流的输出算
    （视频流的 flow matching loss），要么对一个输出头算
    （动作头的 L1）。两个都填是歧义，都不填是死配置。
    """

    name: str
    stream: str | None = None
    head: str | None = None
    weight: float = 1.0                     # [YAML] 恒定权重
    ramp: WeightRamp | None = None          # [YAML] 非 None ⇒ 覆盖 weight
    broadcast: BroadcastSpec = DEFAULT_PER_SAMPLE
    # broadcast 决定「loss 按哪些轴求均值之前先加权」：
    # per-frame timestep 的模型要按帧加权，而轴名必须来自标签（I2）
    enabled: bool = True                    # [YAML] 关掉这条 loss

    def __post_init__(self) -> None:
        if (self.stream is None) == (self.head is None):
            raise ValueError(
                f"LossSpec({self.name!r})：stream 与 head 必须恰好填一个，"
                f"当前 stream={self.stream!r}, head={self.head!r}"
            )
        if self.weight < 0:
            raise ValueError(
                f"LossSpec({self.name!r}).weight 必须非负，得到 {self.weight}"
            )

    @property
    def source(self) -> str:
        """这条 loss 算在谁头上。供错误消息与 V8 对账使用。"""
        return self.stream if self.stream is not None else str(self.head)

    def weight_at(self, step: int) -> float:
        """第 ``step`` 步的实际权重。ramp 为 None ⇒ 恒定 weight。

        ``enabled=False`` ⇒ 0.0，而不是把这条 loss 从列表里摘掉 ——
        摘掉会让 ``metrics`` 的键集合随配置变化，下游看板就断了。
        """
        if not self.enabled:
            return 0.0
        if self.ramp is None:
            return self.weight
        return self.ramp.scale(step)
