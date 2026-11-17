# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 输出头规格 ``HeadSpec``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.6 为准，
字段可配性以 ``mot_spec_config.zh-CN.md`` §3.7 为准：
**七个字段里只有 ``weight`` 可被 YAML 覆写**，其余是拓扑与算法（R5）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..spec.core import Spec
from ..streams.spec import HEAD_SLOTS, ModulationSlots
from .hidden_ref import HiddenStateRef

__all__ = ["HeadSpec"]


@dataclass(frozen=True)
class HeadSpec:
    """一个输出头的声明。

    ``source`` 是 ``HiddenStateRef`` 而不是一个流名字符串：头可以吃
    中间层（InternW0-Δ 的若干辅助头就是），而流名表达不了层号。
    """

    name: str                           # 头名。也是 metrics / 输出字典的键
    impl: Spec                          # 头的实现。build 产出物须满足 OutputHead
    source: HiddenStateRef              # 吃哪条流的哪一层
    output_width: int                   # 输出最后一维宽度。与实现对账（V8）
    modulation_slots: ModulationSlots = HEAD_SLOTS
    # 头自带的调制槽。两槽（shift_out / scale_out）是动作头的档位。
    # 槽表不许为空（见 ModulationSlots.__post_init__），所以「这个头不调制」
    # 由 needs_timestep=False 表达，而不是给一张空槽表
    needs_timestep: bool = False        # 头是否消费 timestep 做调制
    weight: float = 1.0                 # [YAML] **唯一可 YAML 覆写的字段**

    def __post_init__(self) -> None:
        if self.output_width <= 0:
            raise ValueError(
                f"HeadSpec({self.name!r}).output_width 必须为正，"
                f"得到 {self.output_width}"
            )
        if self.weight < 0:
            raise ValueError(
                f"HeadSpec({self.name!r}).weight 必须非负，得到 {self.weight}"
            )

    @property
    def source_stream(self) -> str:
        """头吃的流名。V8 用它校验这条流确实存在。"""
        return self.source.stream

    @property
    def capture_key(self) -> tuple[str, int] | None:
        """需要引擎抓中间层时的键；吃最终输出时为 None。"""
        return self.source.capture_key
