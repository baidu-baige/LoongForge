# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L3 广播规格 ``BroadcastSpec``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.4 为准。

本模块是交界面 **I2**（A → C）在 C 侧的落点：timestep / 噪声按哪些轴广播，
必须由 ``TokenLayout`` 的标签名决定，**不许按张量 ndim 猜**。

现状的反面做法是 ``has_seq = len(t.shape) == 4``：形状一变就静默走错分支。
``axes`` 写死标签名之后，标签缺失是构造期的 KeyError。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..layout import TokenLayout

__all__ = ["BroadcastSpec", "DEFAULT_PER_SAMPLE"]


@dataclass(frozen=True)
class BroadcastSpec:
    """timestep / 噪声沿哪些**标签轴**变化。

    ``axes = ()`` ⇒ 逐样本一个标量（最常见，存量全部模型都是这档）。
    ``axes = ("frame",)`` ⇒ 逐帧一个 timestep（VPP2 的 per-frame 噪声）。
    ``axes = ("frame", "chunk")`` ⇒ 帧 × 动作块。

    轴名必须是 ``TokenLayout`` 里真实存在的标签（I2），
    ``validate`` 在构造期查一次 —— 查不到就抛错，不退化成逐样本。
    退化是现状那条错误路径的入口。
    """

    axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(f"BroadcastSpec.axes 有重复轴：{self.axes}")

    @property
    def is_per_sample(self) -> bool:
        """无轴 ⇒ 逐样本标量。"""
        return not self.axes

    def validate(self, layout: TokenLayout) -> None:
        """I2 的校验：每个轴名必须是布局里真实存在的标签。

        **必须在构造期调用**（``WorldModelSpec`` 的 V5 / V6）。
        forward 期再查就晚了：那时错误表现为广播出一个形状恰好能乘的张量。
        """
        missing = [axis for axis in self.axes if not layout.has_label(axis)]
        if missing:
            raise ValueError(
                f"BroadcastSpec.axes 里的标签 {missing} 不在布局里；"
                f"布局提供的标签为 {sorted(layout.labels)}。"
                "标签缺失不退化成逐样本 —— 那正是现状靠 ndim 猜轴的那条错误路径"
            )

    def expand(self, value: Tensor, layout: TokenLayout) -> Tensor:
        """把一个按 ``axes`` 组织的值广播到 ``[B, total, 1]``。

        ``axes = ()`` ⇒ ``value`` 形状 [B]，结果 [B, 1, 1]（纯视图，不复制）。
        ``axes = ("frame",)`` ⇒ ``value`` 形状 [B, num_frames]，
        按 ``layout.labels_of("frame")`` 做 gather。

        实现属 Lane C；签名在此固定，因为 L2 的调制入口要按它接。
        """
        self.validate(layout)
        if not self.axes:
            if value.ndim != 1:
                raise ValueError(
                    f"BroadcastSpec(axes=()).expand 期望 [B] 的逐样本标量，"
                    f"得到 {tuple(value.shape)}"
                )
            return value.view(-1, 1, 1)
        raise NotImplementedError(
            f"BroadcastSpec.expand(axes={self.axes}) 属 Lane C（L3 广播，S3 交付）；"
            "逐轴 gather 的判据见 C 组等价性测试"
        )


DEFAULT_PER_SAMPLE = BroadcastSpec(axes=())
"""逐样本一个标量。**存量全部模型的档位**，也是缺省值。"""
