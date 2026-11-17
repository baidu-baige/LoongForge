# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""多流世界模型的五层契约。

层次与 lane 的对应（``docs/design/mot_layer_interfaces.zh-CN.md`` §2）：

===========  =========================  ======
层           模块                       lane
===========  =========================  ======
L0 几何      ``geometry`` / ``layout``   A
L1 可见性    ``visibility/``             A
L2 流与参数  ``streams/``                B
L3 监督      ``diffusion/``              C
L4 输出头    ``heads/``                  C
配置层       ``spec/``                   —
装配         ``builder``                 —
===========  =========================  ======

四个交界面（I1–I4）在 M0 冻结，改动走总纲 §9：
  - **I1** A→B：``AttentionPlan`` + ``CompiledMask`` 的四个方法；
  - **I2** A→C：``TokenLayout.labels_of``；
  - **I3** B→C：``EngineOutput.captured``；
  - **I4** C→trainer：``forward(batch) -> (loss, metrics)``。

三条硬不变量：
  - **INV-1** ``AttentionPlan.structural`` 恰为 2 维，不含 batch 维；
  - **INV-2** 形状静态（CUDA graph 捕获）；
  - **INV-3** 不新增 state_dict 前缀。
"""

from __future__ import annotations

from .geometry import GeometrySpec
from .layout import LabelView, SegmentRange, TokenLayout

__all__ = ["GeometrySpec", "TokenLayout", "SegmentRange", "LabelView"]
