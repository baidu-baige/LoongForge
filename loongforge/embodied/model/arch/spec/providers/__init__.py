# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""模型 provider。每个文件一个模型族的工厂 + 一个 ``@register_provider``。

本目录里的代码是**拓扑与算法的唯一落点**（R5）：流怎么分、哪些流进同一张
联合注意力、可见性怎么组合、loss 有几条 —— 全写在这里的 Python 里，
YAML 一行也看不见。

两条约定：

1. **注册表的键是 ``model_type``**，与配置里那个键一一对应。
2. 六个拓扑族（dual_stream_wan 之类）是**不注册的工厂函数**，
   不是对外契约；它们被 provider 调用，但 YAML 与启动参数都看不见它们。
"""

from __future__ import annotations

# import 即注册：``@register_provider`` 是 import 期的副作用，
# 所以每加一个 provider 文件都必须在这里列一行 —— 少列一行的症状是
# ``get_provider('vpp2')`` 报「没有这个 model_type」而文件就在旁边。
from . import dual_stream_wan  # noqa: F401

__all__: list[str] = []
