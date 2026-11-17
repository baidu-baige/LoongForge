# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L4 输出头协议 ``OutputHead``（Lane C）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §5.6 为准。

头是 ``nn.Module``（它有参数），但引擎看不见它 —— 头在
``WorldModel.forward`` 第⑥步被调用，输入是 ``HiddenStateRef.resolve``
取出来的张量。头与流的解耦就是这一条：**头不参与联合注意力**。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from torch import Tensor

__all__ = ["OutputHead"]


class OutputHead(Protocol):
    """一个输出头。两个方法：算输出、声明输出宽度。

    ``forward`` 的 ``conditions`` 是只读映射，给需要条件的头
    （动作头常要 state / proprio）。不需要的头忽略它即可 ——
    空字典是常态，不需要分支。
    """

    def forward(self, hidden: Tensor, *,
                timestep: Tensor | None = None,
                conditions: Mapping[str, Any] | None = None) -> Tensor:
        """隐状态 → 输出。

        ``timestep`` 非 None 仅当头自己带调制（动作头的两槽调制，
        见 ``streams.spec.HEAD_SLOTS``）。
        """
        ...

    @property
    def output_width(self) -> int:
        """输出最后一维的宽度。构造期常量，V8 用它与 ``HeadSpec`` 对账。"""
        ...
