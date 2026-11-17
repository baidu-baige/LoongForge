# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 KV cache 的签名：``CacheSignature`` / ``CacheScope``（Lane B）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.6 为准。

**首批两个模型都不用 KV cache**，本模块只定签名、不实现。
它存在的理由是让「推理期要不要 cache」在构造期就能协商
（``CompiledMask.supports`` 的 ``Scenario.has_kv_cache``），
而不是等真跑到推理才发现表达不了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor

__all__ = ["CacheSignature", "CacheScope"]


@dataclass(frozen=True)
class CacheSignature:
    """一条流一层的 cache 形状签名。静态形状（INV-2）由它保证。

    ``max_tokens`` 必须是构造期常量：cache 一旦进 CUDA graph 捕获，
    运行期改长度等于触发重捕获。
    """

    stream: str
    layer_id: int               # 联合层号
    num_heads: int
    head_dim: int
    max_tokens: int

    def __post_init__(self) -> None:
        for name in ("num_heads", "head_dim", "max_tokens"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"CacheSignature({self.stream!r}, L{self.layer_id})."
                    f"{name} 必须为正，得到 {value}"
                )
        if self.layer_id < 0:
            raise ValueError(
                f"CacheSignature({self.stream!r}).layer_id 必须非负，"
                f"得到 {self.layer_id}"
            )

    @property
    def key(self) -> tuple[str, int]:
        """cache 字典的键。与 ``EngineOutput.captured`` 同构，刻意一致。"""
        return (self.stream, self.layer_id)

    def shape(self, batch_size: int) -> tuple[int, int, int, int]:
        """[B, max_tokens, num_heads, head_dim]。与 ``QKV`` 的布局一致。"""
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，得到 {batch_size}")
        return (batch_size, self.max_tokens, self.num_heads, self.head_dim)


class CacheScope(Protocol):
    """一次推理内的 cache 生命周期。首批只定签名。

    ``reset`` 存在而不是靠重建对象：重建会重新分配显存，
    而 CUDA graph 捕获的是地址。
    """

    def signatures(self) -> tuple[CacheSignature, ...]:
        """本 scope 管的全部 cache 槽。构造期可枚举。"""
        ...

    def read(self, signature: CacheSignature) -> tuple[Tensor, Tensor]:
        """取 (key, value)。形状即 ``signature.shape(batch_size)``。"""
        ...

    def write(self, signature: CacheSignature, *, key: Tensor, value: Tensor,
              at: int) -> None:
        """就地写入第 ``at`` 个位置起的若干 token。不得改变 cache 形状。"""
        ...

    def reset(self) -> None:
        """清空有效长度，**不**重新分配显存。"""
        ...

    def state(self) -> dict[str, Any]:
        """调试用的快照。不进 state_dict（INV-3）。"""
        ...
