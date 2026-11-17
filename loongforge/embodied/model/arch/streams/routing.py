# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 跨流 KV 路由 ``KVRoute``（Lane B）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.5 为准，
可配性以 ``mot_spec_config.zh-CN.md`` §3.5 为准。

与可见性（L1）的分工，这是最容易混的一处：
  - **可见性**说的是「同一张联合注意力里，谁的 query 能看谁的 key」；
  - **KV 路由**说的是「这条流本层的 KV 从**哪一层**的哪条流取」。
前者是 mask，后者是取数。EWAM 的 VL 流把第 k 层的 KV 喂给视频流的第 j 层，
这件事 mask 表达不了 —— 它改变的是取哪个张量，不是遮哪个位置。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KVRoute"]


@dataclass(frozen=True)
class KVRoute:
    """一条有向 KV 路由：``dst`` 流的第 i 层，取 ``src`` 流某一层的 KV。

    ``src_layers`` 与 ``dst_layers`` 是**两条平行列表**，按下标配对，
    长度必须相等。写成两条列表而不是 dict，是因为配置文件里这两列
    常常是两段连续区间，平行列表读起来就是对账表。

    空路由集合（``routes=()``）是**单流模型的常态**：没有对端流可路由。
    引擎里不许出现 ``if not routes`` 的特判 —— 空集合走的是同一条
    「遍历路由」的代码，恰好是空循环。
    """

    src: str                        # 提供 KV 的流名。拓扑，不进 YAML
    dst: str                        # 消费 KV 的流名。拓扑，不进 YAML
    src_layers: tuple[int, ...]     # src 侧的**局部**层号
    dst_layers: tuple[int, ...]     # dst 侧的**局部**层号

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise ValueError(
                f"KVRoute 的 src 与 dst 不得相同（{self.src!r}）："
                "同流取 KV 不是路由，是流自己的事"
            )
        if len(self.src_layers) != len(self.dst_layers):
            raise ValueError(
                f"KVRoute({self.src!r} → {self.dst!r})：src_layers 有 "
                f"{len(self.src_layers)} 项，dst_layers 有 "
                f"{len(self.dst_layers)} 项，两条平行列表长度必须相等"
            )
        if not self.src_layers:
            raise ValueError(
                f"KVRoute({self.src!r} → {self.dst!r}) 的层列表不得为空："
                "不路由任何层就不要建这条路由"
            )
        for name in ("src_layers", "dst_layers"):
            layers = getattr(self, name)
            if any(layer < 0 for layer in layers):
                raise ValueError(
                    f"KVRoute({self.src!r} → {self.dst!r}).{name} 的层号必须非负："
                    f"{layers}"
                )
        if len(set(self.dst_layers)) != len(self.dst_layers):
            raise ValueError(
                f"KVRoute({self.src!r} → {self.dst!r}).dst_layers 有重复："
                f"{self.dst_layers}。同一个 dst 层取两份 KV 是歧义配置"
            )

    def resolve(self, dst_layer: int) -> int | None:
        """``dst`` 流的局部层号 → 该取 ``src`` 的哪个局部层。无路由则 None。

        返回 None 而不是抛错：绝大多数层没有跨流路由，
        「没有」是正常情况而不是错误。
        """
        try:
            index = self.dst_layers.index(dst_layer)
        except ValueError:
            return None
        return self.src_layers[index]

    @classmethod
    def per_layer(cls, src: str, dst: str, *, num_layers: int) -> KVRoute:
        """逐层一对一路由。``dst`` 的第 i 层取 ``src`` 的第 i 层。"""
        if num_layers <= 0:
            raise ValueError(f"num_layers 必须为正，得到 {num_layers}")
        layers = tuple(range(num_layers))
        return cls(src=src, dst=dst, src_layers=layers, dst_layers=layers)

    @classmethod
    def fixed(cls, src: str, dst: str, *, src_layer: int,
              dst_layers: tuple[int, ...]) -> KVRoute:
        """固定源层：``dst`` 的若干层全部取 ``src`` 的**同一层**。

        这是「把 VL 编码器最后一层的 KV 喂给视频流所有层」这种常见写法，
        现状里它表现为一个被反复 index 的缓存变量。
        """
        return cls(src=src, dst=dst,
                   src_layers=(src_layer,) * len(dst_layers),
                   dst_layers=dst_layers)
