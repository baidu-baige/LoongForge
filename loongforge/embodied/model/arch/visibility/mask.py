# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L1 物化：``AttentionPlan`` 两通道、``CompiledMask`` 与能力协商（Lane A）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §3.4 / §3.5 / §3.7 为准。
``AttentionPlan`` + ``CompiledMask`` 的四个方法构成交界面 **I1**：
Lane B 只许调这四个方法，不许读 ``TokenLayout`` 的标签自行判定。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

from torch import Tensor

from ..layout import SegmentRange, TokenLayout
from .combine import Visibility

__all__ = [
    "AttentionPlan", "Capability", "Scenario", "CompiledMask",
    "LayerAttentionOverride",
]


@dataclass(frozen=True)
class AttentionPlan:
    """I1 的载体。两通道拆分是本次重构性能收益的前提。

    为什么拆两通道：结构可见性（谁能看谁）是 batch 共享的，可编译、可缓存、
    可生成 BlockMask；逐样本有效位（这条样本的 padding 在哪）必然带 batch 维。
    混在一张张量里 ⇒ 整张 mask 带上 batch 维 ⇒ BlockMask 复用失效 ⇒ 全退 dense。

    硬不变量（总纲 §1 规则 1 / INV-1）：
        ``structural.shape == (total, total)``，**维数恰为 2，不含 batch 维**。
    这条断言就在 ``__post_init__`` 里，CI 另有一条独立的形状断言用例。
    """

    structural: Tensor | None = None    # [total, total] bool，batch 共享；None = 全可见
    key_valid: Tensor | None = None     # [B, total] bool，逐样本；None = 全有效
    layout: TokenLayout | None = None   # 来源布局，供 B 侧断言段顺序（C8）
    # structural=None 是**单体模型的常态**（interfaces §2.3）：单流 + 无因果 ⇒ 全可见，
    # 此时不应物化一张 [L, L] 的全 True 张量（纯浪费显存且会挡住 flash 的 fast path）。
    # 多流模型里 structural=None 只在「全双向」时出现（存量 motus 就是这种）。

    def __post_init__(self) -> None:
        if self.structural is not None:
            shape = tuple(self.structural.shape)
            if self.structural.ndim != 2:
                raise ValueError(
                    f"AttentionPlan.structural 必须恰为 2 维（不含 batch 维，INV-1），"
                    f"得到 {shape}"
                )
            if shape[0] != shape[1]:
                raise ValueError(
                    f"AttentionPlan.structural 必须是方阵，得到 {shape}"
                )
            if self.layout is not None and shape[0] != self.layout.total:
                raise ValueError(
                    f"AttentionPlan.structural 形状 {shape} 与 layout.total "
                    f"{self.layout.total} 不一致"
                )
        if self.key_valid is not None:
            if self.key_valid.ndim != 2:
                raise ValueError(
                    f"AttentionPlan.key_valid 必须是 [B, total] 两维，"
                    f"得到 {tuple(self.key_valid.shape)}"
                )
            if self.layout is not None \
                    and self.key_valid.shape[1] != self.layout.total:
                raise ValueError(
                    f"AttentionPlan.key_valid 的 total 维 "
                    f"{self.key_valid.shape[1]} != layout.total {self.layout.total}"
                )

    @property
    def total(self) -> int:
        if self.layout is not None:
            return self.layout.total
        if self.structural is not None:
            return int(self.structural.shape[0])
        if self.key_valid is not None:
            return int(self.key_valid.shape[1])
        raise ValueError("AttentionPlan 三个字段全为 None，无法给出 total")

    @classmethod
    def from_visibility(cls, visibility: Visibility | None,
                        layout: TokenLayout,
                        *, key_valid: Tensor | None = None) -> AttentionPlan:
        """``WorldModel.__init__`` 第②步：构造期物化一次结构通道。

        ``visibility is None`` ⇒ 全双向，``structural`` 保持 None
        （缺省即不限制；单流模型与存量 motus 走这条）。
        """
        structural = None if visibility is None else visibility.evaluate(layout)
        return cls(structural=structural, key_valid=key_valid, layout=layout)

    def materialize(self, backend: str, *,
                    block_size: int | None = None) -> Any:
        """按后端产出注意力实际要的那个对象。

        dense → [1, 1, Lq, Lk] bool（广播到 batch，不实体化 batch 维）
        flex  → BlockMask（由 structural 生成，key_valid 走 score_mod）；
                **仅此路径读 block_size**，取 ``WorldModelSpec.attention_block_size``
        sdpa  → 加性 float bias
        两个通道均为 None → 返回 None，调用方走无 mask 的 fast path

        返回类型刻意为 Any：各后端要的对象类型本就不同。
        """
        if self.structural is None and self.key_valid is None:
            return None
        if backend == "dense":
            if self.key_valid is not None:
                raise NotImplementedError(
                    "AttentionPlan.materialize: dense 后端合并 key_valid 属 Lane A"
                )
            return self.structural.unsqueeze(0).unsqueeze(0)
        raise NotImplementedError(
            f"AttentionPlan.materialize(backend={backend!r}) 属 Lane A（L1 物化，"
            "S2 交付；flex 的 as_block_mask 协商在 S4）"
        )


class Capability(Enum):
    """能力协商的三档结果。"""

    EXACT = "exact"              # 该后端能精确表达这条可见性
    DEGRADED = "degraded"        # 能表达但要降级（例如 flash → SDPA）
    UNSUPPORTED = "unsupported"  # 表达不了


@dataclass(frozen=True)
class Scenario:
    """协商的场景描述。训练与推理要分别协商，这是本对象存在的原因。

    五个字段分属三处（backend/block_size 在 L2、cuda_graph 在 trainer、
    has_kv_cache 在推理路径），由 build 期汇总后注入；L1 只消费、不构造。
    职责表述见总纲 §10-5（12/04 到期）。
    """

    phase: Literal["train", "inference"]
    backend: str                     # 取 WorldModelSpec.attention_backend 的**解析后**值
    has_kv_cache: bool = False
    cuda_graph: bool = False
    block_size: int | None = None    # 仅 backend="flex" 有意义。它进 Scenario 而不只是
                                     # as_block_mask 的参数，是因为它改变协商结果：
                                     # 边界不落在 block_size 整数倍上的可见性，
                                     # BlockMask 只能给 DEGRADED

    def __post_init__(self) -> None:
        if self.backend == "auto":
            raise ValueError(
                "Scenario.backend 不接受 'auto'：build 期必须已解析成具体后端"
            )


@dataclass(frozen=True)
class CompiledMask:
    """物化后的 mask。Lane B **只许调**本类的四个方法（I1 的约定）。

    B 不许读 ``TokenLayout`` 的标签去自己判定 —— 否则同一条规则会有两份实现，
    而它们会在边界帧和 padding 行上分叉。
    """

    structural: Tensor          # [total, total] bool，由 Visibility.evaluate 得到
    layout: TokenLayout
    backend: str

    def as_dense(self) -> Tensor:
        """[1, 1, Lq, Lk] bool。所有后端的回退路径，也是等价性测试的基准。"""
        return self.structural.unsqueeze(0).unsqueeze(0)

    def as_block_mask(self, *, block_size: int) -> Any:
        """FlexAttention 的 BlockMask。要求 structural 无 batch 维，否则抛错。

        ``block_size`` 必须与协商时 ``Scenario.block_size`` 用的是同一个值 ——
        换了值就是换一张 mask，而 BlockMask 会被 ``torch.compile`` 捕获，
        运行期再改等于触发重编译（硬约束：静态形状）。
        """
        raise NotImplementedError(
            "CompiledMask.as_block_mask 属 Lane A（S4 与 Lane B 协商 block_size）"
        )

    def as_segments(self) -> tuple[SegmentRange, ...]:
        """段表物化（varlen 路径用）。首批两个模型都不需要，是降级时第一个可砍的。

        实现要先验证结构通道确实是按段分块的，再给段表 ——
        直接返回 ``layout.segments`` 等于把「可见性恰好是块对角」当成前提。
        """
        raise NotImplementedError(
            "CompiledMask.as_segments 属 Lane A（varlen 路径，R3 cosmos3 才需要）"
        )

    @staticmethod
    def supports(vis: Visibility, scenario: Scenario) -> Capability:
        """能力协商。**必须在构造期（build_world_model）调用，不是 forward 期。**

        要拦住的那一类缺陷：训练期 EXACT、推理期 UNSUPPORTED。
        现状里这类缺陷的表现是「训练支持、推理 raise NotImplementedError」，
        而且只有真跑到推理才知道。协商提前之后，它变成一个配置错误。
        """
        raise NotImplementedError(
            "CompiledMask.supports 属 Lane A；V9 在它交付前按「后端已注册」单条校验"
        )


class LayerAttentionOverride(Protocol):
    """层级覆写 / escape hatch：某些层在层**内部**物理截断 key 集合，
    序列长度不是常量，[L, L] 的 mask 表达不了。

    与 INV-2（形状静态）的冲突及边界：
      - 走 override 的层**不能**进 CUDA graph 捕获；
      - 因此 override 必须声明在配置里，由 per_microbatch_cuda_graph
        在捕获时跳过这些层，而不是运行期发现；
      - 首批两个模型（VPP2、InternW0-Δ）都不需要它，所以首批**只定签名、
        不实现消费方**，静态形状规则全程成立。

    约定：override 不得改变各流的 token 数（否则引擎的 C8/C9 会在下一层抛错）。
    它只改变「这一层内部怎么算」，不改变「这一层输出多长」。
    """

    def handles_layer(self, layer_id: int) -> bool:
        """本 override 是否接管该联合层号。构造期即可查，供捕获逻辑排除该层。"""
        ...

    def run_attention(self, layer_id: int,
                      qkvs: Mapping[str, Any],
                      plan: AttentionPlan,
                      layout: TokenLayout) -> dict[str, Tensor]:
        """自行完成这一层的注意力，返回逐流输出。

        拿到的是**未拼接**的逐流 QKV（``streams.state.QKV``）—— 因为 override
        的全部意义就是不走「拼接 + 一次 masked attention」这条路。
        返回的字典键集合必须与 ``qkvs`` 相等，且每个张量的 token 数不变。
        """
        ...
