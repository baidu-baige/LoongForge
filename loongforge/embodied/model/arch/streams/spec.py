# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 流规格与调制槽（Lane B）。

签名以 ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.1 / §4.2 为准，
字段的 YAML 可配性以 ``mot_spec_config.zh-CN.md`` §3.5 的 R5 为准。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from ..spec.core import Spec

__all__ = [
    "LayerMapping", "StreamSpec", "ModulationSlots",
    "DEFAULT_WAN_SLOTS", "HEAD_SLOTS",
]


class LayerMapping(Enum):
    """一条流的层如何对应到联合层号。

    这是把「num_layers 三者全等」这条校验放宽掉的机制：
    放宽后只要求参与联合注意力的流 attn_width 与 head_dim 相等
    （V2 / V3），层数不等由本枚举协调。
    """

    ONE_TO_ONE = "one_to_one"   # 流层数 == 联合层数，逐层参与
    STRIDED = "strided"         # 按等步长参与（流层数 < 联合层数）
    AT_LAYERS = "at_layers"     # 只在 participating_layers 列出的层参与


@dataclass(frozen=True)
class ModulationSlots:
    """调制槽按**名字**访问，取代散落的字面量 6 与 2。

    为什么名字比数字重要：现状里同一个文件里存在
    ``scale, shift = ...chunk(2)`` 与 ``shift, scale = ...chunk(2)`` 两种顺序。
    字面量 2 分不出这两者，具名槽能。顺序装反不改变 shape，
    数值对齐能过一部分，是最难发现的一类缺陷。
    """

    names: tuple[str, ...]      # 槽名，**顺序就是打包张量里的槽顺序**

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("ModulationSlots.names 不得为空")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"ModulationSlots.names 有重名：{self.names}")

    @property
    def width(self) -> int:
        """槽数。参数形状里凡是乘以槽数的地方都取这个值，不写字面量。"""
        return len(self.names)

    def index(self, name: str) -> int:
        """槽名 → 下标。名字不存在时抛 KeyError（构造期就能查出来）。"""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"调制槽 {name!r} 不存在；本槽表为 {list(self.names)}"
            ) from None

    def has(self, name: str) -> bool:
        """槽是否存在。供「有这个槽才做这步调制」的分支使用。"""
        return name in self.names

    def unpack(self, packed: Tensor, *, slot_axis: int) -> dict[str, Tensor]:
        """把打包的调制张量拆成 {槽名: 张量}。

        ``slot_axis`` **必须显式给出**，不许按 ndim 推断。
        现状的反面做法是 ``has_seq = len(t_mod.shape) == 4`` 再据此选轴 ——
        形状一变就静默走错分支，且错得不报错。
        """
        if packed.shape[slot_axis] != self.width:
            raise ValueError(
                f"ModulationSlots.unpack: packed.shape[{slot_axis}]="
                f"{packed.shape[slot_axis]} != 槽数 {self.width}"
            )
        parts = packed.unbind(dim=slot_axis)
        return dict(zip(self.names, parts, strict=True))

    def gather(self, packed: Tensor, group: tuple[str, ...], *,
               slot_axis: int) -> Tensor:
        """按槽名取**一组**槽并保持打包形态。

        存在的理由：现状有 ``.chunk(2, dim=2)`` 这类写法，它不是「两个槽」，
        而是把六个槽按 attn 三联 / ffn 三联分成两组。
        用 unpack 再重新 cat 会改变语义也改变开销；gather 让这类站点的
        迁移 diff 保持一行。
        """
        indices = torch.tensor([self.index(name) for name in group],
                               dtype=torch.long, device=packed.device)
        return packed.index_select(slot_axis, indices)


# 两个预置槽集，覆盖现状全部 24 + 16 处字面量站点
DEFAULT_WAN_SLOTS = ModulationSlots((
    "shift_msa", "scale_msa", "gate_msa",     # 自注意力前的调制三联
    "shift_mlp", "scale_mlp", "gate_mlp",     # FFN 前的调制三联
))
HEAD_SLOTS = ModulationSlots(("shift_out", "scale_out"))   # 输出头 / 动作头


@dataclass(frozen=True)
class StreamSpec:
    """一条流的全部形状事实。构造期不可变，是 C1–C6 校验的对象。

    把 ``residual_width`` 与 ``attn_width`` 分开是本类的核心：
    现状**五个**模型两者不等（fastwam / motus / pi05 / VPP2 / EWAM；
    VPP2 动作流 residual 512 / attn 5120 是唯一的直接证据来源），
    而现状代码里这两个数常常挤在同一个表达式里
    （``.unflatten(2, (6, 3072))`` 同时硬编码了槽数和残差宽度）。

    字段的可配置性分两类，界线即 spec_config §4.2 的 R5：标 **[YAML]** 的
    可被 YAML 覆写（「换预训练权重就会变」或「调参的人会反复扫」），
    其余只在 ``spec/providers/<family>.py`` 里写 —— 它们是拓扑与算法，
    换一个值就是换一个模型。
    """

    # ---- 身份 ----
    name: str                           # 流名。三处同时用它：TokenLayout 的 stream
                                        # 标签值、capture hook 的键、optimizer 分组
    # ---- 实现 ----
    impl: Spec                          # 本流 block 的实现。WorldModel.__init__ 调
                                        # build(impl, spec=self, geometry=...)，
                                        # 产出物必须满足 StreamModule（§4.3）。
                                        # 必填、无缺省，**永不进 YAML**
    # ---- 形状 ----
    residual_width: int                 # [YAML] 残差流宽度 = block 的输入输出宽度
    num_heads: int                      # [YAML] 注意力头数
    head_dim: int                       # [YAML] 单头宽度。联合成员间必须相等（V3）
    num_layers: int                     # [YAML] 本流自己的层数，与联合层数无关
    ffn_width: int | None = None        # [YAML] FFN 中间宽度，None ⇒ 4 × residual_width。
                                        # 只收绝对值：lingbot_va 的 14336 = 3.5 × 4096
    # ---- 层对应 ----
    layer_mapping: LayerMapping = LayerMapping.ONE_TO_ONE
    participating_layers: tuple[int, ...] | None = None   # AT_LAYERS 专用，联合层号
    stride: int | None = None           # STRIDED 专用，等步长
    # 三者都是拓扑：EWAM 的 VL 流只在 28 个联合层里参与是模型定义本身，
    # 换成别的层集合就是换一个模型 ⇒ 不进 YAML
    # ---- 调制 ----
    modulation_slots: ModulationSlots = DEFAULT_WAN_SLOTS
    # 槽集是算法：WAN 系六槽、动作头两槽。换槽集改的是前向公式
    cross_kv_slots: tuple[str, ...] = ()
    # 本流 block 内部的 cross-attention 槽名。空 ⇒ 无 cross-attn。
    # 槽名与 ConditionBus 的条件名对齐（§4.4），是「第 N 个条件输入」
    # 与「第 N 个输出头」分家的落点
    # ---- 行为 ----
    needs_timestep: bool = True         # 本流是否消费 timestep。False ⇒ VL 这类纯条件流
    writable: bool = True               # False ⇒ 只读流：有入边无出边，残差不被更新
    # ---- 参数治理 ----
    param_prefix: str = ""              # state_dict 前缀。空 ⇒ 用 name。
                                        # INV-3：迁移存量模型时必须填成现状前缀
    lr_scale: float = 1.0               # [YAML] 本流参数的学习率缩放

    def __post_init__(self) -> None:
        for field_name in ("residual_width", "num_heads", "head_dim", "num_layers"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(
                    f"StreamSpec({self.name!r}).{field_name} 必须为正，得到 {value}"
                )
        if self.ffn_width is not None and self.ffn_width <= 0:
            raise ValueError(
                f"StreamSpec({self.name!r}).ffn_width 必须为正或 None，"
                f"得到 {self.ffn_width}"
            )
        if self.layer_mapping is LayerMapping.AT_LAYERS:
            if not self.participating_layers:
                raise ValueError(
                    f"StreamSpec({self.name!r}) 的 layer_mapping=AT_LAYERS "
                    "必须给出非空的 participating_layers"
                )
            if len(set(self.participating_layers)) \
                    != len(self.participating_layers):
                raise ValueError(
                    f"StreamSpec({self.name!r}).participating_layers 有重复："
                    f"{self.participating_layers}"
                )
            if len(self.participating_layers) != self.num_layers:
                raise ValueError(
                    f"StreamSpec({self.name!r})：participating_layers 有 "
                    f"{len(self.participating_layers)} 项，"
                    f"num_layers={self.num_layers}，两者必须相等"
                )
        elif self.participating_layers is not None:
            raise ValueError(
                f"StreamSpec({self.name!r}).participating_layers 只在 "
                f"layer_mapping=AT_LAYERS 时有意义，当前为 {self.layer_mapping}"
            )
        if self.layer_mapping is LayerMapping.STRIDED:
            if self.stride is None or self.stride <= 0:
                raise ValueError(
                    f"StreamSpec({self.name!r}) 的 layer_mapping=STRIDED "
                    f"必须给出正的 stride，得到 {self.stride}"
                )
        elif self.stride is not None:
            raise ValueError(
                f"StreamSpec({self.name!r}).stride 只在 layer_mapping=STRIDED "
                f"时有意义，当前为 {self.layer_mapping}"
            )
        if self.lr_scale <= 0:
            raise ValueError(
                f"StreamSpec({self.name!r}).lr_scale 必须为正，得到 {self.lr_scale}"
            )
        if len(set(self.cross_kv_slots)) != len(self.cross_kv_slots):
            raise ValueError(
                f"StreamSpec({self.name!r}).cross_kv_slots 有重名："
                f"{self.cross_kv_slots}"
            )

    # ---- 派生量一律是 property，不是字段（总纲 §1 规则 3）----

    @property
    def attn_width(self) -> int:
        """注意力宽度 = num_heads × head_dim。

        **它与 residual_width 不是一回事**，五个模型两者不等。
        现状代码把它当作同一个数来用，是本类存在的直接原因。
        """
        return self.num_heads * self.head_dim

    @property
    def mlp_width(self) -> int:
        """FFN 中间宽度。None ⇒ 4 × residual_width 这条缺省只在这里一处。"""
        if self.ffn_width is not None:
            return self.ffn_width
        return 4 * self.residual_width

    @property
    def state_dict_prefix(self) -> str:
        """state_dict 前缀。空 param_prefix ⇒ 退化成流名（INV-3）。"""
        return self.param_prefix or self.name

    def participates_in_joint(self, layer_id: int) -> bool:
        """联合层号 ``layer_id`` 这一层，本流是否参与联合注意力。

        三种 LayerMapping 的判定全在这一个方法里 —— 引擎不许自己按
        ``layer_mapping`` 分支，否则 STRIDED 的边界条件会有两份实现。
        """
        if layer_id < 0:
            raise ValueError(f"layer_id 必须非负，得到 {layer_id}")
        match self.layer_mapping:
            case LayerMapping.ONE_TO_ONE:
                return layer_id < self.num_layers
            case LayerMapping.STRIDED:
                assert self.stride is not None    # __post_init__ 已保证
                if layer_id % self.stride != 0:
                    return False
                return layer_id // self.stride < self.num_layers
            case LayerMapping.AT_LAYERS:
                assert self.participating_layers is not None
                return layer_id in self.participating_layers

    def resolve_layer_index_map(self,
                               group_num_layers: int) -> dict[int, int]:
        """联合层号 → 本流**局部**层号。构造期解析一次，forward 期只查表。

        为什么是构造期：现状每层都要算一次「这一层我参不参与」，
        而这个判定只依赖 spec，与数据无关。forward 期算等于把常量塞进计算图，
        也让 CUDA graph 捕获看见一条 Python 分支（INV-2）。

        返回的 dict 只含本流**参与**的联合层；不参与的层不在键集合里。
        """
        if group_num_layers <= 0:
            raise ValueError(
                f"group_num_layers 必须为正，得到 {group_num_layers}"
            )
        index_map: dict[int, int] = {}
        local = 0
        for layer_id in range(group_num_layers):
            if self.participates_in_joint(layer_id):
                index_map[layer_id] = local
                local += 1
        if local != self.num_layers:
            raise ValueError(
                f"StreamSpec({self.name!r})：layer_mapping={self.layer_mapping.value} "
                f"在联合层数 {group_num_layers} 下只解析出 {local} 个参与层，"
                f"而 num_layers={self.num_layers}。两者必须相等 —— "
                "差值意味着有层拿不到参数，或有参数永远不被调用"
            )
        return index_map
