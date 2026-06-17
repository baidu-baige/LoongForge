# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wan layer spec."""

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    TEDotProductAttention,
    TENorm,
)

from megatron.core.transformer.attention import (
    SelfAttentionSubmodules,
)

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec

from .wan_layer import (
    WanLayer,
    WanLayerSubmodules,
    WanCrossAttentionSubmodules,
)
from .wan_attention import WanSelfAttention, WanCrossAttention
from .wan_utils import wan_rope_apply


def get_wan_layer_with_te_spec() -> ModuleSpec:
    """
    Use this spec to use lower level Transformer Engine modules (required for fp8 training)
    """
    return ModuleSpec(
        module=WanLayer,
        submodules=WanLayerSubmodules(
            wan_self_attention=ModuleSpec(
                module=WanSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    apply_rotary_fn=wan_rope_apply,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            wan_cross_attention=ModuleSpec(
                module=WanCrossAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=WanCrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    linear_k_img=TEColumnParallelLinear,
                    linear_v_img=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    k_img_layernorm=TENorm,
                ),
            ),
        ),
    )
