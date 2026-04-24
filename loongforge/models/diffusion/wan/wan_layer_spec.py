# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wan layer spec."""

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
)

from megatron.core.transformer.attention import (
    SelfAttention,
    SelfAttentionSubmodules,
    CrossAttention,
    CrossAttentionSubmodules,
)

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
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
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )

    return ModuleSpec(
        module=WanLayer,
        submodules=WanLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
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
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            wan_cross_attention=ModuleSpec(
                module=WanCrossAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=WanCrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
        ),
    )
