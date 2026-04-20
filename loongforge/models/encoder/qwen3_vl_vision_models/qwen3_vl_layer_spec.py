# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Qwen3-VL layer spec."""

import torch
from typing import Union
from dataclasses import dataclass

from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.extensions.transformer_engine import (
    TELinear,
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
    TENorm
)

from loongforge.models.encoder.qwen3_vl_vision_models.rope_utils import apply_rotary_pos_emb


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""

    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


def get_qwen3_vl_vision_model_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                    apply_rotary_fn=apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_adapeter_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return AdapterSubmodules(
        layernorm=TENorm,
        linear_fc1=TELinear,
        linear_fc2=TELinear,
    )
