# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


"""Intern CLIP/SIGLIP Model layer spec."""

import torch
from typing import Union
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from .intern_vision_attention import InternSelfAttention, SelfAttentionSubmodules
from .intern_vision_transformer_layer import (
    TransformerLayerIntern,
    TransformerLayerInternVisionSubmodules,
)
from .internvl_config import InternVisionConfig
from loongforge.models.dispatch import multiacc_modules
from loongforge.utils import is_te_min_version
from loongforge.models.common.local_layers.local_norm import LocalNorm
from dataclasses import dataclass


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""

    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


def get_vision_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""

    from .intern_vision_attention import (
        InternViTTEDotProductAttention,
        InternViTRMSNorm,
    )

    qk_norm = (
        multiacc_modules.TENorm
        if is_te_min_version("1.9.0")
        and config.normalization in ["LayerNorm", "RMSNorm"]
        else multiacc_modules.LocalNorm
    )
    if config.model_type == "intern_vit_6b":
        qk_layernorm = InternViTRMSNorm
        core_attention = InternViTTEDotProductAttention
    elif config.model_type == "intern_vit_300m":
        qk_layernorm = qk_norm if config.qk_layernorm else IdentityOp
        core_attention = multiacc_modules.DotProductAttention
    else:
        raise NotImplementedError

    return ModuleSpec(
        module=TransformerLayerIntern,
        submodules=TransformerLayerInternVisionSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=InternSelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=multiacc_modules.TELayerNormColumnParallelLinear,
                    core_attention=core_attention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    q_layernorm=qk_layernorm,
                    k_layernorm=qk_layernorm,
                    apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=multiacc_modules.TELayerNormColumnParallelLinear,
                    linear_fc2=multiacc_modules.TERowParallelLinear,
                ),
            ),
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
    )


def get_adapeter_layer_with_te_spec(config: TransformerConfig) -> AdapterSubmodules:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""

    return AdapterSubmodules(
        layernorm=multiacc_modules.TENorm,
        linear_fc1=torch.nn.Linear,
        linear_fc2=torch.nn.Linear,
    )
