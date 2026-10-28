# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


"""Qwen2-VL layer spec."""

import torch
from typing import Union
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.extensions.transformer_engine import (
    TELinear,
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
    TENorm,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from loongforge.models.common.local_layers.local_norm import LocalNorm

from dataclasses import dataclass


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""

    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    t,
    freqs,
    config,
    cu_seqlens=None,
    rotary_interleaved=False,
    mscale: float = 1.0,
    cp_group=None,
    **kwargs,
):
    """ " Apply rotation to positional embedding"""
    orig_dtype = t.dtype
    t = t.float()
    if cu_seqlens is not None:
        freqs = freqs.squeeze(1)
        cos_ = (freqs.cos() * mscale).float().repeat(1, 1, 2)
        sin_ = (freqs.sin() * mscale).float().repeat(1, 1, 2)
    else:
        cos_ = (freqs.cos() * mscale).float().repeat(1, 1, 1, 2)
        sin_ = (freqs.sin() * mscale).float().repeat(1, 1, 1, 2)
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return t.to(orig_dtype)


def get_qwen2_vl_vision_model_layer_with_te_spec(
    config: TransformerConfig,
) -> ModuleSpec:
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
                    apply_rotary_fn=apply_rotary_pos_emb_vision,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
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
        layernorm=(
            TENorm if config.normalization in ["LayerNorm", "RMSNorm"] else LocalNorm
        ),
        linear_fc1=TELinear,
        linear_fc2=TELinear,
    )
