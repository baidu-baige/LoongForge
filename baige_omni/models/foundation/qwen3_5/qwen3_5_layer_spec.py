# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3_5 layer spec."""

from copy import deepcopy

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)

from baige_omni.models.foundation.qwen3_next.gated_deltanet import Qwen3NextRMSNorm, GatedDeltaNet
from baige_omni.models.foundation.qwen3_next.gated_attention import Qwen3NextSelfAttention
from baige_omni.models.foundation.qwen3_next.qwen3_next_layer_spec import (
    get_local_layer_specs,
    get_moe_module_spec,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def get_qwen3_5_transformer_layer_spec(config, vp_stage=None):
    """Helper function to get module spec for Qwen3_5"""
    if not HAVE_TE:
        raise ImportError(
            "Qwen3_5 layer spec requires Transformer Engine. "
            "Please install it with: pip install transformer-engine"
        )

    layer_norm_impl = Qwen3NextRMSNorm

    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
    )
    mlp = get_moe_module_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )

    # Build per-layer specs based on layer_types pattern
    layer_types = [
        'full_attention' if (i + 1) % config.full_attention_interval == 0 else 'linear_attention'
        for i in range(config.num_layers)
    ]

    layer_specs = []
    for layer_type in layer_types:
        layer_spec = deepcopy(moe_layer_spec)
        shard_moe_gate_spec = deepcopy(mlp)

        if layer_type == 'linear_attention':
            layer_spec.submodules.self_attention.module = GatedDeltaNet
        elif layer_type == 'full_attention':
            layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
            layer_spec.submodules.self_attention.params = {"attn_mask_type": AttnMaskType.causal}

        # Replace all layernorms with Qwen3NextRMSNorm (zero-centered)
        layer_spec.submodules.input_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules,
                   'pre_mlp_layernorm') and layer_spec.submodules.pre_mlp_layernorm is not IdentityOp:
            layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, 'q_layernorm'):
            layer_spec.submodules.self_attention.submodules.q_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, 'k_layernorm'):
            layer_spec.submodules.self_attention.submodules.k_layernorm = layer_norm_impl

        layer_spec.submodules.mlp = shard_moe_gate_spec
        layer_specs.append(layer_spec)

    # Slice layer specs for pipeline parallelism
    local_layer_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

    # Build MTP (Multi-Token Prediction) block spec if configured
    mtp_block_spec = None
    if config.mtp_num_layers is not None:
        if hasattr(block_spec, 'layer_specs') and len(block_spec.layer_specs) == 0:
            mtp_input_spec = layer_specs[-1]
        else:
            mtp_input_spec = block_spec
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, mtp_input_spec, use_transformer_engine=HAVE_TE, vp_stage=vp_stage
        )
        if mtp_block_spec is not None:
            for layer_spec in mtp_block_spec.layer_specs:
                layer_spec.submodules.enorm = layer_norm_impl
                layer_spec.submodules.hnorm = layer_norm_impl
                layer_spec.submodules.layer_norm = layer_norm_impl

    return block_spec, mtp_block_spec
