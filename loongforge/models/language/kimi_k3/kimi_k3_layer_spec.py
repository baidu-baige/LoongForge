# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Megatron-Bridge under the Apache-2.0 License:
# https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/kimi/kimi_k3_spec.py

"""MCore layer specification for Kimi K3."""

import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from omegaconf import ListConfig

from .kimi_k3_attention import KimiK3Attention
from .kimi_k3_moe import KimiK3MoELayer
from .kimi_k3_ops import SiTUAndMul
from .kimi_k3_transformer_layer import KimiK3TransformerLayer


def _with_situ_activation(builder):
    """Replace every MLP/MoE activation with SiTU."""
    if isinstance(builder, ModuleSpec):
        submodules = builder.submodules
        if isinstance(submodules, MLPSubmodules):
            submodules.activation_func = SiTUAndMul
        elif isinstance(submodules, MoESubmodules):
            submodules.experts = _with_situ_activation(submodules.experts)
            if submodules.shared_experts is not None:
                submodules.shared_experts = _with_situ_activation(submodules.shared_experts)
            if builder.module is MoELayer:
                builder.module = KimiK3MoELayer
        return builder
    return builder


def build_kimi_k3_spec(config, vp_stage=None):
    """Build the heterogeneous KDA/MLA and dense/MoE Kimi K3 block."""
    if config.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("Kimi K3 does not support virtual pipeline parallelism yet")
    if isinstance(config.moe_layer_freq, ListConfig):
        config.moe_layer_freq = list(config.moe_layer_freq)

    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.module = KimiK3TransformerLayer
        layer_spec.submodules.self_attention = ModuleSpec(module=KimiK3Attention)
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.pre_mlp_layernorm = TENorm
        layer_spec.submodules.mlp = _with_situ_activation(layer_spec.submodules.mlp)

        # We replace the fused layernorm-linear fc1 with a bare TENorm above, so
        # a dense layer's fc1 must drop its fused layernorm too.
        mlp_spec = layer_spec.submodules.mlp
        if isinstance(mlp_spec, ModuleSpec) and isinstance(mlp_spec.submodules, MLPSubmodules):
            mlp_spec.submodules.linear_fc1 = TEColumnParallelLinear

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec
