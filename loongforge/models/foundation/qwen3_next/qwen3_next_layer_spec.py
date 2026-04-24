# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3Next layer spec."""

from copy import deepcopy
from typing import Optional

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec, 
    get_gpt_mtp_block_spec,
)

from .gated_deltanet import GatedDeltaNet, GatedDeltaNetSubmodules, Qwen3NextRMSNorm
from .gated_attention import Qwen3NextSelfAttention
HAVE_TE = True


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    expert_module, expert_submodule = backend.grouped_mlp_modules(
        moe_grouped_gemm is not None and moe_grouped_gemm,
        moe_use_legacy_grouped_gemm is not None and moe_use_legacy_grouped_gemm,
    )
    if expert_submodule is not None:
        expert_submodule.activation_func = activation_func

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": True}, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec


def get_local_layer_specs(config, layer_specs, vp_stage=None):
    """Helper function to get local layer specs"""
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
        from megatron.core.transformer.enums import LayerType
        local_layer_specs = [
            layer_specs[layer_id] for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]
    return local_layer_specs


def get_qwen3_next_transformer_layer_spec(config, vp_stage=None):
    """Helper function to get module spec for Qwen3Next"""
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
            
        # Replace ALL layernorms with Qwen3NextRMSNorm (Zero-Centered)
        layer_spec.submodules.input_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules, 'pre_mlp_layernorm'):
            layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
        # Replace qk_layernorm if present
        if hasattr(layer_spec.submodules.self_attention.submodules, 'q_layernorm'):
            layer_spec.submodules.self_attention.submodules.q_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, 'k_layernorm'):
            layer_spec.submodules.self_attention.submodules.k_layernorm = layer_norm_impl
        layer_spec.submodules.mlp = shard_moe_gate_spec
        layer_specs.append(layer_spec)

    local_layer_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)
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