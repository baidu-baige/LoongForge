# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V-4.6 transformer layer specification."""

from copy import deepcopy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from loongforge.models.foundation.qwen3_next.qwen3_next_layer_spec import (
    get_local_layer_specs,
    get_moe_module_spec,
)

from .minicpm_v_4_6_attention import MiniCPMV46SelfAttention
from .minicpm_v_4_6_gated_deltanet import MiniCPMV46GatedDeltaNet, Qwen3NextRMSNorm
from .minicpm_v_4_6_mlp import MiniCPMV46DenseMLP


def _get_dense_mlp_module_spec():
    return ModuleSpec(
        module=MiniCPMV46DenseMLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )


def get_minicpm_v_4_6_transformer_layer_spec(config, vp_stage=None):
    """Build MiniCPM layers without changing the Qwen3.5 default specification."""
    layer_norm_impl = Qwen3NextRMSNorm
    is_dense = config.num_moe_experts is None
    base_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
    )
    mlp = (
        _get_dense_mlp_module_spec()
        if is_dense
        else get_moe_module_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
        )
    )
    layer_types = [
        "full_attention" if (index + 1) % config.full_attention_interval == 0 else "linear_attention"
        for index in range(config.num_layers)
    ]

    layer_specs = []
    for layer_type in layer_types:
        layer_spec = deepcopy(base_layer_spec)
        if layer_type == "linear_attention":
            layer_spec.submodules.self_attention.module = MiniCPMV46GatedDeltaNet
            layer_spec.submodules.self_attention.params = {
                "projection_split_mode": "qwen3_5",
                "gated_delta_rule_backend": config.gated_delta_rule_backend,
                "gated_norm_backend": config.gated_norm_backend,
                "causal_conv_backend": config.causal_conv_backend,
                "linear_backend": config.linear_attention_linear_backend or config.linear_backend,
            }
        else:
            layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer_spec.submodules.self_attention.module = MiniCPMV46SelfAttention
            layer_spec.submodules.self_attention.params = {
                "attn_mask_type": AttnMaskType.causal,
                "projection_split_mode": "qwen3_5",
                "linear_backend": config.full_attention_linear_backend or config.linear_backend,
            }

        layer_spec.submodules.input_layernorm = layer_norm_impl
        if is_dense:
            layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
        elif (
            hasattr(layer_spec.submodules, "pre_mlp_layernorm")
            and layer_spec.submodules.pre_mlp_layernorm is not IdentityOp
        ):
            layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, "q_layernorm"):
            layer_spec.submodules.self_attention.submodules.q_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, "k_layernorm"):
            layer_spec.submodules.self_attention.submodules.k_layernorm = layer_norm_impl
        layer_spec.submodules.mlp = deepcopy(mlp)
        layer_specs.append(layer_spec)

    local_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
    block_spec = TransformerBlockSubmodules(layer_specs=local_specs, layer_norm=layer_norm_impl)
    mtp_block_spec = None
    if config.mtp_num_layers is not None:
        mtp_input_spec = layer_specs[-1] if not local_specs else block_spec
        mtp_block_spec = get_gpt_mtp_block_spec(
            config,
            mtp_input_spec,
            use_transformer_engine=True,
            vp_stage=vp_stage,
        )
        if mtp_block_spec is not None:
            for layer_spec in mtp_block_spec.layer_specs:
                layer_spec.submodules.enorm = layer_norm_impl
                layer_spec.submodules.hnorm = layer_norm_impl
                layer_spec.submodules.layer_norm = layer_norm_impl
    return block_spec, mtp_block_spec
