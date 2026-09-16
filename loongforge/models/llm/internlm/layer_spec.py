# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""InternLM layer spec."""

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from loongforge.utils import is_te_min_version
from loongforge.models.dispatch import multiacc_modules


def get_internlm_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    # To simplify the code, temporarily remove the compatibility with MoE/MLA.
    # If there is a new LLaMA version in the future, add and test it separately.
    assert config.num_moe_experts is None, "Not support MoE for Llama model yet."
    assert (
        not config.multi_latent_attention
    ), "Not support multi-latent attention for Llama model yet."

    # Dense MLP w/ TE modules.
    dense_mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=multiacc_modules.TELayerNormColumnParallelLinear,
            linear_fc2=multiacc_modules.TERowParallelLinear,
        ),
    )

    # TENorm significantly harms convergence when used for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    qk_norm = (
        multiacc_modules.TENorm
        if is_te_min_version("1.9.0")
        and config.normalization in ["LayerNorm", "RMSNorm"]
        else multiacc_modules.LocalNorm
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=multiacc_modules.TELayerNormColumnParallelLinear,
                    core_attention=multiacc_modules.DotProductAttention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    q_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=dense_mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
    )
