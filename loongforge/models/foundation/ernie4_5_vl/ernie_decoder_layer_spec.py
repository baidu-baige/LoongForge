# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Ernie-VL layer spec."""

from typing import Optional, Tuple
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.attention import SelfAttentionSubmodules, SelfAttention
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules, get_transformer_layer_offset
from loongforge.models.common.local_layers.local_norm import LocalNorm
from loongforge.models.dispatch import multiacc_modules
from loongforge.models.encoder.ernie4_5_vl_vision_models.ernie_adapter import RMSNorm as _AdapterRMSNorm
from loongforge.models.dispatch import multiacc_modules
from .ernie_transformer_layer import TransformerLayerErnie
from .ernie_config import ErnieMoeConfig
from .ernie_moe_layer import ErnieMultiTypeMoE, MultiTypeMoeSubmodules, ErnieMoeLayer
from .ernie_experts import ErnieMLP, SequentialMLP, ErnieSharedExpertMLP
from .ernie_pos_embedding import apply_rotary_3d


class RMSNorm(_AdapterRMSNorm):
    """Wrapper around adapter RMSNorm to accept Megatron build_module calling convention.

    Megatron's build_module passes: RMSNorm(config=config, hidden_size=H, eps=e)
    Adapter RMSNorm expects:        RMSNorm(hidden_size, rms_norm_eps=e)
    """

    def __init__(self, config, hidden_size: int, eps: float = 1e-5, **kwargs):
        super().__init__(hidden_size, rms_norm_eps=eps)


def _get_mlp_module_spec(
    num_experts: int=None,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    if num_experts is None:
        # Dense MLP w/ TE modules.
        return ModuleSpec(
            module=ErnieMLP,
            submodules=MLPSubmodules(
                linear_fc1=multiacc_modules.TEColumnParallelLinear,
                linear_fc2=multiacc_modules.TERowParallelLinear,
            ),
        )
    base_moe = ModuleSpec(
        module=ErnieMoeLayer,
        submodules=MoESubmodules(
            experts=ModuleSpec(
                module=SequentialMLP,
                submodules=MLPSubmodules(
                    linear_fc1=multiacc_modules.TEColumnParallelLinear,
                    linear_fc2=multiacc_modules.TERowParallelLinear,
                )
            )
        )
    )
    # moe mlp
    return ModuleSpec(
        module=ErnieMultiTypeMoE,
        submodules=MultiTypeMoeSubmodules(
            vision_moe_layer=base_moe,
            text_moe_layer=base_moe,
            shared_experts=ModuleSpec(
                module=ErnieSharedExpertMLP,
                params={"gate": False},
                submodules=MLPSubmodules(
                    linear_fc1=multiacc_modules.TEColumnParallelLinear,
                    linear_fc2=multiacc_modules.TERowParallelLinear,
                )
            ),
        )
    )


def _get_ernie4_5_vl_moedecoderlayer_with_spec(num_experts=0, qk_layernorm: bool = False) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    mlp_dense_or_sparse = _get_mlp_module_spec(
        num_experts=num_experts,
    )

    return ModuleSpec(
        module=TransformerLayerErnie,
        submodules=TransformerLayerSubmodules(
            input_layernorm=RMSNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=multiacc_modules.TEColumnParallelLinear,
                    core_attention=multiacc_modules.DotProductAttention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    apply_rotary_fn=apply_rotary_3d

                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=RMSNorm,
            mlp=mlp_dense_or_sparse,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_ernie4_5_vl_decoder_spec(
    config: ErnieMoeConfig,
) -> Tuple[TransformerBlockSubmodules, Optional[ModuleSpec]]:
    """Get the ernie4.5vl layer spec."""
    assert config.num_moe_experts > 0, "num_moe_experts must be greater than 0 in ERNIE-4.5-VL!"

    mlp_layer_types = []
    # ernie4.5_vl uses dense or sparse mlp in different layers
    for i in range(config.num_layers):
        if i in config.dense_layer_index:
            mlp_layer_types.append("dense")
        else:
            mlp_layer_types.append("sparse")

    # dense specs.
    dense_layer_spec = _get_ernie4_5_vl_moedecoderlayer_with_spec(num_experts=None)
    # Layer specs
    moe_layer_spec = _get_ernie4_5_vl_moedecoderlayer_with_spec(num_experts=config.num_moe_experts)

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if mlp_layer_types[layer_number] == "dense":
            layer_specs.append(dense_layer_spec)
        elif mlp_layer_types[layer_number] == "sparse":
            layer_specs.append(moe_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: { mlp_layer_types[layer_number]}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=RMSNorm,
    )
    return block_spec
