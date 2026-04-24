# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Minimax layer spec."""

from typing import Tuple, Optional
from omegaconf import ListConfig

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
    TransformerLayerSubmodules
)

from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules

from .attention import MinimaxSelfAttention, MinimaxSelfAttentionSubmodules
from loongforge.models.dispatch import multiacc_modules
from loongforge.utils import is_te_min_version



def _get_minimax_layer_with_te_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = True,
    qk_layernorm: Optional[bool] = False,
) -> ModuleSpec:
    """Get the transformer layer spec for minimax"""

    mlp = _get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )
    qk_norm = multiacc_modules.TENorm if is_te_min_version("1.9.0") else multiacc_modules.LocalNorm
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=MinimaxSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MinimaxSelfAttentionSubmodules(
                    linear_qkv=multiacc_modules.TELayerNormColumnParallelLinear,
                    core_attention=multiacc_modules.DotProductAttention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    q_layernorm=qk_norm,
                    k_layernorm=qk_norm,
                    apply_rotary_fn=multiacc_modules.apply_rotary_pos_emb,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=multiacc_modules.TENorm,
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        )
    )


def _get_mlp_module_spec(
    num_experts: int=None,
    moe_grouped_gemm: bool=False
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    if num_experts is None:
        # Dense MLP w/ TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=multiacc_modules.TEColumnParallelLinear,
                linear_fc2=multiacc_modules.TERowParallelLinear,
            ),
        )

    # moe mlp
    if moe_grouped_gemm:
        # use TEGroupedLinear
        assert multiacc_modules.TEColumnParallelGroupedLinear is not None
        expert_module = TEGroupedMLP
        linear_fc1 = multiacc_modules.TEColumnParallelGroupedLinear
        linear_fc2 = multiacc_modules.TERowParallelGroupedLinear

    else:
        expert_module = SequentialMLP
        linear_fc1 = multiacc_modules.TEColumnParallelLinear
        linear_fc2 = multiacc_modules.TERowParallelLinear


    return ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=ModuleSpec(
                module=expert_module,
                submodules=MLPSubmodules(
                    linear_fc1=linear_fc1,
                    linear_fc2=linear_fc2,
                ),
            ),
        ),
    )


def get_minimax_decoder_block_and_mtp_spec(
    config: TransformerConfig, 
    vp_stage: int = None,
) -> Tuple[TransformerBlockSubmodules, Optional[ModuleSpec]]:
    """Get the minimax decoder block and multi-token prediction layer spec."""
    assert config.num_moe_experts > 0, "Only support MOE when using Minimax"

    block_spec = None
    mtp_block_spec = None
    use_te = config.transformer_impl == "transformer_engine"

    # Layer specs.
    moe_layer_spec = _get_minimax_layer_with_te_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
    )


    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).

    # compatibility for hydra config
    if isinstance(config.moe_layer_freq, ListConfig):
        config.moe_layer_freq = list(config.moe_layer_freq)

    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0
            for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        layer_specs.append(moe_layer_spec)

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=multiacc_modules.TENorm, # TODO: Whether the Local Norm should be compatible
    )

    # MTP spec
    if config.mtp_num_layers is not None:
        if hasattr(block_spec, 'layer_specs') and len(block_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_minimax_layer_with_te_spec(
                    num_experts=config.num_moe_experts,
                    moe_grouped_gemm=config.moe_grouped_gemm,
                    qk_layernorm=config.qk_layernorm,
                )
        else:
            transformer_layer_spec_for_mtp = block_spec
        
        # TODO: use get_gpt_mtp_block_spec or not? 
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
        )

    return block_spec, mtp_block_spec
