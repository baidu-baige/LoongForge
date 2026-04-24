# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Deepseek layer spec."""

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
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)

from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.enums import Fp8Recipe

from loongforge.models.dispatch import multiacc_modules
from loongforge.utils import get_args

def _get_deepseek_layer_with_te_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = True,
    qk_layernorm: Optional[bool] = False,
    experimental_attention_variant: Optional[str] = None,
) -> ModuleSpec:
    """Get the transformer layer spec for deepseek
    
    Args:
        experimental_attention_variant (str, optional): The type of experimental attention variant.
                                                        Defaults to None.
    """
    mlp = _get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    # Default specs for attention module and its submodules, which can be overridden by experimental attention variants.
    mla_attention_module = MLASelfAttention
    core_attention = multiacc_modules.DotProductAttention
    linear_q_up_proj = (multiacc_modules.TELayerNormColumnParallelLinear if qk_layernorm
                        else multiacc_modules.TEColumnParallelLinear)
    linear_kv_up_proj = (multiacc_modules.TELayerNormColumnParallelLinear if qk_layernorm
                         else multiacc_modules.TEColumnParallelLinear)
    q_layernorm = IdentityOp
    kv_layernorm = IdentityOp

    # Override the attention module and its submodules if using experimental attention variant.
    if experimental_attention_variant is not None:
        # Deepseek Sparse Attention (DSA)
        if experimental_attention_variant == "dsa":
            use_dsa_fused = getattr(get_args(), "use_dsa_fused", False)
            # By default, use the naive MegatronLM implementation of DSA.
            if not use_dsa_fused:
                from megatron.core.transformer.experimental_attention_variant.dsa import (
                    DSAIndexer,
                    DSAIndexerSubmodules,
                    DSAttention,
                    DSAttentionSubmodules,
                )
                indexer_module = DSAIndexer
                indexer_submodules = DSAIndexerSubmodules
                attention_module = DSAttention
                attention_submodules = DSAttentionSubmodules
            # Use Omni optimized fused implementation of DSA, which has better performance.
            else:
                from loongforge.models.common.experimental_attention_variant import (
                    DSAIndexerFused,
                    DSAIndexerFusedSubmodules,
                    DSAttentionFused,
                    DSAttentionFusedSubmodules,
                    MLASelfAttentionFused,
                )
                indexer_module = DSAIndexerFused
                indexer_submodules = DSAIndexerFusedSubmodules
                attention_module = DSAttentionFused
                attention_submodules = DSAttentionFusedSubmodules
                mla_attention_module = MLASelfAttentionFused

            core_attention = ModuleSpec(
                module=attention_module,
                submodules=attention_submodules(
                    indexer=ModuleSpec(
                        module=indexer_module,
                        submodules=indexer_submodules(
                            linear_wq_b=multiacc_modules.TELinear,
                            linear_wk=multiacc_modules.TELinear,
                            k_norm=multiacc_modules.TENorm,
                            linear_weights_proj=multiacc_modules.TELinear,
                        ),
                    )
                ),
            )
            linear_q_up_proj = multiacc_modules.TEColumnParallelLinear
            linear_kv_up_proj = multiacc_modules.TEColumnParallelLinear
            q_layernorm = (multiacc_modules.TENorm if qk_layernorm else IdentityOp)
            kv_layernorm = (multiacc_modules.TENorm if qk_layernorm else IdentityOp)

        # Currently not support other experimental attention variants.
        else:
            raise ValueError(
                f"Invalid experimental attention variant: {experimental_attention_variant}"
            )

    attention = ModuleSpec(
        module=mla_attention_module,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=multiacc_modules.TEColumnParallelLinear,
            linear_q_down_proj=multiacc_modules.TEColumnParallelLinear,
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=multiacc_modules.TEColumnParallelLinear,
            linear_kv_up_proj=linear_kv_up_proj,
            core_attention=core_attention,
            linear_proj=multiacc_modules.TERowParallelLinear,
            q_layernorm=q_layernorm,
            kv_layernorm=kv_layernorm,
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=multiacc_modules.TENorm,
            self_attention=attention,
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

    # share expert
    shared_linear_fc1 = multiacc_modules.TEColumnParallelLinear
    shared_linear_fc2 = multiacc_modules.TERowParallelLinear

    return ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            shared_experts=ModuleSpec(
                module=SharedExpertMLP,
                params={"gate": False},
                submodules=MLPSubmodules(
                    linear_fc1=shared_linear_fc1,
                    linear_fc2=shared_linear_fc2,
                ),
            ),
            experts=ModuleSpec(
                module=expert_module,
                submodules=MLPSubmodules(
                    linear_fc1=linear_fc1,
                    linear_fc2=linear_fc2,
                ),
            ),
        ),
    )


def get_deepseek_decoder_block_and_mtp_spec(
    config: TransformerConfig, 
    vp_stage: int = None,
) -> Tuple[TransformerBlockSubmodules, Optional[ModuleSpec]]:
    """Get the deepseek decoder block and multi-token prediction layer spec."""
    assert config.num_moe_experts > 0, "Only support MOE when using DeepSeek"
    assert (
        config.multi_latent_attention
    ), "Only support multi-latent attention when using DeepSeek"

    block_spec = None
    mtp_block_spec = None
    use_te = config.transformer_impl == "transformer_engine"

    # Layer specs.
    dense_layer_spec = _get_deepseek_layer_with_te_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        experimental_attention_variant=config.experimental_attention_variant,
    )

    moe_layer_spec = _get_deepseek_layer_with_te_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        experimental_attention_variant=config.experimental_attention_variant,
    )

    # In FP8 training, replace `linear_q_down_proj` and `linear_kv_down_proj`
    # with TELinear to support tensor parallelism
    if config.fp8 and config.fp8_recipe == Fp8Recipe.blockwise:
        dense_layer_spec.submodules.self_attention.submodules.linear_q_down_proj = (
            multiacc_modules.TELinear
        )
        dense_layer_spec.submodules.self_attention.submodules.linear_kv_down_proj = (
            multiacc_modules.TELinear
        )
        moe_layer_spec.submodules.self_attention.submodules.linear_q_down_proj = (
            multiacc_modules.TELinear
        )
        moe_layer_spec.submodules.self_attention.submodules.linear_kv_down_proj = (
            multiacc_modules.TELinear
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
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

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
                transformer_layer_spec_for_mtp = _get_deepseek_layer_with_te_spec(
                    num_experts=config.num_moe_experts,
                    moe_grouped_gemm=config.moe_grouped_gemm,
                    qk_layernorm=config.qk_layernorm,
                    experimental_attention_variant=config.experimental_attention_variant,
                )
        else:
            transformer_layer_spec_for_mtp = block_spec
        
        # TODO: use get_gpt_mtp_block_spec or not? 
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
        )

    return block_spec, mtp_block_spec
