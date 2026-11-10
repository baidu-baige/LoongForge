# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ERNIE layers built from LoongForge and Megatron modules."""

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TELinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, DSAIndexerSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from loongforge.models.foundation.deepseek_v4.deepseek_v4_layer_spec import _get_deepseek_v4_layer_with_te_spec

from .ernie_attention import (
    Ernie55DAttention,
    Ernie55DCoreSubmodules,
    Ernie55HAttention,
    Ernie55HCore,
    Ernie55IndexerNorm,
)
from .ernie_moe import Ernie55MoELayer
from .ernie_ops import Ernie55HyperConnection, Ernie55Norm, Ernie55SiTU


def get_ernie55_layer_spec(config, vp_stage=None):
    layers = []
    activation_func = Ernie55SiTU if config.activation_situ_beta is not None else None
    for index, ratio in enumerate(config.compress_ratios):
        if ratio == 128:
            attention = _get_deepseek_v4_layer_with_te_spec(
                TESpecProvider(), qk_layernorm=True
            ).submodules.self_attention
            attention.module = Ernie55HAttention
            attention.submodules.core_attention.module = Ernie55HCore
        else:
            indexer = ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=TELinear,
                    linear_wk=TELinear,
                    k_norm=Ernie55IndexerNorm,
                    linear_weights_proj=TELinear,
                ),
            )
            attention = ModuleSpec(
                module=Ernie55DAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=Ernie55DCoreSubmodules(indexer=indexer),
            )
        dense_submodules = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
            activation_func=activation_func,
        )
        if index < config.first_k_dense_replace:
            mlp = ModuleSpec(module=MLP, submodules=dense_submodules)
        else:
            mlp = ModuleSpec(
                module=Ernie55MoELayer,
                submodules=MoESubmodules(
                    experts=ModuleSpec(
                        module=TEGroupedMLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TEColumnParallelGroupedLinear,
                            linear_fc2=TERowParallelGroupedLinear,
                            activation_func=activation_func,
                        ),
                    ),
                    shared_experts=ModuleSpec(
                        module=SharedExpertMLP,
                        params={"gate": False},
                        submodules=dense_submodules,
                    ),
                ),
            )
        layers.append(
            ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=Ernie55Norm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                    self_attention_hyper_connection=Ernie55HyperConnection,
                    pre_mlp_layernorm=Ernie55Norm,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    mlp_hyper_connection=Ernie55HyperConnection,
                ),
            )
        )
    if config.pipeline_model_parallel_layout is not None:
        local_layers = [
            layers[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder,
                vp_stage=vp_stage,
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layers = layers[offset : offset + get_num_layers_to_build(config, vp_stage=vp_stage)]

    block_spec = TransformerBlockSubmodules(layer_specs=local_layers, layer_norm=Ernie55Norm)
    mtp_spec = get_gpt_mtp_block_spec(
        config,
        block_spec,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )
    return block_spec, mtp_spec
