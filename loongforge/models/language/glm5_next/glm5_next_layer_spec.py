# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Loong-Megatron layer specification for GLM-5.3-Flash."""

from copy import copy

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerSubmodules,
    DSAttentionSubmodules,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import get_pg_rank

try:
    from .glm5_next_attention import (
        Glm5NextTextLinearAttention,
        KPoolDSAIndexer,
        KPoolDSAttention,
    )
except ImportError:
    from glm5_next_attention import (
        Glm5NextTextLinearAttention,
        KPoolDSAIndexer,
        KPoolDSAttention,
    )


class NativeRMSNormFactory:
    def __new__(cls, config, hidden_size, eps=1e-5, **kwargs):
        norm_config = copy(config)
        norm_config.sequence_parallel = False
        return WrappedTorchNorm(config=norm_config, hidden_size=hidden_size, eps=eps)


def _dense_mlp_spec(backend: LocalSpecProvider) -> ModuleSpec:
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
            activation_func=backend.activation_func(),
        ),
    )


def _moe_spec(config, backend: LocalSpecProvider) -> ModuleSpec:
    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        use_te_activation_func=False,
    )


def _layer_spec(config, layer_index: int, backend: LocalSpecProvider) -> ModuleSpec:
    block_type = config.layer_types[layer_index]
    if block_type == "linear_attention":
        attention = ModuleSpec(module=Glm5NextTextLinearAttention)
    elif block_type == "deepseek_sparse_attention":
        attention = ModuleSpec(
            module=MLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=MLASelfAttentionSubmodules(
                linear_q_proj=TEColumnParallelLinear,
                linear_q_down_proj=TELinear,
                linear_q_up_proj=TEColumnParallelLinear,
                linear_kv_down_proj=TELinear,
                linear_kv_up_proj=TEColumnParallelLinear,
                core_attention=ModuleSpec(
                    module=KPoolDSAttention,
                    params={"use_indexer": config.indexer_types[layer_index] == "full"},
                    submodules=DSAttentionSubmodules(
                        indexer=ModuleSpec(
                            module=KPoolDSAIndexer,
                            submodules=DSAIndexerSubmodules(
                                linear_wq_b=TELinear,
                                linear_wk=TELinear,
                                k_norm=TENorm,
                                linear_weights_proj=TELinear,
                            ),
                        )
                    ),
                ),
                linear_proj=TERowParallelLinear,
                q_layernorm=NativeRMSNormFactory,
                kv_layernorm=NativeRMSNormFactory,
            ),
        )
    else:
        raise ValueError(f"unsupported GLM-5.3-Flash layer type: {block_type}")

    mlp = (
        _moe_spec(config, backend)
        if config.mlp_layer_types[layer_index] == "sparse"
        else _dense_mlp_spec(backend)
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=NativeRMSNormFactory,
            self_attention_hyper_connection=HyperConnectionModule,
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=NativeRMSNormFactory,
            mlp_hyper_connection=HyperConnectionModule,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_glm5_next_decoder_block_spec(config, pg_collection=None, vp_stage=None):
    """Build the local PP slice from native TransformerLayer specifications."""
    backend = LocalSpecProvider()
    layer_specs = [_layer_spec(config, index, backend) for index in range(config.num_layers)]
    pp_group = getattr(pg_collection, "pp", None)
    pp_rank = config.pipeline_rank if config.pipeline_rank is not None else get_pg_rank(pp_group)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    count = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return TransformerBlockSubmodules(
        layer_specs=layer_specs[offset : offset + count],
        layer_norm=NativeRMSNormFactory,
    )


def build_glm5_next_model(config, **kwargs):
    from .glm5_next_model import Glm5NextModel

    return Glm5NextModel(config, **kwargs)
