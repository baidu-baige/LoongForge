# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 layer spec.

DeepSeek-V4 owns its model-specific hybrid attention (CSA/HCA) in Omni.  The
remaining Megatron dependencies here are generic transformer/MoE/MTP runtime
building blocks.
"""

from typing import Optional, Tuple

from omegaconf import ListConfig

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from loongforge.models.foundation.deepseek_v4.deepseek_v4_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSubmodules,
)
from loongforge.models.foundation.deepseek_v4.deepseek_v4_csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)


def _get_mlp_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Build the dense or MoE MLP spec used by DeepSeek-V4."""
    activation_func = backend.activation_func() if use_te_activation_func else None

    if num_experts is None:
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
                activation_func=activation_func,
            ),
        )

    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_activation_func=use_te_activation_func,
    )


def _get_deepseek_v4_layer_with_te_spec(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    moe_use_legacy_grouped_gemm: bool = False,
    qk_layernorm: bool = False,
    enable_hyper_connection: bool = False,
    normalization: str = "RMSNorm",
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Build one DeepSeek-V4 transformer layer spec with Omni-owned DSv4 attention."""
    mlp = _get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_activation_func=use_te_activation_func,
    )

    rms_norm = normalization == "RMSNorm"
    qk_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=True) if qk_layernorm else IdentityOp
    hc_module = HyperConnectionModule if enable_hyper_connection else IdentityOp

    compressor_spec = ModuleSpec(
        module=Compressor,
        submodules=CompressorSubmodules(
            linear_wkv=backend.linear(),
            linear_wgate=backend.linear(),
            norm=backend.layer_norm(rms_norm=True, for_qk=False),
        ),
    )

    indexer_spec = ModuleSpec(
        module=CSAIndexer,
        submodules=CSAIndexerSubmodules(
            linear_wq_b=backend.linear(),
            linear_weights_proj=backend.linear(),
            compressor=compressor_spec,
        ),
    )

    core_attention = ModuleSpec(
        module=CompressedSparseAttention,
        submodules=CompressedSparseAttentionSubmodules(
            compressor=compressor_spec,
            indexer=indexer_spec,
        ),
    )

    attention = ModuleSpec(
        module=DSv4HybridSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DSv4HybridSelfAttentionSubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_linear(),
            linear_kv_proj=backend.linear(),
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend.layer_norm(),
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hc_module,
            pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=hc_module,
        ),
    )


def _get_moe_layer_pattern(config: TransformerConfig):
    """Return the dense/MoE pattern for decoder layers."""
    if isinstance(config.moe_layer_freq, ListConfig):
        config.moe_layer_freq = list(config.moe_layer_freq)

    if isinstance(config.moe_layer_freq, int):
        return [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)]

    if isinstance(config.moe_layer_freq, list):
        assert len(config.moe_layer_freq) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(config.moe_layer_freq)}, "
            f"expected {config.num_layers}, current moe layer pattern: {config.moe_layer_freq}"
        )
        return config.moe_layer_freq

    raise ValueError(f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}")


def get_deepseek_v4_decoder_block_and_mtp_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
) -> Tuple[TransformerBlockSubmodules, Optional[ModuleSpec]]:
    """Return (decoder block spec, MTP block spec) for DeepSeek-V4."""
    assert config.num_moe_experts and config.num_moe_experts > 0, "DeepSeek-V4 requires MoE"
    assert config.multi_latent_attention, "DeepSeek-V4 requires MLA"
    assert getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid", (
        "DeepSeek-V4 requires experimental_attention_variant='dsv4_hybrid'"
    )

    use_te = getattr(config, "transformer_impl", "transformer_engine") == "transformer_engine"
    assert use_te, "DeepSeek-V4 hybrid attention currently requires transformer_engine specs"

    backend = TESpecProvider()

    dense_layer_spec = _get_deepseek_v4_layer_with_te_spec(
        backend=backend,
        num_experts=None,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        enable_hyper_connection=config.enable_hyper_connections,
        normalization=config.normalization,
        use_te_activation_func=config.use_te_activation_func,
    )
    moe_layer_spec = _get_deepseek_v4_layer_with_te_spec(
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        enable_hyper_connection=config.enable_hyper_connections,
        normalization=config.normalization,
        use_te_activation_func=config.use_te_activation_func,
    )

    moe_layer_pattern = _get_moe_layer_pattern(config)

    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder,
                vp_stage=vp_stage,
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=backend.layer_norm(),
    )

    mtp_block_spec = None
    if config.mtp_num_layers and config.mtp_num_layers > 0:
        if hasattr(block_spec, "layer_specs") and len(block_spec.layer_specs) == 0:
            transformer_layer_spec_for_mtp = _get_deepseek_v4_layer_with_te_spec(
                backend=backend,
                num_experts=config.num_moe_experts,
                moe_grouped_gemm=config.moe_grouped_gemm,
                moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
                qk_layernorm=config.qk_layernorm,
                enable_hyper_connection=config.enable_hyper_connections,
                normalization=config.normalization,
                use_te_activation_func=config.use_te_activation_func,
            )
        else:
            transformer_layer_spec_for_mtp = block_spec

        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config,
            spec=transformer_layer_spec_for_mtp,
            use_transformer_engine=use_te,
            vp_stage=vp_stage,
        )

    return block_spec, mtp_block_spec
