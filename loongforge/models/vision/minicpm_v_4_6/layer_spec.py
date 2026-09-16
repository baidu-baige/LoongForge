# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V-4.6 vision Transformer layer specification."""

import inspect

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from loongforge.models.dispatch import multiacc_modules


class MiniCPMV46TEDotProductAttention(TEDotProductAttention):
    """Keep only packed-sequence arguments supported by the installed TE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from transformer_engine.pytorch.attention.dot_product_attention import (
                DotProductAttention as TEDotProductAttentionImpl,
            )

            supported = set(inspect.signature(TEDotProductAttentionImpl.forward).parameters)
        except (ImportError, TypeError, ValueError):
            supported = None
        if supported is not None:
            self.kept_packed_seq_params.intersection_update(supported)


class MiniCPMV46TEDuplicatedLinear(TELinear):
    """TP1-local TE linear matching the HF vision tower's dense GEMMs."""

    def __init__(self, input_size, output_size, **kwargs):
        kwargs.pop("gather_output", None)
        kwargs.pop("input_is_parallel", None)
        kwargs["parallel_mode"] = "duplicated"
        kwargs["tp_group"] = None
        kwargs["skip_bias_add"] = False
        kwargs["skip_weight_param_allocation"] = False
        super().__init__(input_size, output_size, **kwargs)

def get_minicpm_v_4_6_vision_layer_spec(config) -> ModuleSpec:
    """Build MiniCPM ViT blocks with the standard Megatron/TE backend."""
    del config
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=multiacc_modules.TENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=MiniCPMV46TEDuplicatedLinear,
                    core_attention=MiniCPMV46TEDotProductAttention,
                    linear_proj=MiniCPMV46TEDuplicatedLinear,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=multiacc_modules.TENorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=MiniCPMV46TEDuplicatedLinear,
                    linear_fc2=MiniCPMV46TEDuplicatedLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
