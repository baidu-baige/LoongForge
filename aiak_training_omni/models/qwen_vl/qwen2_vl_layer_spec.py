"""Qwen2-VL layer spec."""

import torch
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.extensions.transformer_engine import (
    TELinear,
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)

from aiak_training_omni.models.custom.common.local_norm import LocalNorm
from .adapter import AdapterSubmodules
from .vision_model import apply_rotary_pos_emb_vision
from .qwen2_vl_model import apply_mrope


def get_vision_layer_with_spec() -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    apply_rotary_fn=apply_rotary_pos_emb_vision,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_adapeter_layer_with_spec() -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return AdapterSubmodules(
        layernorm=LocalNorm,
        linear_fc1=TELinear,
        linear_fc2=TELinear,
    )


def get_qwen_layer_with_spec(qk_layernorm: bool = False) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=LocalNorm if qk_layernorm else IdentityOp,
                    k_layernorm=LocalNorm if qk_layernorm else IdentityOp,
                    apply_rotary_fn=apply_mrope,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )
