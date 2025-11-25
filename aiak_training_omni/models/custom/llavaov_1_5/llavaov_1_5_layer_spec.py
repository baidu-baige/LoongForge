"""LLaVA OneVision 1.5  layer spec."""

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules

from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

from aiak_training_omni.utils import is_te_min_version
from aiak_training_omni.models.dispatch import multiacc_modules
from megatron.core.extensions.transformer_engine import (
    TELinear,
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from aiak_training_omni.models.custom.local_layers.local_norm import LocalNorm
import torch
from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add


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
                linear_fc1=multiacc_modules.TELayerNormColumnParallelLinear,
                linear_fc2=multiacc_modules.TERowParallelLinear,
                bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
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
                    bias_activation_func_impl=multiacc_modules.bias_activation_func_impl,
                )
            )
        )
    )

def get_llava_ov_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    # To simplify the code, temporarily remove the compatibility with MoE/MLA.
    # If there is a new version in the future, add and test it separately.
    assert not config.multi_latent_attention, "Not supporting multi-latent attention for Qwen model yet."

    mlp = _get_mlp_module_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )

    # TENorm significantly harms convergence when used for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    # qk_norm = multiacc_modules.TENorm if is_te_min_version("1.9.0") else multiacc_modules.LocalNorm
    qk_norm = multiacc_modules.LocalNorm

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
            pre_mlp_layernorm=(
                multiacc_modules.TENorm if config.num_moe_experts else IdentityOp
            ),
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        )
    )