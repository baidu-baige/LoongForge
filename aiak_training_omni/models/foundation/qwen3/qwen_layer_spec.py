"""Qwen layer spec."""

import torch

from megatron.core import parallel_state
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


def _get_mlp_module_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False
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
                ),
            )
        ),
    )


def get_qwen3_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    # To simplify the code, temporarily remove the compatibility with MoE/MLA.
    # If there is a new version in the future, add and test it separately.
    assert (
        not config.multi_latent_attention
    ), "Not supporting multi-latent attention for Qwen model yet."

    mlp = _get_mlp_module_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )

    # TENorm significantly harms convergence when used for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    qk_norm = (
        multiacc_modules.TENorm
        if is_te_min_version("1.9.0")
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
            pre_mlp_layernorm=(
                multiacc_modules.TENorm if config.num_moe_experts else IdentityOp
            ),
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
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


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_mrope_bshd(t, freq, config, cu_seqlens,):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freq (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    # t [84, 1, 32, 128]
    # freq [1, 1, 84, 128]
    rot_dim = freq.shape[-1]
    
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t = t.permute(1, 2, 0, 3).contiguous()
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freq).to(t.dtype) # [1, 1, 84, 128]
    sin_ = torch.sin(freq).to(t.dtype) # [1, 1, 84, 128]

    # t [84, 1, 32, 128] -> [1, 32, 84, 128]
    t = (t * cos_) + (_rotate_half(t) * sin_)

    t = torch.cat((t, t_pass), dim=-1).permute(2, 0, 1, 3).contiguous()
    return t
    

def apply_mrope(t, freq, config, cu_seqlens=None):
    """ mrope """
    if cu_seqlens is not None:
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        return torch.cat(
            [
                _apply_mrope_bshd(
                    x.unsqueeze(1),
                    freq[int(cu_seqlens[i]) : int(cu_seqlens[i]) + x.size(0)],
                    config,
                    cu_seqlens,
                )
                for i, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)
    else:
        return _apply_mrope_bshd(t, freq, config, cu_seqlens)


def get_qwen3_vl_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    # To simplify the code, temporarily remove the compatibility with MoE/MLA.
    # If there is a new version in the future, add and test it separately.
    assert (
        not config.multi_latent_attention
    ), "Not supporting multi-latent attention for Qwen model yet."

    mlp = _get_mlp_module_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )

    # TENorm significantly harms convergence when used for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    qk_norm = (
        multiacc_modules.TENorm
        if is_te_min_version("1.9.0")
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
                    apply_rotary_fn=apply_mrope,
                ),
            ),
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            pre_mlp_layernorm=(
                multiacc_modules.TENorm if config.num_moe_experts else IdentityOp
            ),
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
    )
