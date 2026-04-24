# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Qwen layer spec."""

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


from loongforge.utils import is_te_min_version
from loongforge.models.dispatch import multiacc_modules
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from loongforge.models.common.local_layers.local_norm import LocalNorm
import torch
from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add


def _get_mlp_module_spec() -> ModuleSpec:
    """Helper function to get module spec for MLP"""
    # Dense MLP w/ TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=multiacc_modules.TELayerNormColumnParallelLinear,
            linear_fc2=multiacc_modules.TERowParallelLinear,
        ),
    )


def get_qwen2_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    # To simplify the code, temporarily remove the compatibility with MoE/MLA.
    # If there is a new version in the future, add and test it separately.
    assert (
        not config.multi_latent_attention
    ), "Not supporting multi-latent attention for Qwen model yet."

    mlp = _get_mlp_module_spec()

    # TENorm significantly harms convergence when used for QKLayerNorm if TE Version < 1.9;
    # we instead use the Apex implementation.
    qk_norm = (
        multiacc_modules.TENorm
        if is_te_min_version("1.9.0")
        and config.normalization in ["LayerNorm", "RMSNorm"]
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


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_mrope_bshd(
    t, freq, config, cu_seqlens=None, mrope_section=[16, 24, 24], mscale: float = 1.0
):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    (https://qwenlm.github.io/blog/qwen2-vl/).
    Args:
        t (torch.Tensor): Input tensor of shape [S, B, heads, dim]
        freq (torch.Tensor): Frequency tensor of shape [S, B, 3, dim]
    """
    cos = (freq.cos() * mscale).to(dtype=t.dtype)
    sin = (freq.sin() * mscale).to(dtype=t.dtype)
    mrope_section = mrope_section * 2

    cos = torch.cat(
        [m[..., i % 3, :] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(2)
    sin = torch.cat(
        [m[..., i % 3, :] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(2)

    t = (t * cos) + (_rotate_half(t) * sin)
    return t


def apply_mrope(
    t,
    freq,
    config,
    cu_seqlens=None,
    mrope_section=[16, 24, 24],
    mscale: float = 1.0,
    cp_group=None,
):
    """mrope"""
    if cu_seqlens is not None:
        cp_size = (
            cp_group.size()
            if cp_group is not None
            else parallel_state.get_context_parallel_world_size()
        )
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        return torch.cat(
            [
                _apply_mrope_bshd(
                    x.unsqueeze(1),
                    freq[int(cu_seqlens[i]) : int(cu_seqlens[i]) + x.size(0)],
                    config,
                    cu_seqlens,
                    mrope_section,
                    mscale,
                )
                for i, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)
    else:
        return _apply_mrope_bshd(t, freq, config, cu_seqlens, mrope_section, mscale)


def get_qwen2_vl_layer_with_te_spec(config: TransformerConfig) -> ModuleSpec:
    """
    Use this spec for an implementation using transformer, local or multi-accel engine
    """
    qk_norm = (
        multiacc_modules.TENorm
        if is_te_min_version("1.9.0")
        and config.normalization in ["LayerNorm", "RMSNorm"]
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
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
                    k_layernorm=qk_norm if config.qk_layernorm else IdentityOp,
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
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )
