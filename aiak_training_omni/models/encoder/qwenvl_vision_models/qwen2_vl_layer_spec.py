"""Qwen2-VL layer spec."""

import torch
from typing import Union
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.extensions.transformer_engine import (
    TELinear,
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from aiak_training_omni.models.custom.local_layers.local_norm import LocalNorm

# from .adapter import AdapterSubmodules
# from .vision_model import apply_rotary_pos_emb_vision
from megatron.core import parallel_state
from dataclasses import dataclass


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""

    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_mrope_bshd(t, freq, config, cu_seqlens=None, mrope_section=[16, 24, 24]):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    (https://qwenlm.github.io/blog/qwen2-vl/).
    Args:
        t (torch.Tensor): Input tensor of shape [S, B, heads, dim]
        freq (torch.Tensor): Frequency tensor of shape [S, B, 3, dim]
    """
    cos = freq.cos().to(dtype=t.dtype)
    sin = freq.sin().to(dtype=t.dtype)
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


def apply_mrope(t, freq, config, cu_seqlens=None, mrope_section=[16, 24, 24]):
    """mrope"""
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
                    mrope_section,
                )
                for i, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)
    else:
        return _apply_mrope_bshd(t, freq, config, cu_seqlens, mrope_section)


def apply_rotary_pos_emb_vision(
    t, freqs, config, cu_seqlens=None, rotary_interleaved=False
):
    """ " Apply rotation to positional embedding"""
    orig_dtype = t.dtype
    t = t.float()
    if cu_seqlens is not None:
        freqs = freqs.squeeze(1)
        cos_ = freqs.cos().float().repeat(1, 1, 2)
        sin_ = freqs.sin().float().repeat(1, 1, 2)
    else:
        cos_ = freqs.cos().float().repeat(1, 1, 1, 2)
        sin_ = freqs.sin().float().repeat(1, 1, 1, 2)
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return t.to(orig_dtype)


def get_vision_layer_with_spec(config: TransformerConfig) -> ModuleSpec:
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


def get_adapeter_layer_with_spec(config: TransformerConfig) -> ModuleSpec:
    """Use this spec for an implementation using transformer, local or multi-accel engine."""
    return AdapterSubmodules(
        layernorm=LocalNorm,
        linear_fc1=TELinear,
        linear_fc2=TELinear,
    )
