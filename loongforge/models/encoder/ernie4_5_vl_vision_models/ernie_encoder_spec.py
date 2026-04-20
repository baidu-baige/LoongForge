"""ERNIE-4.5-VL vision encoder layer spec.

Reuses the standard Megatron TransformerLayer plumbing with ERNIE-specific
rotary-position-embedding application (same convention as Qwen2-VL).
"""

import torch
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from loongforge.models.dispatch import multiacc_modules


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    t: torch.Tensor,
    freqs: torch.Tensor,
    config: TransformerConfig,
    cu_seqlens=None,
    rotary_interleaved: bool = False,
    mscale: float = 1.0,
    cp_group=None,
    **kwargs,
) -> torch.Tensor:
    """Apply 2-D rotary position embedding - minimal modification from original.

    Reuses original logic: cos/sin tiled ×2, applied as t * cos + rotate_half(t) * sin.
    Adapts to handle freqs with optional batch/singleton dimensions.
    """
    orig_dtype = t.dtype
    t = t.float()

    # Squeeze unnecessary singleton dimensions from freqs (keep seq_len dim at position 0 or 1)
    # Handles [S, 1, 1, D/2] -> [S, D/2] or [B, S, D/2] unchanged
    while freqs.dim() > 2 and freqs.shape[1] == 1:
        freqs = freqs.squeeze(1)

    # Compute cos/sin with mscale
    cos = (freqs.cos() * mscale).float()
    sin = (freqs.sin() * mscale).float()

    cos = cos.unsqueeze(-2).repeat(1, 1, 2) if cos.dim() == 2 else cos.unsqueeze(-2).repeat(1, 1, 1, 2)
    sin = sin.unsqueeze(-2).repeat(1, 1, 2) if sin.dim() == 2 else sin.unsqueeze(-2).repeat(1, 1, 1, 2)

    # Add batch dimension for broadcasting with t (matches original unsqueeze(0))
    while cos.dim() < t.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # Apply rotary embedding: t * cos + rotate_half(t) * sin
    t = (t * cos) + (_rotate_half(t) * sin)

    return t.to(orig_dtype)


def get_ernie_vl_vision_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """Return the ModuleSpec for one ERNIE ViT transformer layer.
      → pre-attn LayerNorm
      → SelfAttention with QKV bias
      → pre-MLP LayerNorm (fused into fc1 by TE)
      → MLP (QuickGELU)
    """
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=multiacc_modules.TENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=multiacc_modules.TEColumnParallelLinear,
                    core_attention=multiacc_modules.DotProductAttention,
                    linear_proj=multiacc_modules.TERowParallelLinear,
                    apply_rotary_fn=apply_rotary_pos_emb_vision,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=multiacc_modules.TENorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=multiacc_modules.TEColumnParallelLinear,
                    linear_fc2=multiacc_modules.TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
