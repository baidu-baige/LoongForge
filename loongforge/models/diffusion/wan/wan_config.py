# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""wan model config"""

from dataclasses import dataclass
from loongforge.models.common import BaseModelStditConfig


@dataclass
class WanConfig(BaseModelStditConfig):
    """configuration for Wan model

    The fields need to be consistent with the definitions in args
    """
    latent_in_channels: int
    latent_out_channels: int
    latent_patch_size: tuple
    latent_space_scale: float
    latent_time_scale: float
    num_layers: int
    hidden_size: int
    # kv_channels: int
    ffn_hidden_size: int
    num_attention_heads: int
    model_type: str
    # WAN-specific fields
    text_dim: int = 4096
    has_image_input: bool = False
    # Number of CLIP image tokens prepended to text tokens for I2V cross-attention.
    # Wan2.1 uses CLIP ViT-H/14: (224/14)**2 + 1 = 257.
    clip_num_image_tokens: int = 257
    in_dim: int = 36
    freq_dim: int = 256
    out_dim: int = 16
    norm_epsilon: float = 1e-06
    has_image_pos_emb: bool = False
    require_clip_embedding: bool = True
    group_query_attention: bool = False
    num_query_groups: int = 1
    position_embedding_type: str = "learned_absolute"
    rotary_interleaved: bool = False
    use_fused_wan_rope: bool = False
    normalization: str = "RMSNorm"

    vae_temporal_compress: int = 4
    vae_spatial_compress: int = 8

    swiglu: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = True
    add_position_embedding: bool = True
    attention_softmax_in_fp32: bool = True
