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
    caption_channels: int
    num_attention_heads: int
    model_type: str
    norm_epsilon: float = 1e-06
    group_query_attention: bool = False
    num_query_groups: int = 1
    position_embedding_type: str = "learned_absolute"
    rotary_interleaved: bool = False
    normalization: str = "RMSNorm"

    swiglu: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = True
    add_position_embedding: bool = True
    attention_softmax_in_fp32: bool = True
