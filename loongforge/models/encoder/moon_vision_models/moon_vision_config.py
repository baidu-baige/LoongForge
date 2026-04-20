# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register moon vision model with different config"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
from loongforge.models.common.base_model_config import BaseModelConfig


@dataclass
class MoonVisionModelConfig(BaseModelConfig):
    """configuration for moon vision model (Kimi-K2.5)"""
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    kv_channels: int

    num_query_groups: Optional[int] = None
    activation_func: Optional[torch.nn.Module] = None
    normalization: str = "LayerNorm"
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    swiglu: bool = False
    in_channels: int = 3
    group_query_attention: bool = False
    gated_linear_unit: bool = False
    position_embedding_type: str = "none"
    bias_activation_fusion: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    bias_dropout_fusion: bool = False
    apply_rope_fusion: bool = False

    patch_size: int = 14
    temporal_patch_size: int = 1
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    vision_token_id: int = 163603
    pos_emb_type: str = "divided_fixed"
    video_attn_type: str = "spatial_temporal"
    merge_kernel_size: List[int] = field(default_factory=lambda: [2, 2])
    merge_type: str = "sd2_tpool"
    image_token_id: int = 163603
    model_type: str = "moon_vit_3d"


@dataclass
class PatchMergerMLPAdapterConfig(BaseModelConfig):
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """

    normalization: str
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06

    model_type: str = "patch_merger_adapter"
