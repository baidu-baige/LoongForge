# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register llava model with different config"""

import torch
from dataclasses import dataclass
from loongforge.models.common.base_model_config import BaseModelConfig


@dataclass
class RiceVisionConfig(BaseModelConfig):
    """configuration for vision model

    The fields need to be consistent with the definitions in args
    """

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    kv_channels: int
    normalization: str
    patch_size: tuple[int] = 14
    image_size: tuple[int] = (1344, 1344)
    swiglu: bool = False
    class_token_len: int = 0
    group_query_attention: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    layernorm_epsilon: float = 1e-05
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    in_channels: int = 3
    num_query_groups: int = None
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    position_embedding_type: str = "none"
    image_token_id: int = 151655
    video_token_id: int = 151656
    mix_used_vision_encoder: bool = False
    mix_used_vision_projector: bool = False

    model_type: str = "rice_vit"
