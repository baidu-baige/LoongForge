# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register qwen model with different config"""

import torch
from dataclasses import dataclass, fields

from megatron.core.activations import quick_gelu
from ...common.base_model_config import BaseModelConfig


@dataclass
class Qwen2VisionModelConfig(BaseModelConfig):
    """configuration for vision model"""

    # ------- Fields without default values, write first -------

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    patch_size: int = 14
    image_size: tuple[int] = (1344, 1344)
    kv_channels: int = 80

    # ------- Fields with default values, write later -------
    normalization: str = "LayerNorm"
    swiglu: bool = False
    class_token_len: int = 0
    group_query_attention: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    activation_func: torch.nn.Module = quick_gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    in_channels: int = 3
    num_query_groups: int = None
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    position_embedding_type: str = "none"
    spatial_merge_size: int = 2
    image_token_id: int = 151655
    video_token_id: int = 151656
    mix_used_vision_encoder: bool = True
    mix_used_vision_projector: bool = True

    model_type = "qwen2_vit"


@dataclass
class Qwen2VisionRMSNormConfig(Qwen2VisionModelConfig):
    """configuration for vision model using RMSNorm"""

    normalization: str = "RMSNorm"

    model_type = "qwen2_5_vit_rmsnorm"


@dataclass
class MLPAdapterConfig(BaseModelConfig):
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """

    normalization: str
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06

    model_type = "qwen2_5_vl_adapter"
