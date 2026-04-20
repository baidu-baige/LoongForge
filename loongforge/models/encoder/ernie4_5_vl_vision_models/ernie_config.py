# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ernie-VL configuration"""

import torch
from dataclasses import dataclass
from typing import List, Optional
from dataclasses import dataclass, field
from megatron.core.activations import quick_gelu
from loongforge.models.common.base_model_config import BaseModelConfig


@dataclass
class ErnieVisionConfig(BaseModelConfig):
    """Configuration for the ERNIE-4.5-VL vision encoder.

    Inherits from BaseModelConfig (→ TransformerConfig) so that the
    TransformerBlock machinery can be used directly.

    TransformerConfig required fields are derived from the ViT-specific
    fields (embed_dim, num_heads, mlp_ratio) in __post_init__.
    """
    # ---- ViT-specific fields ----
    num_layers: int = 32
    embed_dim: int = 1280          # ViT hidden dimension
    in_channels: int = 3
    mlp_ratio: float = 4.0
    patch_size: int = 14
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-6
    temporal_merge_size: int = 2
    use_temporal_conv: bool = True
    resampler_hidden_in: int = 1280
    resampler_hidden_out: int = 2560
    image_token_id: int = 100295
    freeze: bool = True

    # ---- TransformerConfig fields (overwritten in __post_init__) ----
    hidden_size: int = 5120
    ffn_hidden_size: int = 5120
    num_attention_heads: int = 16

    # ViT uses bias in QKV, output projection, and MLP fc1/fc2.
    add_qkv_bias: bool = True
    add_bias_linear: bool = True

    # Standard ViT normalization / attention settings
    normalization: str = "LayerNorm"
    layernorm_epsilon: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    group_query_attention: bool = False
    num_query_groups: int = None

    # Activation: QuickGELU (not SwiGLU)
    activation_func: torch.nn.Module = quick_gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    swiglu: bool = False

    # layer spec — default points to ERNIE's own spec function
    model_spec: Optional[List[str]] = field(default_factory=lambda: [
        "loongforge.models.encoder.ernie4_5_vl_vision_models.ernie_encoder_spec",
        "get_ernie_vl_vision_layer_spec",
    ])

    def __post_init__(self):
        self.activation_func = quick_gelu
        super().__post_init__()


@dataclass
class ErnieAdapterConfig(BaseModelConfig):
    """configuration for adapter model
    """
    in_dim: int = None
    out_dim: int = None
    rms_norm_eps: float = None
