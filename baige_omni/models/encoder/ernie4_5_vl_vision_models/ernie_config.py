# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ernie-VL configuration"""

from dataclasses import dataclass
from baige_omni.models.common.base_model_config import BaseModelConfig


@dataclass
class ErnieVisionConfig(BaseModelConfig):
    """configuration for vision model
    """
    attn_implementation: str = "flash"
    num_layers: int = 2
    embed_dim: int = 1280
    hidden_act: str = "quick_gelu"
    hidden_size: int = 5120
    in_channels: int = 3
    in_chans: int = 3
    mlp_ratio: float = 4
    num_heads: int = 16
    patch_size: int = 14
    spatial_merge_size: int = 2
    spatial_patch_size: int = 14
    vit_first_fwd_bsz: int = 128
    attn_sep: bool = True
    rms_norm_eps: float = 1e-5
    temporal_merge_size: int = 2
    use_temporal_conv: bool = True

    resampler_hidden_in: int = 1280
    resampler_hidden_out: int = 2560
    image_token_id: int = 100295
    freeze: bool = True


@dataclass
class ErnieAdapterConfig(BaseModelConfig):
    """configuration for adapter model
    """
    in_dim: int = None
    out_dim: int = None
    rms_norm_eps: float = None