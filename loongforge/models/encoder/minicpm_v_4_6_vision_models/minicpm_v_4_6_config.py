# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V-4.6 vision and merger configs."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

from loongforge.models.common.base_model_config import BaseModelConfig


def gelu_pytorch_tanh(inputs: torch.Tensor) -> torch.Tensor:
    """Match the MiniCPM checkpoint's `gelu_pytorch_tanh` activation."""
    return torch.nn.functional.gelu(inputs, approximate="tanh")


@dataclass(kw_only=True)
class MiniCPMV46VisionConfig(BaseModelConfig):
    """Configuration for the MiniCPM-V-4.6 vision tower."""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    patch_size: int = 14
    image_size: int = 980
    in_channels: int = 3
    kv_channels: int = 72
    num_query_groups: int = 16
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    activation_func: Callable[[torch.Tensor], torch.Tensor] = gelu_pytorch_tanh
    bias_activation_fusion: bool = False
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    normalization: str = "LayerNorm"
    position_embedding_type: str = "none"
    insert_layer_id: int = 6
    window_kernel_size: Tuple[int, int] = (2, 2)
    image_token_id: int = 248056
    downsample_mode: str = "16x"
    mix_used_vision_encoder: bool = True
    mix_used_vision_projector: bool = True

    model_spec: Optional[Tuple[str, str]] = (
        "loongforge.models.encoder.minicpm_v_4_6_vision_models.minicpm_v_4_6_layer_spec",
        "get_minicpm_v_4_6_vision_layer_spec",
    )

    model_type: str = "minicpm_v_4_6_vit"

    @property
    def num_position_embeddings(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side

    @property
    def window_hidden_size(self) -> int:
        return self.hidden_size * self.window_kernel_size[0] * self.window_kernel_size[1]

    @property
    def window_intermediate_size(self) -> int:
        return self.ffn_hidden_size * self.window_kernel_size[0] * self.window_kernel_size[1]


@dataclass(kw_only=True)
class MiniCPMV46MergerConfig(BaseModelConfig):
    """Configuration for the MiniCPM-V-4.6 final visual merger."""

    num_layers: int = 1
    hidden_size: int = 1152
    ffn_hidden_size: int = 4608
    num_attention_heads: int = 1
    kv_channels: int = 1152
    normalization: str = "LayerNorm"
    activation_func: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.gelu
    bias_activation_fusion: bool = False
    add_bias_linear: bool = True
    layernorm_epsilon: float = 1e-6
    merge_kernel_size: Tuple[int, int] = (2, 2)
    merger_times: int = 1
    model_type: str = "minicpm_v_4_6_merger"
