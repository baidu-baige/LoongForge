# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Lora model config"""

from ..base_model_config import BasePeftModelConfig
from dataclasses import dataclass, field, fields
from typing import List, Literal, Optional
import torch

@dataclass
class LoraConfig(BasePeftModelConfig):
    """Configuration for LoRA"""
    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None

@dataclass
class VLMLoraConfig(LoraConfig):
    """Configuration for VLM LoRA"""
    apply_to_foundation: bool = False
    apply_to_image_projector: bool = False
    apply_to_image_encoder: bool = False
    apply_to_video_projector: bool = False
    apply_to_video_encoder: bool = False