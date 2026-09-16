# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register qwen model with different config"""

from dataclasses import dataclass
from ..qwen2_vl_vision_models.qwen2_vl_config import Qwen2VisionModelConfig


@dataclass
class Qwen3VisionModelConfig(Qwen2VisionModelConfig):
    """configuration for vision model"""
    
    patch_size: int = 16
    deepstack_visual_indexes: list = None
    num_position_embeddings: int = 2304

    model_type = "qwen3_vit"