# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Omni Model Config"""

from dataclasses import dataclass
from loongforge.utils.constants import VisionLanguageModelFamilies
from loongforge.models.common.base_model_config import BaseModelConfig, BasePeftModelConfig


@dataclass
class VLMModelConfig:
    """config for omni model"""

    image_encoder: BaseModelConfig = None
    image_projector: BaseModelConfig = None
    video_encoder: BaseModelConfig = None
    video_projector: BaseModelConfig = None
    foundation: BaseModelConfig = None

    peft_class: BasePeftModelConfig = None
    model_type: str = None

    def __init__(
        self,
        image_encoder=None,
        image_projector=None,
        video_encoder=None,
        video_projector=None,
        foundation=None,
        peft_class=None,
        model_type=VisionLanguageModelFamilies.VLM,
        **kwargs,
    ):
        self.image_encoder = image_encoder
        self.image_projector = image_projector
        self.video_encoder = video_encoder
        self.video_projector = video_projector
        self.foundation = foundation
        self.peft_class = peft_class
        self.model_type = model_type
        for k, v in kwargs.items():
            setattr(self, k, v)
