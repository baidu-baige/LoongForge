"""Omni Model Config"""
from aiak_training_omni.models.common.base_config import BaseModelConfig
from dataclasses import dataclass


@dataclass
class OmniModelConfig:
    """config for omni model"""
    image_encoder: BaseModelConfig = None
    image_projector: BaseModelConfig = None
    audio_encoder: BaseModelConfig = None
    audio_projector: BaseModelConfig = None
    video_encoder: BaseModelConfig = None
    video_projector: BaseModelConfig = None
    foundation: BaseModelConfig = None
    model_name: str = None
