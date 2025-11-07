"""encoder models package."""

from .qwenvl_vision_models.vision_model import VisionModel, VisionModelWithRMSNorm
from .qwenvl_vision_models.adapter import Adapter
from .qwenvl_vision_models.qwen2_vl_config import (
    QwenVisionConfig,
    QwenVisionRMSNormConfig,
    MLPAdapterConfig,
)
from transformers import AutoModel

AutoModel.register(QwenVisionConfig, VisionModel)
AutoModel.register(QwenVisionRMSNormConfig, VisionModelWithRMSNorm)
AutoModel.register(MLPAdapterConfig, Adapter)
