"""encoder models package."""

from .qwenvl_vision_models.vision_model import VisionModel, VisionModelWithRMSNorm
from .qwenvl_vision_models.adapter import Adapter
from .qwenvl_vision_models.qwen2_vl_config import (
    QwenVisionConfig,
    QwenVisionRMSNormConfig,
    MLPAdapterConfig,
)
from .llavaov_vision_models.llavaov_1_5_config import RiceVisionConfig
from .llavaov_vision_models.rice_vision_model import RiceViTModel
from transformers import AutoModel

from .internvl_vision_models.intern_vision_model import InternVisionModel
from .internvl_vision_models.adapter import InternAdapter
from .internvl_vision_models.internvl_config import (
    InternVisionConfig,
    InternMLPAdapterConfig,
)

AutoModel.register(QwenVisionConfig, VisionModel)
AutoModel.register(QwenVisionRMSNormConfig, VisionModelWithRMSNorm)
AutoModel.register(MLPAdapterConfig, Adapter)
AutoModel.register(RiceVisionConfig, RiceViTModel)
AutoModel.register(InternVisionConfig, InternVisionModel)
AutoModel.register(InternMLPAdapterConfig, InternAdapter)