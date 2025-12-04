"""encoder models package."""

from .qwen2_vl_vision_models.vision_model import Qwen2VisionModel, Qwen2VisionModelWithRMSNorm
from .qwen2_vl_vision_models.adapter import Adapter
from .qwen2_vl_vision_models.qwen2_vl_config import (
    Qwen2VisionModelConfig,
    Qwen2VisionRMSNormConfig,
    MLPAdapterConfig,
)
from .llavaov1_5_vision_models.llavaov_1_5_config import RiceVisionConfig
from .llavaov1_5_vision_models.rice_vision_model import RiceViTModel
from transformers import AutoModel

from .internvl_vision_models.intern_vision_model import InternVisionModel
from .internvl_vision_models.adapter import InternAdapter
from .internvl_vision_models.internvl_config import (
    InternVisionConfig,
    InternMLPAdapterConfig,
)

AutoModel.register(Qwen2VisionModelConfig, Qwen2VisionModel)
AutoModel.register(Qwen2VisionRMSNormConfig, Qwen2VisionModelWithRMSNorm)
AutoModel.register(MLPAdapterConfig, Adapter)
AutoModel.register(RiceVisionConfig, RiceViTModel)
AutoModel.register(InternVisionConfig, InternVisionModel)
AutoModel.register(InternMLPAdapterConfig, InternAdapter)