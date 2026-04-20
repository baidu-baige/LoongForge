# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""encoder models package."""

from .base_vision_models.base_vision_model import BaseVisionModel
from .qwen2_vl_vision_models.vision_model import Qwen2VisionModelWithRMSNorm
from .qwen2_vl_vision_models.adapter import Adapter
from .qwen2_vl_vision_models.qwen2_vl_config import (
    Qwen2VisionModelConfig,
    Qwen2VisionRMSNormConfig,
    MLPAdapterConfig,
)
from .qwen3_vl_vision_models.vision_model import Qwen3VisionModel
from .qwen3_vl_vision_models.qwen3_vl_config import Qwen3VisionModelConfig

from .qwen3_5_vision_models.vision_model import Qwen35VisionModel
from .qwen3_5_vision_models.qwen3_5_vision_config import Qwen35VisionConfig

from .llavaov1_5_vision_models.llavaov_1_5_config import RiceVisionConfig
from .llavaov1_5_vision_models.rice_vision_model import RiceViTModel
from transformers import AutoModel

from .internvl_vision_models.intern_vision_model import InternVisionModel
from .internvl_vision_models.adapter import InternAdapter
from .internvl_vision_models.internvl_config import (
    InternVisionConfig,
    InternMLPAdapterConfig,
)

from .ernie4_5_vl_vision_models.ernie_vision_model import ErnieVisionModel
from .ernie4_5_vl_vision_models.ernie_adapter import ErnieAdapter
from .ernie4_5_vl_vision_models.ernie_config import ErnieVisionConfig, ErnieAdapterConfig

from .moon_vision_models.moon_vision_model import MoonVisionModel
from .moon_vision_models.moon_vision_config import MoonVisionModelConfig
from .moon_vision_models.patch_merger_adapter import PatchMergerMLPAdapterConfig
from .moon_vision_models.patch_merger_adapter import PatchMergerMLP


AutoModel.register(Qwen2VisionModelConfig, BaseVisionModel)
AutoModel.register(Qwen2VisionRMSNormConfig, Qwen2VisionModelWithRMSNorm)
AutoModel.register(Qwen3VisionModelConfig, Qwen3VisionModel)
AutoModel.register(Qwen35VisionConfig, Qwen35VisionModel)
AutoModel.register(MLPAdapterConfig, Adapter)
AutoModel.register(RiceVisionConfig, RiceViTModel)
AutoModel.register(InternVisionConfig, InternVisionModel)
AutoModel.register(InternMLPAdapterConfig, InternAdapter)
AutoModel.register(ErnieVisionConfig, ErnieVisionModel)
AutoModel.register(ErnieAdapterConfig, ErnieAdapter)
AutoModel.register(MoonVisionModelConfig, MoonVisionModel)
AutoModel.register(PatchMergerMLPAdapterConfig, PatchMergerMLP)
