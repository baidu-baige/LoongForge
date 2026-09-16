# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Vision towers and the shared ViT block."""

from .base_vision_model import BaseVisionModel
from .qwen2_vl.vision_model import Qwen2VisionModelWithRMSNorm
from .qwen2_vl.adapter import Adapter
from .qwen2_vl.config import (
    Qwen2VisionModelConfig,
    Qwen2VisionRMSNormConfig,
    MLPAdapterConfig,
)
from .qwen3_vl.vision_model import Qwen3VisionModel
from .qwen3_vl.config import Qwen3VisionModelConfig

from .qwen3_5_vl.vision_model import Qwen35VisionModel
from .qwen3_5_vl.config import Qwen35VisionConfig

from .llava_ov_1_5.config import RiceVisionConfig
from .llava_ov_1_5.vision_model import RiceViTModel
from transformers import AutoModel

from .internvl.vision_model import InternVisionModel
from .internvl.adapter import InternAdapter
from .internvl.config import (
    InternVisionConfig,
    InternMLPAdapterConfig,
)

from .ernie4_5_vl.vision_model import ErnieVisionModel
from .ernie4_5_vl.adapter import ErnieAdapter
from .ernie4_5_vl.config import ErnieVisionConfig, ErnieAdapterConfig

from .moon.vision_model import MoonVisionModel
from .moon.config import (
    KimiK3PatchMergerConfig,
    MoonVisionModelConfig,
)
from .moon.patch_merger_adapter import PatchMergerMLPAdapterConfig
from .moon.patch_merger_adapter import PatchMergerMLP
from .moon.patch_merger_adapter import KimiK3PatchMerger
from .minicpm_v_4_6 import (
    MiniCPMV46Merger,
    MiniCPMV46MergerConfig,
    MiniCPMV46VisionConfig,
    MiniCPMV46VisionModel,
)


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
AutoModel.register(KimiK3PatchMergerConfig, KimiK3PatchMerger)
AutoModel.register(MiniCPMV46VisionConfig, MiniCPMV46VisionModel)
AutoModel.register(MiniCPMV46MergerConfig, MiniCPMV46Merger)
