# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""foundation models"""

from transformers import AutoModel

from .qwen2 import Qwen2Model, Qwen2Config
from .qwen3 import Qwen3Model, Qwen3Config
from .qwen3_5 import Qwen35Model, Qwen35Config
from .qwen3_next import Qwen3NextModel, Qwen3NextConfig
from .deepseek import DeepseekConfig, DeepseekModelWithMTP
from .mimo import MimoConfig, MimoModelWithMTP
from .minimax import MinimaxConfig, MinimaxModelWithMTP
from .llama import LLaMAConfig, LLaMAModel
from .internlm import InternLMModel, InternLMConfig

from .ernie4_5_vl import ErnieMoeModel, ErnieMoeConfig
from .glm import GlmConfig, GlmModelWithMTP
# The config name should not be the same as the huggingface config name
# or we can use exist_ok flag?
AutoModel.register(Qwen2Config, Qwen2Model, exist_ok=True)  # overwrite existing Qwen2Config
AutoModel.register(Qwen3Config, Qwen3Model, exist_ok=True)  # overwrite existing Qwen3Config
AutoModel.register(Qwen3NextConfig, Qwen3NextModel, exist_ok=True)
AutoModel.register(Qwen35Config, Qwen35Model, exist_ok=True)
AutoModel.register(GlmConfig, GlmModelWithMTP, exist_ok=True)
AutoModel.register(InternLMConfig, InternLMModel)
AutoModel.register(DeepseekConfig, DeepseekModelWithMTP)
AutoModel.register(MimoConfig, MimoModelWithMTP)
AutoModel.register(MinimaxConfig, MinimaxModelWithMTP)
AutoModel.register(LLaMAConfig, LLaMAModel)
AutoModel.register(ErnieMoeConfig, ErnieMoeModel)
