"""model configuration classes."""

from transformers import PretrainedConfig

from megatron.core.transformer import TransformerConfig
import dataclasses
from typing import Optional, List

@dataclasses.dataclass
class BaseModelConfig(TransformerConfig, PretrainedConfig):
    """Base configuration class for AIAK training LLM models."""

    freeze: bool = False
    model_type: str = None
    model_spec: Optional[List[str]] = None

    def __post_init__(self):
        PretrainedConfig.__init__(self)
        TransformerConfig.__post_init__(self)
