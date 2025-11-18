"""model configuration classes."""

from transformers import PretrainedConfig

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig
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


@dataclasses.dataclass
class BaseModelMLAConfig(MLATransformerConfig, PretrainedConfig):
    """Base configuration class for AIAK training LLM models with multi-latent attention."""

    freeze: bool = False
    model_type: str = None
    model_spec: Optional[List[str]] = None

    def __post_init__(self):
        PretrainedConfig.__init__(self)
        MLATransformerConfig.__post_init__(self)
