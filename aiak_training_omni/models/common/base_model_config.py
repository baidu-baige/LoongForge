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
    padded_vocab_size: int = None
    fp16_lm_cross_entropy: bool = False
    rotary_seq_len_interpolation_factor: int = None
    rotary_percent: float = 0.1
    rotary_base: int = 10000
    use_rope_scaling: bool = False
    rope_scaling_factor: float = 8.0

    def __post_init__(self):
        PretrainedConfig.__init__(self)
        TransformerConfig.__post_init__(self)


@dataclasses.dataclass
class BaseModelMLAConfig(MLATransformerConfig, PretrainedConfig):
    """Base configuration class for AIAK training LLM models."""

    freeze: bool = False
    model_type: str = None
    model_spec: Optional[List[str]] = None
    # TODO: abstract the repeat part
    padded_vocab_size: int = None
    fp16_lm_cross_entropy: bool = False
    rotary_seq_len_interpolation_factor: int = None
    rotary_percent: float = 0.1
    rotary_base: int = 10000
    use_rope_scaling: bool = False
    rope_scaling_factor: float = 8.0

    def __post_init__(self):
        PretrainedConfig.__init__(self)
        MLATransformerConfig.__post_init__(self)