"""model configuration classes."""

from transformers import PretrainedConfig

from megatron.core.transformer import TransformerConfig
import dataclasses
from dataclasses import fields, asdict
from megatron.training.activations import squared_relu
import torch.nn.functional as F
import torch


@dataclasses.dataclass
class BaseModelConfig(TransformerConfig, PretrainedConfig):
    """Base configuration class for AIAK training LLM models."""

    model_name: str = None

    def __post_init__(self):
        PretrainedConfig.__init__(self)
        TransformerConfig.__post_init__(self)
