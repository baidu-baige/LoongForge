"""register qwen model with different config"""

import torch
from dataclasses import dataclass

from megatron.training.activations import quick_gelu
from ...common.base_config import BaseModelConfig


@dataclass
class QwenVisionConfig(BaseModelConfig):
    """configuration for vision model
    
    The fields need to be consistent with the definitions in args
    """
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    patch_size: tuple[int]
    image_size: tuple[int]
    ffn_hidden_size: int  
    kv_channels: int
    normalization: str = "LayerNorm"
    swiglu: bool = False
    class_token_len: int = 0
    group_query_attention: bool = False
    attention_dropout: float = 0
    hidden_dropout: float = 0
    layernorm_epsilon: float = 1e-06
    activation_func: torch.nn.Module = quick_gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    in_channels: int = 3
    num_query_groups: int = None
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    position_embedding_type: str = "none"
    spatial_merge_size: int = 2
    image_token_id: int = 151655
    video_token_id: int = 151656

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class QwenVisionRMSNormConfig(QwenVisionConfig):
    """configuration for vision model using RMSNorm"""
    normalization: str = "RMSNorm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class MLPAdapterConfig(BaseModelConfig):
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """
    normalization: str
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06

    def __init__(self, **kwargs):
        super().__init__(**kwargs)