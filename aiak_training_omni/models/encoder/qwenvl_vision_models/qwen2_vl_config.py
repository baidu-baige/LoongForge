"""register qwen model with different config"""

import torch
from dataclasses import dataclass, fields

from megatron.training.activations import quick_gelu
from ...common.base_config import BaseModelConfig
from megatron.core.transformer import TransformerConfig
from aiak_training_omni.models.factory import register_model_config
from aiak_training_omni.utils.constants import VisionLanguageModelFamilies


@dataclass
class QwenVisionConfig(BaseModelConfig):
    """configuration for vision model"""

    # ------- 无默认值字段 先写 -------

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    patch_size: int = 14
    image_size: tuple[int] = (1344, 1344)
    kv_channels: int = 80

    # ------- 有默认值字段 后写 -------
    normalization: str = "LayerNorm"
    swiglu: bool = False
    class_token_len: int = 0
    group_query_attention: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
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
    model_spec = None
    model_name: str = "qwen_vit"


@dataclass
class QwenVisionRMSNormConfig(QwenVisionConfig):
    """configuration for vision model using RMSNorm"""

    normalization: str = "RMSNorm"


@dataclass
class MLPAdapterConfig(BaseModelConfig):
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """

    normalization: str
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06
    model_spec = None
    model_name: str = "qwen_adapter"


@register_model_config(
    model_family=VisionLanguageModelFamilies.QWEN2_5_VL, model_arch="qwen_vit"
)
def get_vision_config():
    """get vision config"""
    return dict(
        num_layers=32,
        hidden_size=1280,
        kv_channels=80,
        ffn_hidden_size=5120,
        patch_size=14,
        num_attention_heads=16,
        num_query_groups=16,
        image_size=(1344, 1344),
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        add_bias_linear=True,
        add_qkv_bias=True,
        swiglu=True,
        gated_linear_unit=True,
    )


@register_model_config(
    model_family=VisionLanguageModelFamilies.QWEN2_5_VL, model_arch="qwen_adpater"
)
def get_adapter():
    """get vision config"""
    return dict(normalization="RMSNorm", add_bias_linear=True)
