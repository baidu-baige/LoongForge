"""register qwen model with different config"""

import torch
from dataclasses import dataclass

from megatron.training.activations import quick_gelu
from aiak_training_omni.utils.constants import VisionLanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class VisionConfig:
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
    normalization: str
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
    

@dataclass
class AdapterConfig:
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """
    normalization: str
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-06


@dataclass
class Qwen2VLConfig:
    """config for qwen model"""
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    group_query_attention: bool = False
    num_query_groups: int = 1
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    normalization: str = "RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    qk_layernorm: bool = False
    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: int = None
    make_vocab_size_divisible_by: int = 128
    norm_epsilon: float = 1e-06
    rotary_base: int = 1000000


@register_model_config(model_family=VisionLanguageModelFamilies.QWEN2_VL, model_arch="qwen2-vl-2b")
def qwen2_vl_2b():
    """qwen2-vl-2b"""
    return Qwen2VLConfig(
        num_layers=28,
        hidden_size=1536,
        ffn_hidden_size=8960,
        num_attention_heads=12,
        group_query_attention=True,
        num_query_groups=2,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        untie_embeddings_and_output_weights=False,
    )


@register_model_config(model_family=VisionLanguageModelFamilies.QWEN2_5_VL, model_arch="qwen2_5-vl-3b")
def qwen2_5_vl_3b():
    """qwen2-vl-2b"""
    return Qwen2VLConfig(
        num_layers=36,
        hidden_size=2048,
        ffn_hidden_size=11008,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=2,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        untie_embeddings_and_output_weights=False,
    )


def _qwen2_7b():
    """qwen2-vl-7b"""
    return Qwen2VLConfig(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        group_query_attention=True,
        num_query_groups=4,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )
@register_model_config(model_family=VisionLanguageModelFamilies.QWEN2_VL, model_arch="qwen2-vl-7b")
def qwen2_vl_7b():
    """qwen2-vl-7b"""
    return _qwen2_7b()

@register_model_config(model_family=VisionLanguageModelFamilies.QWEN2_5_VL, model_arch="qwen2_5-vl-7b")
def qwen2_5_vl_7b():
    """qwen2-vl-7b"""
    return _qwen2_7b()


@register_model_config(model_family=VisionLanguageModelFamilies.QWEN2_5_VL, model_arch="qwen2_5-vl-32b")
def qwen2_5_vl_32b():
    """qwen2-vl-2b"""
    return Qwen2VLConfig(
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=27648,
        num_attention_heads=40,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


def qwen2_72b():
    """qwen2-vl-72b"""
    return Qwen2VLConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=29568,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )

@register_model_config(model_family=VisionLanguageModelFamilies.QWEN2_VL, model_arch="qwen2-vl-72b")
def qwen2_vl_72b():
    """qwen2-vl-72b"""
    return qwen2_72b()

@register_model_config(model_family=VisionLanguageModelFamilies.QWEN2_5_VL, model_arch="qwen2_5-vl-72b")
def qwen2_5_vl_72b():
    """qwen2-vl-72b"""
    return qwen2_72b()


def get_vision_config(model_family, model_name):
    """ get vision config """
    config = VisionConfig(
        num_layers=32,
        hidden_size=1280,
        kv_channels=80,
        ffn_hidden_size=5120,
        patch_size=14,
        num_attention_heads=16,
        num_query_groups=16,
        image_size=(1344, 1344),
        normalization="LayerNorm",
        add_bias_linear=True,
        add_qkv_bias=True,
    )
    if model_family == VisionLanguageModelFamilies.QWEN2_5_VL:
        config.ffn_hidden_size = 3456 if model_name in ['qwen2_5-vl-72b', 'qwen2_5-vl-32b'] else 3420
        config.swiglu = True
        config.normalization = "RMSNorm"
        config.activation_func = torch.nn.functional.silu
        config.gated_linear_unit = True
    return config


def get_adapeter_config(model_family):
    """ get adapeter config """
    config = AdapterConfig(
        normalization="LayerNorm",
        add_bias_linear=True,
    )
    if model_family == VisionLanguageModelFamilies.QWEN2_5_VL:
        config.normalization = "RMSNorm"
    return config