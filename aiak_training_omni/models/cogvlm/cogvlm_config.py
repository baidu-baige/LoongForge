"""cogvlm configuration"""

import torch
from dataclasses import dataclass

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
    layernorm_epsilon: float = 1e-05
    activation_func: torch.nn.Module = torch.nn.functional.gelu
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
    hidden_size: int
    ffn_hidden_size: int
    normalization: str
    layernorm_epsilon: float = 1e-05
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    bias_activation_fusion: bool = True
    activation_func: torch.nn.Module = torch.nn.functional.silu

@dataclass
class LanguageConfig:
    """configuration for cogvlm model
    
    The fields need to be consistent with the definitions in args
    """
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    group_query_attention: bool = False
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    normalization: str = "RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = False
    untie_embeddings_and_output_weights: bool = True
    apply_rope_fusion: bool = True


@register_model_config(model_family=VisionLanguageModelFamilies.COGVLM2, model_arch="cogvlm2-llama3-chinese-chat-19b")
def cogvlm2_llama3_chinese_chat_19b():
    """cogvlm2-llama3-chinese-chat-19B"""
    return LanguageConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        num_query_groups=8,
        group_query_attention=True,
        apply_rope_fusion=False,
    )

def get_vision_config():
    """ get vision config """
    return VisionConfig(
        num_layers=63,
        hidden_size=1792,
        kv_channels=112,
        ffn_hidden_size=15360,
        patch_size=14,
        class_token_len=1,
        num_attention_heads=16,
        num_query_groups=16,
        image_size=(1344, 1344),
        layernorm_epsilon=1e-06,
        normalization="LayerNorm",
        add_bias_linear=True,
        add_qkv_bias=True,
    )


def get_adapeter_config():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=4096,
        ffn_hidden_size=14336,
        normalization="LayerNorm"
    )