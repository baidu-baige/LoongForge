"""register llama model with different config"""

from dataclasses import dataclass

from aiak_training_omni.utils.constants import LanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class LlamaConfig:
    """configuration for llama model
    
    The fields need to be consistent with the definitions in args
    """
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
    add_qkv_bias: bool = False
    qk_layernorm: bool = False
    untie_embeddings_and_output_weights: bool = True


@register_model_config(model_family=LanguageModelFamilies.LLAMA, model_arch="llama-7b")
def llama_7b():
    """llama 7b"""
    return LlamaConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA, model_arch="llama-13b")
def llama_13b():
    """llama 13b"""
    return LlamaConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13824,
        num_attention_heads=40,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA, model_arch="llama-30b")
def llama_30b():
    """llama 30b"""
    return LlamaConfig(
        num_layers=60,
        hidden_size=6656,
        ffn_hidden_size=17920,
        num_attention_heads=52,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA, model_arch="llama-65b")
def llama_65b():
    """llama 65b"""
    return LlamaConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=22016,
        num_attention_heads=64,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA2, model_arch="llama2-7b")
def llama2_7b():
    """llama2 7b"""
    return LlamaConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA2, model_arch="llama2-13b")
def llama2_13b():
    """llama2 13b"""
    return LlamaConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13824,
        num_attention_heads=40,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA2, model_arch="llama2-70b")
def llama2_70b():
    """llama2 70b"""
    return LlamaConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=28672,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA3, model_arch="llama3-8b")
def llama3_8b():
    """llama3 8b"""
    return LlamaConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA3, model_arch="llama3-70b")
def llama3_70b():
    """llama3 70b"""
    return LlamaConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=28672,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,    
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA3_1, model_arch="llama3.1-8b")
def llama3_1_8b():
    """llama3.1 8b"""
    return LlamaConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA3_1, model_arch="llama3.1-70b")
def llama3_1_70b():
    """llama3.1 70b"""
    return LlamaConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=28672,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
    )


@register_model_config(model_family=LanguageModelFamilies.LLAMA3_1, model_arch="llama3.1-405b")
def llama3_1_405b():
    """llama3.1 405b"""
    return LlamaConfig(
        num_layers=126,
        hidden_size=16384,
        ffn_hidden_size=53248,
        num_attention_heads=128,
        group_query_attention=True,
        num_query_groups=8,
    )
