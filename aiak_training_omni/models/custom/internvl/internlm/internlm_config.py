"""register internlm model with different config"""

from dataclasses import dataclass

from aiak_training_llm.utils.constants import LanguageModelFamilies
from aiak_training_llm.models.factory import register_model_config


@dataclass
class InternLMConfig:
    """configuration for internlm model

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


@register_model_config(
    model_family=LanguageModelFamilies.INTERNLM2_5, model_arch="internlm2.5-1.8b"
)
def internlm2_5_1_8b():
    """intern2.5 1.8b"""
    return InternLMConfig(
        num_layers=24,
        hidden_size=2048,
        ffn_hidden_size=8192,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
    )


@register_model_config(
    model_family=LanguageModelFamilies.INTERNLM2_5, model_arch="internlm2.5-7b"
)
def internlm2_5_7b():
    """intern2.5 7b"""
    return InternLMConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
    )


@register_model_config(
    model_family=LanguageModelFamilies.INTERNLM2_5, model_arch="internlm2.5-20b"
)
def internlm2_5_20b():
    """intern2.5 20b"""
    return InternLMConfig(
        num_layers=48,
        hidden_size=6144,
        ffn_hidden_size=16384,
        num_attention_heads=48,
        group_query_attention=True,
        num_query_groups=8,
    )
