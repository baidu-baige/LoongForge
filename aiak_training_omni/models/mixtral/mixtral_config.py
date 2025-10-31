"""register mixtral model with different config"""

from dataclasses import dataclass

from aiak_training_omni.utils.constants import LanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class MixtralConfig:
    """config for mixtral model"""
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    moe_ffn_hidden_size: int
    num_attention_heads: int
    num_experts: int
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
    vocab_size_in_config_file: int = None
    make_vocab_size_divisible_by: int = 128


@register_model_config(model_family=LanguageModelFamilies.MIXTRAL, model_arch="mixtral-8x7b")
def mixtral_8x7b():
    """mixtral 8x7b"""
    return MixtralConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=14336,
        moe_ffn_hidden_size=14336,
        num_attention_heads=32,
        num_experts=8,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=32000,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.MIXTRAL, model_arch="mixtral-8x22b")
def mixtral_8x22b():
    """mixtral 8x22b"""
    return MixtralConfig(
        num_layers=56,
        hidden_size=6144,
        ffn_hidden_size=16384,
        moe_ffn_hidden_size=16384,
        num_attention_heads=48,
        num_experts=8,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=32000,
        make_vocab_size_divisible_by=128,
    )
