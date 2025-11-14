"""register baichuan model with different config"""

from dataclasses import dataclass

from aiak_training_omni.utils.constants import LanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class BaichuanConfig:
    """config for baichuan model"""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    group_query_attention: bool = False
    num_query_groups: int = 1
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    use_normhead: bool = False
    normalization: str = "RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = False
    untie_embeddings_and_output_weights: bool = True


@register_model_config(
    model_family=LanguageModelFamilies.BAICHUAN, model_arch="baichuan-7b"
)
def baichuan_7b():
    """baichuan 7b"""
    return BaichuanConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
        position_embedding_type="rope",
    )


@register_model_config(
    model_family=LanguageModelFamilies.BAICHUAN, model_arch="baichuan-13b"
)
def baichuan_13b():
    """baichuan 13b"""
    return BaichuanConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13696,
        num_attention_heads=40,
        position_embedding_type="alibi",
    )


@register_model_config(
    model_family=LanguageModelFamilies.BAICHUAN2, model_arch="baichuan2-7b"
)
def baichuan2_7b():
    """baichuan2 7b"""
    return BaichuanConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
        position_embedding_type="rope",
        use_normhead=True,
    )


@register_model_config(
    model_family=LanguageModelFamilies.BAICHUAN2, model_arch="baichuan2-13b"
)
def baichuan2_13b():
    """baichuan2 13b"""
    return BaichuanConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13696,
        num_attention_heads=40,
        position_embedding_type="alibi",
        use_normhead=True,
    )
