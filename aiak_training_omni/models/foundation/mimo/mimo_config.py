"""register mimo model with different config"""

from dataclasses import dataclass

from aiak_training_llm.utils.constants import LanguageModelFamilies
from aiak_training_llm.models.factory import register_model_config


@dataclass
class MimoConfig:
    """config for mimo model"""
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
    kv_channels: int = None
    num_experts: int = None
    moe_ffn_hidden_size: int = None
    mtp_num_layers: int = 0


@register_model_config(model_family=LanguageModelFamilies.MIMO, model_arch="mimo-7b")
def mimo_7b() -> MimoConfig:
    """mimo 7b"""
    return MimoConfig(
        num_layers=36,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151680,
        make_vocab_size_divisible_by=128,
    )

