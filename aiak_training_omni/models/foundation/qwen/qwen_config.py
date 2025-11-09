"""register qwen model with different config"""

from dataclasses import dataclass, fields
from aiak_training_omni.models.common.base_config import BaseModelConfig
from megatron.core.transformer import TransformerConfig
from aiak_training_omni.models.factory import register_model_config
from aiak_training_omni.utils.constants import LanguageModelFamilies


@dataclass
class QwenConfig(BaseModelConfig):
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
    kv_channels: int = None
    num_experts: int = None
    moe_ffn_hidden_size: int = None
    rotary_emb_func: str = "RotaryEmbedding"
    model_spec = None
    model_name: str = "qwen"


@register_model_config(model_family=LanguageModelFamilies.QWEN, model_arch="qwen-7b")
def qwen_7b():
    """qwen 7b"""
    return dict(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        group_query_attention=False,
        num_query_groups=1,
        position_embedding_type="rope",
        add_position_embedding=False,
        rotary_interleaved=False,
        normalization="RMSNorm",
        swiglu=True,
        attention_dropout=0,
        hidden_dropout=0,
        add_bias_linear=False,
        add_qkv_bias=True,
        qk_layernorm=False,
        untie_embeddings_and_output_weights=True,
        kv_channels=None,
        num_experts=None,
        moe_ffn_hidden_size=None,
        rotary_emb_func="RotaryEmbedding",
        model_name="qwen",
    )
