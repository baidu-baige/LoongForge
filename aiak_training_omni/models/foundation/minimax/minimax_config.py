"""register minimax model with different config"""

from typing import Optional, Union, List
from dataclasses import dataclass

from aiak_training_omni.utils.constants import LanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class MinimaxConfig:
    """config for minimax model"""
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_experts: int
    moe_ffn_hidden_size: int
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = None
    moe_layer_freq: Optional[Union[int, List[int]]] = None
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
    num_query_groups: int = None
    apply_rope_fusion: bool = True
    group_query_attention: bool = False
