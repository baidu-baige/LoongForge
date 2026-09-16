# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register qwen model with different config"""

from dataclasses import dataclass, fields
from loongforge.models.common.base_model_config import BaseModelConfig
from loongforge.utils.constants import LanguageModelFamilies


@dataclass
class Qwen2Config(BaseModelConfig):
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
    rotary_base: int = 10000
    rotary_emb_func: str = "RotaryEmbedding"
    model_type = LanguageModelFamilies.QWEN2
    word_embeddings_for_head: str = "lm_head"
