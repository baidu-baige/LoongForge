# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register internlm model with different config"""

from dataclasses import dataclass

from loongforge.utils.constants import LanguageModelFamilies
from loongforge.models.factory import register_model_config
from loongforge.models.common.base_model_config import BaseModelConfig


@dataclass
class InternLMConfig(BaseModelConfig):
    """configuration for internlm model
    The fields need to be consistent with the definitions in args
    """

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    max_sequence_length: int = 32768
    vocab_size: int = 92553
    group_query_attention: bool = False
    num_query_groups: int = 1
    max_position_embeddings: int = 32768
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
    rotary_base: int = 1000000
    rotary_emb_func: str = "RotaryEmbedding"
    model_spec = [
        "loongforge.models.foundation.internlm.internlm_layer_spec",
        "get_internlm_layer_with_te_spec",
    ]
    
    model_type = LanguageModelFamilies.INTERNLM2_5
