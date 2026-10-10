# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register llama model with different config"""

from dataclasses import dataclass
from loongforge.models.common import BaseModelConfig

@dataclass
class LLaMAConfig(BaseModelConfig):
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

