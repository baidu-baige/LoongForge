# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""register qwen3.5 model with different config"""

from dataclasses import dataclass

from loongforge.models.common.base_model_config import BaseModelConfig
from loongforge.utils.constants import VisionLanguageModelFamilies


@dataclass
class Qwen35Config(BaseModelConfig):
    """config for qwen3.5 model"""

    # core architecture
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    # grouped query attention
    group_query_attention: bool = False
    num_query_groups: int = 1

    # position embedding / RoPE
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    rotary_percent: float = 0.25
    apply_rope_fusion: bool = False
    rotary_emb_func: str = "RotaryEmbedding"
    rotary_base: int = 10000000

    # normalization and activation
    normalization: str = "RMSNorm"
    swiglu: bool = True

    # dropout
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # bias and layernorm
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    qk_layernorm: bool = False

    # vocabulary and embedding
    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: int = 248320
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = 256

    # mixture of experts (None for dense models)
    num_experts: int = None
    moe_ffn_hidden_size: int = None

    # linear attention
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    layer_types: list[str] = None

    # multimodal RoPE (M-RoPE)
    mrope_section: list[int] = None
    mrope_interleaved: bool = True

    model_type = VisionLanguageModelFamilies.QWEN3_5
