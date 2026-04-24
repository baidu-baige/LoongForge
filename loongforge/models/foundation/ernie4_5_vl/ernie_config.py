# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ernie-VL configuration"""

import torch
import math
from dataclasses import dataclass
from loongforge.models.common.base_model_config import BaseModelConfig


@dataclass
class ErnieMoeConfig(BaseModelConfig):
    """configuration for ernie model
    The fields need to be consistent with the definitions in args
    """
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    group_query_attention: bool = False
    num_query_groups: int = 4
    position_embedding_type: str = "rope"
    vocab_size_in_config_file: int = 103424
    make_vocab_size_divisible_by: int = 128
    qk_layernorm: bool = True
    kv_channels: int = 128
    add_qkv_bias: bool = False
    num_experts: int = 64
    rotary_base: int = 10000
    mrope_section: tuple[int] = (22, 22, 20)
    gated_linear_unit: bool = False
    bias_activation_fusion: bool = False

    moe_num_shared_experts: int = 2
    rotary_emb_func: str = "RotaryEmbedding"
    max_position_embeddings: int = 131072
    moe_intermediate_size: tuple[int] = (1536, 512)
    moe_router_topk: int = 6
    moe_layer_end_index: tuple[int] = (29, 28)

    dense_layer_index: tuple[int] = (0, )
    moe_layer_start_index: tuple[int] = (1, 1)
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    normalization: str = "Torch_RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0
    hidden_dropout: float = 0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    untie_embeddings_and_output_weights: bool = False
    apply_rope_fusion: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    im_patch_id: int = 100295
    moe_router_pre_softmax: bool = True
    softmax_scale: float = 1 / math.sqrt(128)
    attention_softmax_in_fp32: bool = True
    activation_func: str = "silu"