# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""InternVL configuration"""

import torch
from dataclasses import dataclass, field
from ...common.base_model_config import BaseModelConfig
from loongforge.utils import get_tokenizer
from loongforge.data.multimodal.internvl.internvl_constants import IMG_CONTEXT_TOKEN


def generate_id() -> int:
    """generate image_token_id automatically"""
    tokenizer = get_tokenizer().tokenizer
    image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return image_token_id

@dataclass
class InternVisionConfig(BaseModelConfig):
    """configuration for intern vision model
    
    The fields need to be consistent with the definitions in args
    """

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    kv_channels: int
    normalization: str
    swiglu: bool = False
    group_query_attention: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layernorm_epsilon: float = 1e-06
    patch_size: int = 14
    image_size: int = 448
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    bias_activation_fusion: bool = False
    gated_linear_unit: bool = False
    in_channels: int = 3
    num_query_groups: int = None
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = False
    apply_rope_fusion: bool = False
    position_embedding_type: str = "none"
    use_return_dict: bool = True
    downsample_ratio: float = 0.5
    initializer_factor: float = 1.0
    select_layer: int = -1
    ps_version: str = "v2"
    # "drop_path_rate": 0.1,  # TODO
    original_num_attention_heads: int = None
    original_num_query_groups: int = None
    model_spec = None
    image_token_id: int = field(default_factory=generate_id)
    model_type: str = "intern_vit"


@dataclass
class InternMLPAdapterConfig(BaseModelConfig):
    """configuration for adapter model
    The fields need to be consistent with the definitions in args
    """

    hidden_size: int
    ffn_hidden_size: int
    normalization: str
    downsample_ratio: float = 0.5
    layernorm_epsilon: float = 1e-05
    add_bias_linear: bool = True
    gated_linear_unit: bool = False
    bias_activation_fusion: bool = True
    activation_func: torch.nn.Module = torch.nn.functional.gelu
    model_spec = None
    
    model_type: str = "intern_adapter"
