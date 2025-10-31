"""InternVL configuration"""

import torch
from dataclasses import dataclass

from aiak_training_omni.utils.constants import VisionLanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config
from aiak_training_omni.models.qwen.qwen_config import qwen2_5_72b, qwen2_5_32b, qwen2_5_7b, qwen2_5_14b, \
    qwen3_0_6b, qwen3_1_7b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3_32b, qwen3_30b_a3b, qwen3_235b_a22b
from aiak_training_omni.models.internlm.internlm_config import internlm2_5_7b, internlm2_5_20b

@dataclass
class InternVisionConfig:
    """configuration for intern vision model
    
    The fields need to be consistent with the definitions in args
    """
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    patch_size: int
    image_size: int
    kv_channels: int
    normalization: str
    swiglu: bool = False
    group_query_attention: bool = False
    attention_dropout: float = 0.
    hidden_dropout: float = 0.
    layernorm_epsilon: float = 1e-06
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
    vision_type: str = "vit_300m"
    original_num_attention_heads: int = None


@dataclass
class AdapterConfig:
    """configuration for adapter model
    
    The fields need to be consistent with the definitions in args
    """
    hidden_size: int
    ffn_hidden_size: int
    normalization: str
    layernorm_epsilon: float = 1e-05
    add_bias_linear: bool = True
    gated_linear_unit: bool = False
    bias_activation_fusion: bool = True
    activation_func: torch.nn.Module = torch.nn.functional.gelu


def get_vision_config_for_300m():
    """ get vision config OpenGVLab/InternViT-300M-448px-V2_5 """
    return InternVisionConfig(
        num_layers=24,
        hidden_size=1024,
        kv_channels=64,
        ffn_hidden_size=4096,
        patch_size=14,
        num_attention_heads=16,
        num_query_groups=16,
        image_size=448,
        layernorm_epsilon=1e-06,
        normalization="LayerNorm",
        add_bias_linear=True,
        add_qkv_bias=True,
        qk_layernorm=False,
        apply_rope_fusion=False)


def get_vision_config_for_6b():
    """ get vision config OpenGVLab/InternViT-6B-448px-V2_5 """
    return InternVisionConfig(
        num_layers=45,
        hidden_size=3200,
        kv_channels=128,
        ffn_hidden_size=12800,
        patch_size=14,
        num_attention_heads=32,  # 25 pad to 32
        num_query_groups=32,  # 25 pad to 32
        original_num_attention_heads=25,
        image_size=448,
        layernorm_epsilon=1e-06,
        normalization="RMSNorm",
        add_bias_linear=True,
        add_qkv_bias=False,
        qk_layernorm=True,
        apply_rope_fusion=False,
        initializer_factor=0.1,
        vision_type="vit_6b")

get_vision_config = {
    "internvl2.5-8b": get_vision_config_for_300m,
    "internvl2.5-26b": get_vision_config_for_6b,
    "internvl2.5-38b": get_vision_config_for_6b,
    "internvl2.5-78b": get_vision_config_for_6b,
    "internvl3-8b": get_vision_config_for_300m,
    "internvl3-14b": get_vision_config_for_300m,
    "internvl3-38b": get_vision_config_for_6b,
    "internvl3-78b": get_vision_config_for_6b,
    "internvl3.5-1b": get_vision_config_for_300m,
    "internvl3.5-2b": get_vision_config_for_300m,
    "internvl3.5-4b": get_vision_config_for_300m,
    "internvl3.5-8b": get_vision_config_for_300m,
    "internvl3.5-14b": get_vision_config_for_300m,
    "internvl3.5-38b": get_vision_config_for_6b,
    "internvl3.5-30b-a3b": get_vision_config_for_300m,
    "internvl3.5-241b-a28b": get_vision_config_for_6b,
}

def get_adapeter_config_for_internvl2_5_8b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=4096,  # same as llm hidden_size
        ffn_hidden_size=4096,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl2_5_26b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=6144,  # same as llm hidden_size
        ffn_hidden_size=6144,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl2_5_38b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=5120,  # same as llm hidden_size
        ffn_hidden_size=5120,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl2_5_78b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=8192,  # same as llm hidden_size
        ffn_hidden_size=8192,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_8b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=3584,  # same as llm hidden_size
        ffn_hidden_size=3584,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_14b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=5120,  # same as llm hidden_size
        ffn_hidden_size=5120,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_38b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=5120,  # same as llm hidden_size
        ffn_hidden_size=5120,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_78b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=8192,  # same as llm hidden_size
        ffn_hidden_size=8192,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_5_1b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=1024,  # same as llm hidden_size
        ffn_hidden_size=1024,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_5_2b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=2048,  # same as llm hidden_size
        ffn_hidden_size=2048,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_5_4b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=2560,  # same as llm hidden_size
        ffn_hidden_size=2560,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_5_8b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=4096,  # same as llm hidden_size
        ffn_hidden_size=4096,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_5_14b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=5120,  # same as llm hidden_size
        ffn_hidden_size=5120,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_5_38b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=5120,  # same as llm hidden_size
        ffn_hidden_size=5120,  # same as llm hidden_size
        normalization="LayerNorm")
def get_adapeter_config_for_internvl3_5_30b_a3b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=2048,  # same as llm hidden_size
        ffn_hidden_size=2048,  # same as llm hidden_size
        normalization="LayerNorm")

def get_adapeter_config_for_internvl3_5_241b_a28b():
    """ get adapeter config """
    return AdapterConfig(
        hidden_size=4096,  # same as llm hidden_size
        ffn_hidden_size=4096,  # same as llm hidden_size
        normalization="LayerNorm")

get_adapeter_config = {
    "internvl2.5-8b": get_adapeter_config_for_internvl2_5_8b,
    "internvl2.5-26b": get_adapeter_config_for_internvl2_5_26b,
    "internvl2.5-38b": get_adapeter_config_for_internvl2_5_38b,
    "internvl2.5-78b": get_adapeter_config_for_internvl2_5_78b,
    "internvl3-8b": get_adapeter_config_for_internvl3_8b,
    "internvl3-14b": get_adapeter_config_for_internvl3_14b,
    "internvl3-38b": get_adapeter_config_for_internvl3_38b,
    "internvl3-78b": get_adapeter_config_for_internvl3_78b,
    "internvl3.5-1b": get_adapeter_config_for_internvl3_5_1b,
    "internvl3.5-2b": get_adapeter_config_for_internvl3_5_2b,
    "internvl3.5-4b": get_adapeter_config_for_internvl3_5_4b,
    "internvl3.5-8b": get_adapeter_config_for_internvl3_5_8b,
    "internvl3.5-14b": get_adapeter_config_for_internvl3_5_14b,
    "internvl3.5-38b": get_adapeter_config_for_internvl3_5_38b,
    "internvl3.5-30b-a3b": get_adapeter_config_for_internvl3_5_30b_a3b,
    "internvl3.5-241b-a28b": get_adapeter_config_for_internvl3_5_241b_a28b,
}

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl2.5-8b")
def internvl2_5_8b():
    """ internvl2.5-8b """
    return internlm2_5_7b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl2.5-26b")
def internvl2_5_26b():
    """ internvl2.5-26b """
    return internlm2_5_20b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl2.5-38b")
def internvl2_5_38b():
    """ internvl2.5-38b """
    config = qwen2_5_32b()
    config.vocab_size_in_config_file = 151674
    return config

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl2.5-78b")
def internvl2_5_78b():
    """ internvl2.5-78b """
    config = qwen2_5_72b()
    config.vocab_size_in_config_file = 151674
    return config

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3-8b")
def internvl3_8b():
    """ internvl3-8b """
    config = qwen2_5_7b()
    config.vocab_size_in_config_file = 151674
    return config

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3-14b")
def internvl3_14b():
    """ internvl3-14b """
    config = qwen2_5_14b()
    config.vocab_size_in_config_file = 151674
    return config

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3-38b")
def internvl3_38b():
    """ internvl3-38b """
    config = qwen2_5_32b()
    config.vocab_size_in_config_file = 151674
    return config

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3-78b")
def internvl3_78b():
    """ internvl3-78b """
    config = qwen2_5_72b()
    config.vocab_size_in_config_file = 151674
    return config

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-1b")
def internvl3_5_1b():
    """ internvl3.5-1b """
    return qwen3_0_6b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-2b")
def internvl3_5_2b():
    """ internvl3.5-2b """
    return qwen3_1_7b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-4b")
def internvl3_5_4b():
    """ internvl3.5-4b """
    return qwen3_4b()
@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-8b")
def internvl3_5_8b():
    """ internvl3.5-8b """
    return qwen3_8b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-14b")
def internvl3_5_14b():
    """ internvl3.5-14b """
    return qwen3_14b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-38b")
def internvl3_5_38b():
    """ internvl3.5-8b """
    return qwen3_32b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-30b-a3b")
def internvl3_5_30b_a3b():
    """ internvl3.5-30b-a3b """
    return qwen3_30b_a3b()

@register_model_config(model_family=VisionLanguageModelFamilies.INTERN_VL, model_arch="internvl3.5-241b-a28b")
def internvl3_5_241b_a28b():
    """ internvl3.5-241b-a28b """
    return qwen3_235b_a22b()