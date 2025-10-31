"""register qwen model with different config"""

from dataclasses import dataclass

from aiak_training_omni.utils.constants import LanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config


@dataclass
class QwenConfig:
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


@register_model_config(model_family=LanguageModelFamilies.QWEN, model_arch="qwen-1.8b")
def qwen_1_8b():
    """qwen 1.8b"""
    return QwenConfig(
      num_layers=24,
      hidden_size=2048,
      ffn_hidden_size=5504,
      num_attention_heads=16,
      vocab_size_in_config_file=151936,
      make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN, model_arch="qwen-7b")
def qwen_7b():
    """qwen 7b"""
    return QwenConfig(
      num_layers=32,
      hidden_size=4096,
      ffn_hidden_size=11008,
      num_attention_heads=32,
      vocab_size_in_config_file=151936,
      make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN, model_arch="qwen-14b")
def qwen_14b():
    """qwen 14b"""
    return QwenConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13696,
        num_attention_heads=40,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN, model_arch="qwen-72b")
def qwen_72b():
    """qwen 72b"""
    return QwenConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=24576,
        num_attention_heads=64,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN1_5, model_arch="qwen1.5-0.5b")
def qwen1_5_0_5b() -> QwenConfig:
    """qwen 1.5 0.5b"""
    return QwenConfig(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=2816,
        num_attention_heads=16,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN1_5, model_arch="qwen1.5-1.8b")
def qwen1_5_1_8b() -> QwenConfig:
    """qwen 1.5 1.8b"""
    return QwenConfig(
        num_layers=24,
        hidden_size=2048,
        ffn_hidden_size=5504,
        num_attention_heads=16,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN1_5, model_arch="qwen1.5-4b")
def qwen1_5_4b() -> QwenConfig:
    """qwen 1.5 4b"""
    return QwenConfig(
        num_layers=40,
        hidden_size=2560,
        ffn_hidden_size=6912,
        num_attention_heads=20,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN1_5, model_arch="qwen1.5-7b")
def qwen1_5_7b() -> QwenConfig:
    """qwen 1.5 7b"""
    return QwenConfig(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=11008,
        num_attention_heads=32,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN1_5, model_arch="qwen1.5-14b")
def qwen1_5_14b() -> QwenConfig:
    """qwen 1.5 14b"""
    return QwenConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=13696,
        num_attention_heads=40,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN1_5, model_arch="qwen1.5-32b")
def qwen1_5_32b() -> QwenConfig:
    """qwen 1.5 32b"""
    return QwenConfig(
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=27392,
        num_attention_heads=40,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN1_5, model_arch="qwen1.5-72b")
def qwen1_5_72b() -> QwenConfig:
    """qwen 1.5 72b"""
    return QwenConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=24576,
        num_attention_heads=64,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2, model_arch="qwen2-0.5b")
def qwen2_0_5b() -> QwenConfig:
    """qwen 2 0.5b"""
    return QwenConfig(
        num_layers=24,
        hidden_size=896,
        ffn_hidden_size=4864,
        num_attention_heads=14,
        group_query_attention=True,
        num_query_groups=2,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        untie_embeddings_and_output_weights=False,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2, model_arch="qwen2-1.5b")
def qwen2_1_5b() -> QwenConfig:
    """qwen 2 1.5b"""
    return QwenConfig(
        num_layers=28,
        hidden_size=1536,
        ffn_hidden_size=8960,
        num_attention_heads=12,
        group_query_attention=True,
        num_query_groups=2,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        untie_embeddings_and_output_weights=False,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2, model_arch="qwen2-7b")
def qwen2_7b() -> QwenConfig:
    """qwen 2 7b"""
    return QwenConfig(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        group_query_attention=True,
        num_query_groups=4,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2, model_arch="qwen2-72b")
def qwen2_72b() -> QwenConfig:
    """qwen 2 72b"""
    return QwenConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=29568,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2_5, model_arch="qwen2.5-0.5b")
def qwen2_5_0_5b() -> QwenConfig:
    """qwen2.5 0.5b"""
    return QwenConfig(
        num_layers=24,
        hidden_size=896,
        ffn_hidden_size=4864,
        num_attention_heads=14,
        group_query_attention=True,
        num_query_groups=2,
        untie_embeddings_and_output_weights=False,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2_5, model_arch="qwen2.5-1.5b")
def qwen2_5_1_5b() -> QwenConfig:
    """qwen2.5 1.5b"""
    return QwenConfig(
        num_layers=28,
        hidden_size=1536,
        ffn_hidden_size=8960,
        num_attention_heads=12,
        group_query_attention=True,
        num_query_groups=2,
        untie_embeddings_and_output_weights=False,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2_5, model_arch="qwen2.5-3b")
def qwen2_5_3b() -> QwenConfig:
    """qwen2.5 3b"""
    return QwenConfig(
        num_layers=36,
        hidden_size=2048,
        ffn_hidden_size=11008,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=2,
        untie_embeddings_and_output_weights=False,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2_5, model_arch="qwen2.5-7b")
def qwen2_5_7b() -> QwenConfig:
    """qwen2.5 7b"""
    return QwenConfig(
        num_layers=28,
        hidden_size=3584,
        ffn_hidden_size=18944,
        num_attention_heads=28,
        group_query_attention=True,
        num_query_groups=4,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2_5, model_arch="qwen2.5-14b")
def qwen2_5_14b() -> QwenConfig:
    """qwen2.5 14b"""
    return QwenConfig(
        num_layers=48,
        hidden_size=5120,
        ffn_hidden_size=13824,
        num_attention_heads=40,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2_5, model_arch="qwen2.5-32b")
def qwen2_5_32b() -> QwenConfig:
    """qwen2.5 32b"""
    return QwenConfig(
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=27648,
        num_attention_heads=40,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN2_5, model_arch="qwen2.5-72b")
def qwen2_5_72b() -> QwenConfig:
    """qwen2.5 72b"""
    return QwenConfig(
        num_layers=80,
        hidden_size=8192,
        ffn_hidden_size=29568,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=152064,
        make_vocab_size_divisible_by=128,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-30b-a3b")
def qwen3_30b_a3b() -> QwenConfig:
    """qwen3 30b a3b"""
    return QwenConfig(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=4,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        num_experts=128,
        moe_ffn_hidden_size=768,
    )

@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-coder-30b-a3b")
def qwen3_coder_30b_a3b() -> QwenConfig:
    """qwen3 coder 30b a3b"""
    return QwenConfig(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=5472,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=4,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        num_experts=128,
        moe_ffn_hidden_size=768,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-0.6b")
def qwen3_0_6b() -> QwenConfig:
    """qwen3 0.6b"""
    return QwenConfig(
        num_layers=28,
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-1.7b")
def qwen3_1_7b() -> QwenConfig:
    """qwen3 1.7b"""
    return QwenConfig(
        num_layers=28,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-4b")
def qwen3_4b() -> QwenConfig:
    """qwen3 4b"""
    return QwenConfig(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-8b")
def qwen3_8b() -> QwenConfig:
    """qwen3 8b"""
    return QwenConfig(
        num_layers=36,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-14b")
def qwen3_14b() -> QwenConfig:
    """qwen3 14b"""
    return QwenConfig(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=17408,
        num_attention_heads=40,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
    )


@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-32b")
def qwen3_32b() -> QwenConfig:
    """qwen3 32b"""
    return QwenConfig(
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=25600,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        add_qkv_bias=False,
        kv_channels=128,
    )

@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-235b-a22b")
def qwen3_235b_a22b() -> QwenConfig:
    """qwen3 235b a22b"""
    return QwenConfig(
        num_layers=94,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=4,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        num_experts=128,
        moe_ffn_hidden_size=1536,
    )

@register_model_config(model_family=LanguageModelFamilies.QWEN3, model_arch="qwen3-480b-a35b")
def qwen3_480b_a35b() -> QwenConfig:
    """qwen3 480b a35b"""
    return QwenConfig(
        num_layers=62,
        hidden_size=6144,
        ffn_hidden_size=8192,
        num_attention_heads=96,
        group_query_attention=True,
        num_query_groups=8,
        vocab_size_in_config_file=151936,
        make_vocab_size_divisible_by=128,
        qk_layernorm=True,
        kv_channels=128,
        add_qkv_bias=False,
        num_experts=160,
        moe_ffn_hidden_size=2560,
    )
