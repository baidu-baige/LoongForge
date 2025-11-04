"""register deepseek model with different config"""

from typing import Optional, Union, List
from dataclasses import dataclass

from aiak_training_omni.utils.constants import LanguageModelFamilies
from aiak_training_omni.models.factory import register_model_config
from megatron.training.arguments import moe_freq_type


@dataclass
class DeepseekConfig:
    """config for deepseek model"""
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_experts: int
    moe_ffn_hidden_size: int
    moe_shared_expert_intermediate_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int
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
    multi_latent_attention: bool = True


@register_model_config(model_family=LanguageModelFamilies.DEEPSEEK, model_arch="deepseek-v2-lite")
def deepseek_v2_lite():
    """deepseek-v2-lite"""
    return DeepseekConfig(
        num_layers=27,
        hidden_size=2048,
        ffn_hidden_size=10944,
        num_attention_heads=16,
        num_query_groups=16,
        num_experts=64,
        moe_ffn_hidden_size=1408,
        moe_shared_expert_intermediate_size=2816,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        moe_layer_freq=moe_freq_type("[0]*1+[1]*26"),
        qk_layernorm=True,
        vocab_size_in_config_file=102400
    )


@register_model_config(model_family=LanguageModelFamilies.DEEPSEEK, model_arch="deepseek-v2")
def deepseek_v2():
    """deepseek-v2"""
    return DeepseekConfig(
        num_layers=60,
        hidden_size=5120,
        ffn_hidden_size=12288,
        num_attention_heads=128,
        num_query_groups=128,
        num_experts=160,
        moe_ffn_hidden_size=1536,
        moe_shared_expert_intermediate_size=3072,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        moe_layer_freq=moe_freq_type("[0]*1+[1]*59"),
        qk_layernorm=True,
        vocab_size_in_config_file=102400
    )


@register_model_config(model_family=LanguageModelFamilies.DEEPSEEK, model_arch="deepseek-v3")
def deepseek_v3():
    """deepseek-v3"""
    return DeepseekConfig(
        num_layers=61,
        hidden_size=7168,
        ffn_hidden_size=18432,
        num_attention_heads=128,
        num_query_groups=128,
        num_experts=256,
        moe_ffn_hidden_size=2048,
        moe_shared_expert_intermediate_size=2048,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        moe_layer_freq=moe_freq_type("[0]*3+[1]*58"),
        qk_layernorm=True,
        vocab_size_in_config_file=129280
    )
