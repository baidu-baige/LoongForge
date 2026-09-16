"""register glm model with different config"""

from typing import Optional, Union, List
from dataclasses import dataclass
from loongforge.models.common.base_model_config import BaseModelMLAConfig


@dataclass
class GlmConfig(BaseModelMLAConfig):
    """config for GLM model"""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_experts: int = None
    moe_ffn_hidden_size: int = None
    moe_shared_expert_intermediate_size: int = None
    q_lora_rank: int = None
    kv_lora_rank: int = None
    qk_head_dim: int = None
    qk_pos_emb_head_dim: int = None
    v_head_dim: int = None
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
    make_vocab_size_divisible_by: int = 128
    # MTP hyperparameters
    mtp_num_layers: int = 0
    mtp_loss_coef: float = 0.1
