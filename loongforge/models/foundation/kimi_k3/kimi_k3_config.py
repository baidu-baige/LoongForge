# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Kimi K3 language backbone."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from loongforge.models.common.base_model_config import BaseModelMLAConfig


@dataclass
class KimiK3Config(BaseModelMLAConfig):
    """MCore configuration for K3's heterogeneous KDA/MLA transformer."""

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    # K3 attention schedule and dimensions.
    kimi_kda_layers: Tuple[int, ...] = ()
    kimi_linear_num_heads: int = 96
    kimi_linear_head_dim: int = 128
    kimi_linear_conv_kernel_size: int = 4
    kimi_kda_gate_lower_bound: float = -5.0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128

    # Latent MoE.
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_latent_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    moe_layer_freq: Optional[Union[int, List[int]]] = None
    moe_router_topk: int = 1
    moe_router_score_function: str = "sigmoid"
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_pre_softmax: bool = True
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0.0
    moe_grouped_gemm: bool = True
    moe_shared_expert_overlap: bool = False
    moe_token_dispatcher_type: str = "alltoall"

    # AttnRes.
    kimi_attn_res_block_size: int = 12
    kimi_output_layer_index: Optional[int] = None

    position_embedding_type: str = "none"
    # K3 uses NoPE; MCore requires this compatibility flag.
    add_position_embedding: bool = True
    normalization: str = "RMSNorm"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    gated_linear_unit: bool = True
    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: int = 0
    make_vocab_size_divisible_by: int = 128
    variable_seq_lengths: bool = True

    model_type: str = "kimi_k3"

    def __post_init__(self, **kwargs):
        # Hydra can supply these from ${oc.env:...}, which yields strings.
        for field_name in ("num_experts", "moe_router_topk"):
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, int(value))
        # MCore reads the canonical expert count from num_moe_experts while
        # LoongForge model YAMLs use num_experts.
        if self.num_moe_experts is None:
            self.num_moe_experts = self.num_experts or None
        if self.moe_ffn_hidden_size == 0 and self.num_moe_experts is not None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size
        # K3's gated activation is SiTU, not a stock GLU. MCore only builds
        # MLPSubmodules.activation_func (SiTUAndMul) when this flag is set;
        # otherwise every MLP and expert silently falls back to
        # config.activation_func and computes GEGLU instead.
        self.use_te_activation_func = True
        super().__post_init__(**kwargs)
        self.kimi_kda_layers = tuple(self.kimi_kda_layers)
        if self.kimi_output_layer_index is None:
            self.kimi_output_layer_index = self.num_layers - 1
        if self.hidden_dropout != 0.0:
            raise ValueError("Kimi K3 requires hidden_dropout=0")
