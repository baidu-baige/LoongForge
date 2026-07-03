# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 model config."""

from typing import Optional, Union, List
from dataclasses import dataclass

from loongforge.models.common.base_model_config import BaseModelMLAConfig
from loongforge.utils.constants import LanguageModelFamilies


@dataclass
class DeepseekV4Config(BaseModelMLAConfig):
    """Configuration for DeepSeek-V4 model (MLA + MoE + Hyper-Connections + CSA/HCA).

    Fields beyond BaseModelMLAConfig defaults:
    hc_mult, hc_sinkhorn_iters, hc_eps, swiglu_limit,
    o_groups, o_lora_rank, head_dim, qk_rope_head_dim,
    csa_compress_ratios, csa_compress_rotary_base, csa_window_size,
    index_head_dim, index_n_heads, index_topk,
    moe_n_hash_layers, moe_router_topk_scaling_factor,
    moe_router_score_function, moe_router_enable_expert_bias,
    norm_topk_prob, actual_vocab_size
    """

    # ── Required fields (NO default values, filled by YAML) ──────────────────
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    # ── MQA / MLA ──────────────────────────────────────────────────────────────
    group_query_attention: bool = True
    num_query_groups: int = 1
    q_lora_rank: int = None
    head_dim: int = None
    qk_rope_head_dim: int = None
    multi_latent_attention: bool = True

    # ── MoE ────────────────────────────────────────────────────────────────────
    num_experts: int = None
    moe_ffn_hidden_size: int = None
    moe_shared_expert_intermediate_size: int = None
    moe_layer_freq: Optional[Union[int, List[int]]] = None
    moe_n_hash_layers: int = 0
    actual_vocab_size: Optional[int] = None
    moe_router_topk_scaling_factor: Optional[float] = None
    moe_router_score_function: str = "sqrtsoftplus"
    moe_router_enable_expert_bias: bool = True
    norm_topk_prob: bool = True

    # ── Hyper-Connections ──────────────────────────────────────────────────────
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def __post_init__(self):
        if self.num_moe_experts is None:
            self.num_moe_experts = self.num_experts
        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size

        super().__post_init__()

        assert self.num_moe_experts and self.num_moe_experts > 0, "DeepSeek-V4 requires MoE."
        assert self.multi_latent_attention, "DeepSeek-V4 requires multi_latent_attention."
        assert self.csa_compress_ratios is not None, "csa_compress_ratios must be set."
        mtp_layers = self.mtp_num_layers or 0
        expected_len = self.num_layers + mtp_layers
        assert len(self.csa_compress_ratios) == expected_len, (
            f"csa_compress_ratios length ({len(self.csa_compress_ratios)}) must equal "
            f"num_layers + mtp_num_layers ({self.num_layers} + {mtp_layers} = {expected_len})."
        )
        assert all(ratio in [0, 4, 128] for ratio in self.csa_compress_ratios), (
            "csa_compress_ratios must contain only 0, 4, or 128."
        )
        assert not getattr(self, "qk_clip", False), (
            "QK clipping is not supported with DeepSeek-V4 hybrid attention."
        )
        assert not getattr(self, "mla_down_proj_fusion", False), (
            "MLA down projection fusion must be disabled for DeepSeek-V4 hybrid attention."
        )

        # Map V4-specific HC fields to Megatron TransformerConfig fields
        self.enable_hyper_connections = True
        self.num_residual_streams = self.hc_mult
        self.mhc_sinkhorn_iterations = self.hc_sinkhorn_iters
        self.mhc_init_gating_factor = self.hc_eps

        # Map V4-specific DSA indexer fields to Megatron TransformerConfig fields
        self.dsa_indexer_head_dim = self.index_head_dim
        self.dsa_indexer_n_heads = self.index_n_heads
        self.dsa_indexer_topk = self.index_topk

        # DeepSeek-V4 always uses qk_layernorm
        self.qk_layernorm = True

        # DSv4 hybrid derives MLA latent dimensions from the HF-style head dimensions.
        self.qk_head_dim = self.v_head_dim - self.qk_pos_emb_head_dim
        self.kv_lora_rank = self.qk_head_dim
        self.hetereogenous_dist_checkpoint = True

        # Map swiglu_limit to activation_func_clamp_value for shared expert MLP
        self.activation_func_clamp_value = self.swiglu_limit

        if self.moe_n_hash_layers > 0:
            assert self.actual_vocab_size is not None, (
                "actual_vocab_size must be set when moe_n_hash_layers > 0."
            )

    # ── CSA / HCA (Compressed Sparse Attention) ──────────────────────────────
    csa_window_size: int = 128
    csa_compress_ratios: Optional[List[int]] = None
    csa_compress_rotary_base: float = 160000.0
    csa_dense_mode: bool = False

    # ── Lightning Indexer ──────────────────────────────────────────────────────
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512

    # ── Grouped Output Projection ──────────────────────────────────────────────
    o_groups: int = 8
    o_lora_rank: int = 1024

    # ── SwiGLU Clamping ──────────────────────────────────────────────────────
    swiglu_limit: float = 10.0

    # ── RoPE ──────────────────────────────────────────────────────────────────
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    apply_rope_fusion: bool = True

    # ── Normalization ─────────────────────────────────────────────────────────
    normalization: str = "RMSNorm"

    # ── FFN ────────────────────────────────────────────────────────────────────
    swiglu: bool = True

    # ── Dropout ────────────────────────────────────────────────────────────────
    attention_dropout: float = 0
    hidden_dropout: float = 0

    # ── Linear bias ────────────────────────────────────────────────────────────
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True

    # ── Vocab / Embedding ──────────────────────────────────────────────────────
    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: int = None
    make_vocab_size_divisible_by: int = 128

    # ── MTP ────────────────────────────────────────────────────────────────────
    mtp_num_layers: int = 0
    mtp_loss_coef: float = 0.1

    model_spec = [
        "loongforge.models.foundation.deepseek_v4.deepseek_v4_layer_spec",
        "get_deepseek_v4_decoder_block_and_mtp_spec",
    ]
    model_type = LanguageModelFamilies.DEEPSEEK_V4
