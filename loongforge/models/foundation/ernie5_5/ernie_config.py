# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ERNIE 5.5 configuration for the LoongForge LLM trainer."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from loongforge.models.common.base_model_config import BaseModelMLAConfig


@dataclass
class Ernie55Config(BaseModelMLAConfig):
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    model_type: str = "ernie5_5"
    compress_ratios: Optional[List[int]] = None
    first_k_dense_replace: int = 0
    num_experts: int = 256
    moe_latent_size: Optional[int] = None
    moe_use_norm_before_up_proj: bool = False
    norm_topk_prob: bool = True
    mha_vha_rank: int = 16
    activation_situ_beta: Optional[float] = None
    activation_situ_linear_beta: Optional[float] = None
    d_num_attention_heads: int = 64
    d_q_lora_rank: int = 1024
    d_kv_lora_rank: int = 512
    d_qk_head_dim: int = 448
    d_qk_pos_emb_head_dim: int = 64
    d_v_head_dim: int = 256
    hca_rope_type: str = "yarn"
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    csa_window_size: int = 128
    csa_compress_rotary_base: float = 160000.0
    csa_dense_mode: bool = True
    force_unfused_dsa: bool = True
    vocab_size_in_config_file: int = 200027
    make_vocab_size_divisible_by: int = 1
    position_embedding_type: str = "none"
    group_query_attention: bool = True
    num_query_groups: int = 1
    normalization: str = "RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = True
    apply_rope_fusion: bool = False
    mtp_shared_last_layer: bool = True

    def __post_init__(self, **kwargs):
        if self.compress_ratios is None or len(self.compress_ratios) != self.num_layers:
            raise ValueError("compress_ratios must contain one entry per decoder layer.")
        self.compress_ratios = list(self.compress_ratios)
        if any(ratio not in (128, -2) for ratio in self.compress_ratios):
            raise ValueError("ERNIE 5.5 supports H (128) and D (-2) attention only.")
        if self.tensor_model_parallel_size != 1 or self.expert_tensor_parallel_size not in (None, 1):
            raise ValueError("ERNIE 5.5 requires TP=1 and ETP=1.")
        if self.context_parallel_size != 1:
            raise ValueError("ERNIE 5.5 does not support context parallelism yet.")
        ep_size = self.expert_model_parallel_size or 1
        if self.num_experts % ep_size:
            raise ValueError("ERNIE 5.5 num_experts must be divisible by expert_model_parallel_size.")
        if not self.bf16 or self.fp16 or self.fp8:
            raise ValueError("ERNIE 5.5 requires BF16 parameters and activations.")
        if self.hidden_dropout or self.attention_dropout:
            raise ValueError("ERNIE 5.5 requires zero dropout.")
        if self.recompute_hyper_connections:
            raise ValueError("ERNIE 5.5 does not support selective hyper-connection recompute.")
        if self.mtp_num_layers not in (None, 0, 1):
            raise ValueError("ERNIE 5.5 supports one shared MTP layer.")
        if self.mtp_num_layers and (not self.mtp_shared_last_layer or self.compress_ratios[-1] != -2):
            raise ValueError("ERNIE 5.5 MTP must share the last D layer.")
        self.num_moe_experts = self.num_experts
        self.activation_func = F.silu
        self.gated_linear_unit = True
        self.use_te_activation_func = self.activation_situ_beta is not None
        self.bias_activation_fusion = False
        self.enable_hyper_connections = True
        self.num_residual_streams = self.hc_mult
        self.mhc_sinkhorn_iterations = self.hc_sinkhorn_iters
        self.use_fused_mhc = False
        # compress_ratios is the checkpoint/ERNIE view (-2 marks a D layer);
        # Megatron-Core's inherited csa_compress_ratios is derived from it (0 marks D).
        self.csa_compress_ratios = [128 if ratio == 128 else 0 for ratio in self.compress_ratios]
        self.csa_compress_ratios += [0] * (self.mtp_num_layers or 0)
        self.dsa_indexer_n_heads = self.index_n_heads
        self.dsa_indexer_head_dim = self.index_head_dim
        self.dsa_indexer_topk = self.index_topk
        self.dsa_indexer_rope_interleaved = False
        self.dsa_indexer_rotate_activation = True
        self.moe_router_score_function = "sigmoid"
        self.moe_router_enable_expert_bias = True
        self.moe_router_load_balancing_type = "none"
        self.moe_router_dtype = "fp32"
        self.moe_aux_loss_coeff = 0.0
        self.moe_shared_expert_overlap = False
        self.moe_router_pre_softmax = False
        if self.moe_latent_size is None:
            self.moe_use_norm_before_up_proj = False
        self.hetereogenous_dist_checkpoint = True
        super().__post_init__(**kwargs)
        if self.pipeline_model_parallel_size > 1:
            self.pipeline_dtype = torch.float32
