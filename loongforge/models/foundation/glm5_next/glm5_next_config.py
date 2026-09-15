# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the LoongForge GLM-5.3-Flash adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch.nn.functional as F

if __package__:
    from loongforge.models.common.base_model_config import BaseModelMLAConfig
else:
    from megatron.core.transformer.transformer_config import MLATransformerConfig as BaseModelMLAConfig

@dataclass
class Glm5NextVisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int = 336
    patch_size: int = 14
    rms_norm_eps: float = 1e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 1536
    intermediate_size: int = 4096
    initializer_range: float = 0.02
    projection_intermediate_size: int = 10240
    swiglu_limit: float = 10.0
    model_type: str = "glm5_next_vision"
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = False
    persist_layer_norm: bool = False
    sequence_parallel: bool = False
    memory_efficient_layer_norm: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("vision hidden_size must be divisible by num_heads")
        if self.patch_size <= 0 or self.temporal_patch_size <= 0 or self.spatial_merge_size <= 0:
            raise ValueError("vision patch and merge sizes must be positive")


@dataclass
class Glm5NextConfig(BaseModelMLAConfig):
    num_layers: int = 45
    hidden_size: int = 4096
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 64
    vocab_size: int = 154880
    intermediate_size: int | None = None
    moe_intermediate_size: int = 2048
    num_hidden_layers: int | None = None
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    rms_norm_eps: float = 1e-5
    swiglu_limit: float = 10.0
    initializer_range: float = 0.02
    max_position_embeddings: int = 1048576
    pad_token_id: int = 154820
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    vision_config: Glm5NextVisionConfig = field(default_factory=Glm5NextVisionConfig)
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    indexer_types: list[str] | None = None
    model_type: str = "glm5_next"
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    pipeline_rank: int | None = None

    position_embedding_type: str = "none"
    add_position_embedding: bool = False
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-5
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    gated_linear_unit: bool = True
    activation_func: object = F.silu
    activation_func_clamp_value: float | None = 10.0
    untie_embeddings_and_output_weights: bool = True
    num_query_groups: int | None = None
    num_moe_experts: int | None = None
    moe_ffn_hidden_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_router_topk: int = 8
    moe_router_num_groups: int | None = 1
    moe_router_group_topk: int | None = 1
    moe_router_score_function: str = "sigmoid"
    moe_router_topk_scaling_factor: float | None = 2.5
    moe_router_enable_expert_bias: bool = True
    moe_router_load_balancing_type: str = "none"
    moe_router_dtype: str | None = "fp32"
    moe_token_dispatcher_type: str = "allgather"
    moe_grouped_gemm: bool = False
    moe_layer_freq: list[int] | int = field(default_factory=list)
    enable_hyper_connections: bool = True
    num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    mhc_init_gating_factor: float = 1.0
    use_fused_mhc: bool = False
    experimental_attention_variant: str = "dsa"
    dsa_indexer_topk_freq: int = 1
    dsa_indexer_skip_topk_offset: int = 0
    dsa_indexer_rotate_activation: bool = False
    dsa_indexer_k_norm_epsilon: float | None = 1e-6
    multi_latent_attention: bool = True
    rope_type: str = "rope"
    transformer_impl: str = "local"
    use_cpu_initialization: bool = True
    padded_vocab_size: int | None = None
    vocab_size_in_config_file: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.vision_config, dict):
            self.vision_config = Glm5NextVisionConfig(**self.vision_config)
        if self.num_hidden_layers is None:
            self.num_hidden_layers = self.num_layers
        else:
            self.num_layers = self.num_hidden_layers
        if self.intermediate_size is None:
            self.intermediate_size = self.ffn_hidden_size
        else:
            self.ffn_hidden_size = self.intermediate_size
        self.num_query_groups = self.num_key_value_heads
        self.qk_head_dim = self.qk_nope_head_dim
        self.qk_pos_emb_head_dim = self.qk_rope_head_dim
        self.dsa_indexer_n_heads = self.index_n_heads
        self.dsa_indexer_head_dim = self.index_head_dim
        self.dsa_indexer_topk = self.index_topk
        self.num_moe_experts = self.n_routed_experts
        self.moe_ffn_hidden_size = self.moe_intermediate_size
        self.moe_shared_expert_intermediate_size = (
            self.moe_intermediate_size * self.n_shared_experts
            if self.n_shared_experts > 0
            else None
        )
        self.moe_router_topk = self.num_experts_per_tok
        self.moe_router_num_groups = self.n_group
        self.moe_router_group_topk = self.topk_group
        self.moe_router_topk_scaling_factor = self.routed_scaling_factor
        self.num_residual_streams = self.hc_mult
        self.mhc_sinkhorn_iterations = self.hc_sinkhorn_iters
        self.layernorm_epsilon = self.rms_norm_eps
        self.activation_func_clamp_value = self.swiglu_limit
        self.padded_vocab_size = self.vocab_size
        self.vocab_size_in_config_file = self.vocab_size
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if index % 4 != 3 else "deepseek_sparse_attention"
                for index in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * max(
                self.num_hidden_layers - 3, 0
            )
        if self.indexer_types is None:
            self.indexer_types = ["full"] * self.num_hidden_layers
        if not (
            len(self.layer_types) == len(self.mlp_layer_types) == len(self.indexer_types) == self.num_hidden_layers
        ):
            raise ValueError("layer schedules must match num_hidden_layers")
        if self.indexer_types[0] != "full":
            raise ValueError("the first GLM-5.3 indexer layer must be full")
        if self.qk_rope_head_dim != 0:
            raise ValueError("GLM-5.3-Flash DSA is NoPE and requires qk_rope_head_dim=0")
        if self.index_topk % self.index_kpool:
            raise ValueError("index_topk must be divisible by index_kpool")
        if self.n_group > 1 and self.num_experts_per_tok // self.topk_group != 2:
            raise ValueError(
                "Loong-Megatron group routing is GLM-equivalent only when "
                "num_experts_per_tok / topk_group == 2"
            )
        if not self.norm_topk_prob and self.num_experts_per_tok > 1:
            raise ValueError("Loong-Megatron sigmoid routing normalizes selected top-k weights")
        self.moe_layer_freq = [int(kind == "sparse") for kind in self.mlp_layer_types]
        super().__post_init__()

    @property
    def num_local_experts(self) -> int:
        return self.n_routed_experts

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, **overrides) -> "Glm5NextConfig":
        payload = json.loads((Path(checkpoint) / "config.json").read_text())
        values = payload.get("text_config", payload)
        accepted = {field.name for field in fields(cls)}
        combined = {key: value for key, value in values.items() if key in accepted}
        # GLM-5.3 releases the MTP depth as num_nextn_predict_layers.
        if "num_nextn_predict_layers" in values and "mtp_num_layers" not in values:
            combined["mtp_num_layers"] = values["num_nextn_predict_layers"]
        for key in (
            "image_token_id",
            "video_token_id",
            "image_start_token_id",
            "image_end_token_id",
            "video_start_token_id",
            "video_end_token_id",
            "vision_config",
        ):
            if key in payload:
                combined[key] = payload[key]
        combined.update(overrides)
        return cls(**combined)
