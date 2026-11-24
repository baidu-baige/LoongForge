# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash configuration."""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch.nn.functional as F

from loongforge.models.common.base_model_config import BaseModelMLAConfig
from loongforge.utils.constants import VisionLanguageModelFamilies


@dataclass
class DeepseekV41Config(BaseModelMLAConfig):
    """LoongForge configuration for the complete DeepSeek-V4.1-Flash model.

    The text and vision configuration are kept together because the released
    checkpoint is one multimodal model.  DSpark and CP are separate runtime
    features and remain disabled for this integration.
    """

    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    group_query_attention: bool = True
    num_query_groups: int = 1
    num_key_value_heads: int = 1
    q_lora_rank: int = 1280
    head_dim: int = 512
    v_head_dim: int = 512
    qk_rope_head_dim: int = 64
    qk_pos_emb_head_dim: int = 64
    multi_latent_attention: bool = True

    num_experts: int = 384
    num_moe_experts: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_layer_freq: Optional[Union[int, List[int]]] = None
    num_experts_per_tok: int = 6
    moe_router_topk: int = 6
    moe_router_score_function: str = "sqrtsoftplus"
    # Preserve FP32 router scores and weights through dispatch.
    moe_router_dtype: Optional[str] = "fp32"
    moe_router_topk_scaling_factor: float = 1.5
    moe_router_enable_expert_bias: bool = True
    norm_topk_prob: bool = True

    csa_window_size: int = 128
    csa_compress_ratios: Optional[List[int]] = None
    kv_source_layer_ids: Optional[List[int]] = None
    index_source_layer_ids: Optional[List[int]] = None
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 512
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    compress_rope_theta: float = 160000.0

    o_groups: int = 8
    o_lora_rank: int = 1024

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    swiglu_limit: float = 10.0

    engram_enabled: bool = True
    engram_layer_ids: Optional[List[int]] = None
    engram_num_embeddings: Optional[List[int]] = None
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092

    image_token_id: int = 129264
    vision_config: Optional[dict] = None

    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = False
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    rope_type: str = "yarn"
    rotary_scaling_factor: float = 16.0
    original_max_position_embeddings: int = 65536
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    apply_rope_fusion: bool = False

    normalization: str = "RMSNorm"
    swiglu: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True

    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: Optional[int] = None
    padded_vocab_size: Optional[int] = None
    max_position_embeddings: int = 1048576
    make_vocab_size_divisible_by: int = 128
    mtp_num_layers: int = 0
    mtp_loss_coef: float = 0.1
    # Released checkpoints serialize DSpark metadata with the text config.
    # Keeping it here lets the native backbone load the complete config while
    # the DSpark objective remains an explicit deferred integration.
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: Optional[List[int]] = None
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    model_type: str = VisionLanguageModelFamilies.DEEPSEEK_V41

    experimental_attention_variant: Optional[str] = None
    dsa_indexer_topk_freq: int = 1
    dsa_indexer_skip_topk_offset: int = 0

    def __post_init__(self, **kwargs):
        if self.vocab_size_in_config_file is not None and self.padded_vocab_size is None:
            self.padded_vocab_size = self.vocab_size_in_config_file
        if self.num_moe_experts is None:
            self.num_moe_experts = self.num_experts
        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size
        if self.moe_shared_expert_intermediate_size is None:
            self.moe_shared_expert_intermediate_size = self.moe_ffn_hidden_size
        self.moe_router_topk = self.num_experts_per_tok
        if self.csa_compress_ratios is None:
            self.csa_compress_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        else:
            self.csa_compress_ratios = list(self.csa_compress_ratios)
        if self.kv_source_layer_ids is None:
            self.kv_source_layer_ids = [2, 8, 14, 20]
        else:
            self.kv_source_layer_ids = list(self.kv_source_layer_ids)
        if self.index_source_layer_ids is None:
            self.index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
        else:
            self.index_source_layer_ids = list(self.index_source_layer_ids)
        if self.engram_layer_ids is None:
            self.engram_layer_ids = [1, 14] if self.engram_enabled else []
        else:
            self.engram_layer_ids = list(self.engram_layer_ids)
        if self.engram_num_embeddings is None:
            self.engram_num_embeddings = [384006168, 384016682] if self.engram_enabled else []
        else:
            self.engram_num_embeddings = list(self.engram_num_embeddings)
        if self.dspark_target_layer_ids is None:
            self.dspark_target_layer_ids = [37, 38, 39]
        else:
            self.dspark_target_layer_ids = list(self.dspark_target_layer_ids)

        self.kv_channels = self.head_dim
        self.num_key_value_heads = 1
        checkpoint_num_query_groups = 1
        # MCore validates query-group divisibility before constructing the
        # model. V4.1's custom attention still owns one shared KV head; expose
        # one validation group per TP rank without changing checkpoint shapes.
        self.num_query_groups = max(1, self.tensor_model_parallel_size)
        self.qk_head_dim = self.head_dim - self.qk_pos_emb_head_dim
        self.kv_lora_rank = self.qk_head_dim
        self.num_residual_streams = self.hc_mult
        self.enable_hyper_connections = True
        self.mhc_sinkhorn_iterations = self.hc_sinkhorn_iters
        self.mhc_init_gating_factor = self.hc_eps
        self.dsa_indexer_head_dim = self.index_head_dim
        self.dsa_indexer_n_heads = self.index_n_heads
        self.dsa_indexer_topk = self.index_topk
        self.activation_func_clamp_value = self.swiglu_limit
        self.gated_linear_unit = self.swiglu
        self.activation_func = F.silu
        self.hetereogenous_dist_checkpoint = True
        self.qk_layernorm = True
        # The released checkpoint uses fp32 RMS arithmetic with this exact
        # epsilon.  Keep it independent from the historical V4 default.
        self.layernorm_epsilon = 1e-20

        # mHC coefficient parameters are replicated across TP ranks and the
        # Loong-Megatron contract requires sequence parallelism for their
        # gradient synchronization. The V4.1 decoder gathers the full
        # sequence before its batch-first attention and scatters it again at
        # the block boundary.
        if self.tensor_model_parallel_size > 1:
            self.sequence_parallel = True

        super().__post_init__(**kwargs)
        self.num_query_groups = checkpoint_num_query_groups

        if self.num_layers > len(self.csa_compress_ratios):
            raise ValueError("DeepSeek-V4.1 csa_compress_ratios must cover every layer")
        active_ratios = self.csa_compress_ratios[: self.num_layers]
        if any(ratio not in (0, 1, 2) for ratio in active_ratios):
            raise ValueError("DeepSeek-V4.1 csa_compress_ratios supports only 0, 1 and 2")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Engram layer IDs and table sizes must have equal length")
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek-V4.1 requires one shared KV head")
        if self.num_experts_per_tok > self.num_moe_experts:
            raise ValueError("num_experts_per_tok must not exceed num_moe_experts")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if self.context_parallel_size != 1:
            raise ValueError("DeepSeek-V4.1 CP is intentionally not enabled in this integration")
        if self.pipeline_model_parallel_size > 1:
            # CSA2 and mHC carry changes the tensor width between PP stages.
            self.variable_seq_lengths = True
        if (
            self.pipeline_model_parallel_size > 1
            and self.pipeline_model_parallel_layout is None
            and self.num_layers_in_first_pipeline_stage is None
            and self.num_layers_in_last_pipeline_stage is None
            and self.num_layers % self.pipeline_model_parallel_size
        ):
            raise ValueError("DeepSeek-V4.1 PP requires divisible num_layers or explicit first/last stage sizes")
        if self.mtp_num_layers:
            raise ValueError("DeepSeek-V4.1 DSpark/MTP is not enabled in this integration")
        if set(self.kv_source_layer_ids) - set(self.index_source_layer_ids):
            raise ValueError("Every V4.1 KV source must also be an index source")
        if any(layer >= self.num_layers for layer in self.engram_layer_ids):
            raise ValueError("Engram layer IDs must refer to active decoder layers")
        if any(layer >= self.num_layers for layer in self.index_source_layer_ids):
            raise ValueError("Index source layer IDs must refer to active decoder layers")
        if any(layer >= self.num_layers for layer in self.kv_source_layer_ids):
            raise ValueError("KV source layer IDs must refer to active decoder layers")
        for layer_id, ratio in enumerate(active_ratios):
            is_kv_source = layer_id in self.kv_source_layer_ids
            is_index_source = layer_id in self.index_source_layer_ids
            if ratio == 0 and (is_kv_source or is_index_source):
                raise ValueError("SWA-only layers cannot be KV or index sources")
            if ratio > 0 and not is_kv_source:
                previous = active_ratios[:layer_id]
                if ratio not in previous:
                    raise ValueError(f"Compressed layer {layer_id} has no preceding ratio-{ratio} KV source")
        if self.candidate_source_layer_id >= self.num_layers:
            self.candidate_source_layer_id = -1
        if self.candidate_source_layer_id >= 0:
            if self.candidate_source_layer_id not in self.kv_source_layer_ids:
                raise ValueError("candidate_source_layer_id must identify a KV source")

        vision = {
            "num_hidden_layers": 32,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 2816,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        }
        vision.update(self.vision_config or {})
        self.vision_config = vision
