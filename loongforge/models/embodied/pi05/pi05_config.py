# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Register pi05 config with the model factory."""

from loongforge.models.factory import register_model_config
from loongforge.utils.constants import VisionLanguageActionModelFamilies

from .configuration_pi05 import PI05Config


@register_model_config(
    model_family=VisionLanguageActionModelFamilies.PI05,
    model_arch=VisionLanguageActionModelFamilies.PI05,
)
def pi05_config():
    """Return a default PI05Config instance."""
    cfg = PI05Config()

    # Megatron-lite defaults so downstream schedulers/modules can find expected fields.
    # These are heuristic values that keep the plumbing satisfied; the underlying model
    # does not use them for architecture definition.
    cfg.tensor_model_parallel_size = getattr(cfg, "tensor_model_parallel_size", 1)
    cfg.pipeline_model_parallel_size = getattr(cfg, "pipeline_model_parallel_size", 1)
    cfg.virtual_pipeline_model_parallel_size = getattr(cfg, "virtual_pipeline_model_parallel_size", None)
    cfg.context_parallel_size = getattr(cfg, "context_parallel_size", 1)
    cfg.expert_model_parallel_size = getattr(cfg, "expert_model_parallel_size", 1)

    cfg.hidden_size = getattr(cfg, "hidden_size", 1024)
    cfg.ffn_hidden_size = getattr(cfg, "ffn_hidden_size", cfg.hidden_size * 4)
    cfg.num_layers = getattr(cfg, "num_layers", 1)
    cfg.num_attention_heads = getattr(cfg, "num_attention_heads", 16)
    cfg.seq_length = getattr(cfg, "seq_length", cfg.tokenizer_max_length)
    cfg.max_position_embeddings = getattr(cfg, "max_position_embeddings", cfg.seq_length)

    cfg.attention_dropout = getattr(cfg, "attention_dropout", 0.0)
    cfg.hidden_dropout = getattr(cfg, "hidden_dropout", 0.0)
    cfg.norm_epsilon = getattr(cfg, "norm_epsilon", 1e-6)
    cfg.add_bias_linear = getattr(cfg, "add_bias_linear", True)
    cfg.add_qkv_bias = getattr(cfg, "add_qkv_bias", True)
    cfg.untie_embeddings_and_output_weights = getattr(cfg, "untie_embeddings_and_output_weights", True)
    cfg.add_position_embedding = getattr(cfg, "add_position_embedding", True)
    cfg.qk_layernorm = getattr(cfg, "qk_layernorm", False)
    cfg.swiglu = getattr(cfg, "swiglu", False)

    cfg.fp16 = getattr(cfg, "fp16", False)
    cfg.bf16 = getattr(cfg, "bf16", False)
    cfg.fp8 = getattr(cfg, "fp8", None)
    cfg.fp4 = getattr(cfg, "fp4", None)
    cfg.enable_autocast = getattr(cfg, "enable_autocast", False)
    cfg.calculate_per_token_loss = getattr(cfg, "calculate_per_token_loss", False)
    cfg.init_model_with_meta_device = getattr(cfg, "init_model_with_meta_device", False)
    cfg.barrier_with_L1_time = getattr(cfg, "barrier_with_L1_time", False)
    cfg.fine_grained_activation_offloading = getattr(cfg, "fine_grained_activation_offloading", False)
    cfg.no_sync_func = getattr(cfg, "no_sync_func", None)
    cfg.overlap_moe_expert_parallel_comm = getattr(cfg, "overlap_moe_expert_parallel_comm", False)
    # Pipeline schedule expects this flag; default to False for pi05.
    cfg.deallocate_pipeline_outputs = getattr(cfg, "deallocate_pipeline_outputs", False)

    return cfg
