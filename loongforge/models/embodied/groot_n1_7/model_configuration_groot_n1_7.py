#!/usr/bin/env python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""Configuration for GR00T-N1.7 in the embodied trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict


_DEFAULT_BACKBONE = "nvidia/Cosmos-Reason2-2B"
_DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.7-3B"


@dataclass
class GrootN1d7Config:
    """Single flat config object consumed by model, data, and trainer code."""

    model_type: str = "Gr00tN1d7"
    base_model_path: str = _DEFAULT_BASE_MODEL
    model_name: str = _DEFAULT_BACKBONE
    backbone_model_type: str = "qwen"
    model_revision: str | None = None

    model_dtype: str = "bfloat16"
    tune_top_llm_layers: int = 0
    backbone_embedding_dim: int = 2048
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False
    backbone_trainable_params_fp32: bool = True

    max_state_dim: int = 132
    max_action_dim: int = 132
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    state_history_length: int = 1

    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2
    diffusion_model_cfg: Dict[str, Any] = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 32,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )
    use_vl_self_attention: bool = True
    vl_self_attention_cfg: Dict[str, Any] | None = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 4,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "dropout": 0.2,
            "final_dropout": True,
        }
    )

    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    use_bf16: bool = True

    state_dropout_prob: float = 0.2
    max_num_embodiments: int = 32

    @classmethod
    def from_config(cls, cfg: Any) -> "GrootN1d7Config":
        """Create from OmegaConf/dict/object."""
        if isinstance(cfg, cls):
            return cfg
        if hasattr(cfg, "items"):
            items = dict(cfg.items())
        elif isinstance(cfg, dict):
            items = dict(cfg)
        else:
            items = {
                key: getattr(cfg, key)
                for key in cls.__dataclass_fields__
                if hasattr(cfg, key)
            }
        values = {
            key: value
            for key, value in items.items()
            if key in cls.__dataclass_fields__ and key != "_target_"
        }
        return cls(**values)

    def __post_init__(self) -> None:
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
        if checkpoint_path:
            self.base_model_path = checkpoint_path

        cosmos_local = os.environ.get("COSMOS_LOCAL_PATH")
        if cosmos_local:
            self.model_name = cosmos_local

        if self.tune_top_llm_layers < 0:
            raise ValueError(f"tune_top_llm_layers ({self.tune_top_llm_layers}) must be non-negative")
