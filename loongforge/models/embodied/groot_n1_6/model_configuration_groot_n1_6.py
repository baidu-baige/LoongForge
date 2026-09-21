#!/usr/bin/env python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""GR00T-N1.6 ModelConfig — model-structure params (config, from YAML ``model:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/groot_n1_6.yaml``, ``model:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``model:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``model_cfg.hidden_size``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``.
3. To add or change a model-structure parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields used by both model and data pipeline (``action_dim``, ``action_horizon``,
``max_action_dim``, ``max_state_dim``, etc.) are defined here once.
``GrootN1d6DataConfig`` does not duplicate them; the data side reads them from
the ``model_cfg`` instance passed alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict


_DEFAULT_EAGLE_ASSETS = "aravindhs-NV/eagle3-processor-groot-n1d6"
_DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.6-3B"


@dataclass(frozen=True)
class GrootN1d6ModelConfig:
    """GR00T-N1.6 model-structure config (maps to YAML ``model:`` section)."""

    model_type: str = "Gr00tN1d6"

    # Backbone identity (structure, also consumed by data-side collator)
    base_model_path: str = _DEFAULT_BASE_MODEL
    model_name: str = _DEFAULT_EAGLE_ASSETS
    vlm_tokenizer_path: str | None = _DEFAULT_EAGLE_ASSETS
    backbone_model_type: str = "eagle"

    # Task dimensions (shared with data side)
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 50
    max_action_dim: int = 128
    max_state_dim: int = 128

    # Embedding / backbone structure
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    backbone_embedding_dim: int = 2048
    select_layer: int = 16
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True
    backbone_trainable_params_fp32: bool = True
    tune_top_llm_layers: int = 4
    tune_llm: bool = False
    tune_visual: bool = False

    # DiT / diffusion structure
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
    add_pos_embed: bool = True
    use_vlln: bool = True
    max_seq_len: int = 1024
    max_num_embodiments: int = 32

    # Flow-matching / noise schedule
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0

    # Trainable-part switches
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    use_bf16: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> "GrootN1d6ModelConfig":
        """Return the typed ModelConfig instance (passthrough) or build from dict/obj.

        After the parameter-system migration, ``build_model`` receives a
        ModelConfig instance directly. This method keeps a small adapter so
        callers passing a dict/OmegaConf still work during construction.
        """
        if isinstance(cfg, cls):
            return cfg
        if hasattr(cfg, "items"):
            items = dict(cfg.items())
        elif isinstance(cfg, dict):
            items = dict(cfg)
        else:
            raise TypeError(
                "GrootN1d6ModelConfig.from_config expects a typed "
                "GrootN1d6ModelConfig or a mapping object."
            )
        values = {
            key: value
            for key, value in items.items()
            if key in cls.__dataclass_fields__ and key != "_target_"
        }
        return cls(**values)

    def __post_init__(self) -> None:
        # frozen dataclass: mutate via object.__setattr__ for env-driven overrides.
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
        if checkpoint_path:
            object.__setattr__(self, "base_model_path", checkpoint_path)

        eagle_local = os.environ.get("EAGLE_LOCAL_PATH")
        if eagle_local:
            object.__setattr__(self, "model_name", eagle_local)
            object.__setattr__(self, "vlm_tokenizer_path", eagle_local)
        else:
            if not self.model_name:
                object.__setattr__(self, "model_name", _DEFAULT_EAGLE_ASSETS)
            if self.vlm_tokenizer_path is None:
                object.__setattr__(self, "vlm_tokenizer_path", _DEFAULT_EAGLE_ASSETS)

        if self.tune_top_llm_layers < 0:
            raise ValueError(
                f"tune_top_llm_layers ({self.tune_top_llm_layers}) must be non-negative"
            )
