# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Motus ModelConfig — model-structure params (config, from YAML ``model:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/motus.yaml``, ``model:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``model:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``model_cfg.action_dim``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``.
3. To add or change a model-structure parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields used by both model and data pipeline (``action_dim``, ``state_dim``,
``num_video_frames``, ``video_action_freq_ratio``, ``global_downsample_rate``,
``video_height``, ``video_width``) are defined here once. ``MotusDataConfig`` does
not duplicate them; the data side reads them from the ``model_cfg`` instance passed
alongside.

Field-name mapping to the original ``models.motus.MotusConfig``
---------------------------------------------------------------
This config is a flat YAML-facing view; ``MotusPolicy.from_pretrained`` translates it
into the nested ``MotusConfig`` that the transplanted model expects. The mapping is
1:1 by name except for the nested-vs-flat layout (see modeling_motus.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class MotusModelConfig:
    """Motus model-structure config (maps to YAML ``model:`` section)."""

    model_type: str = "motus"

    # ---- WAN video backbone (structure + weights) ----
    wan_config_path: str = ""
    wan_checkpoint_path: str = ""
    vae_path: str = ""
    video_precision: str = "bfloat16"

    # ---- VLM (frozen understanding backbone) ----
    vlm_checkpoint_path: str = ""
    vlm_precision: str = "bfloat16"

    # ---- Task dimensions (shared with data side) ----
    action_dim: int = 14
    state_dim: int = 14

    # ---- Action expert ----
    action_expert_hidden_size: int = 1024
    action_expert_ffn_dim_multiplier: int = 4
    action_expert_norm_eps: float = 1e-6

    # ---- Understanding expert ----
    und_expert_hidden_size: int = 512
    und_expert_ffn_dim_multiplier: int = 4
    und_expert_norm_eps: float = 1e-6
    vlm_adapter_input_dim: int = 2048
    vlm_adapter_projector_type: str = "mlp3x_silu"

    # ---- MoT backbone depth (WAN joint/cross layers) ----
    num_layers: int = 30

    # ---- Video / sampling geometry (shared with data side) ----
    num_video_frames: int = 8
    video_action_freq_ratio: int = 6
    global_downsample_rate: int = 1
    video_height: int = 384
    video_width: int = 320

    # ---- Loss weights ----
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # ---- Training mode / switches ----
    training_mode: str = "finetune"  # 'pretrain' or 'finetune'
    # None = default (load), False = init from config only (no pretrained weights).
    load_pretrained_backbones: Optional[bool] = None

    @classmethod
    def from_config(cls, cfg: Any) -> "MotusModelConfig":
        """Return the typed ModelConfig (passthrough) or build from a mapping.

        ``build_model`` normally passes a typed ``MotusModelConfig`` directly; this
        adapter keeps dict/OmegaConf callers working during construction.
        """
        if isinstance(cfg, cls):
            return cfg
        if hasattr(cfg, "items"):
            items = dict(cfg.items())
        elif isinstance(cfg, dict):
            items = dict(cfg)
        else:
            raise TypeError(
                "MotusModelConfig.from_config expects a typed MotusModelConfig "
                "or a mapping object."
            )
        values = {
            key: value
            for key, value in items.items()
            if key in cls.__dataclass_fields__ and key != "_target_"
        }
        return cls(**values)
