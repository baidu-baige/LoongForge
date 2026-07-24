# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""FastWAM policy configuration for LoongForge.

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/fastwam.yaml``, ``model:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``model:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``model_cfg.action_dim``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``:
   - a default supplied there creates a second source of truth and hides the real one;
   - a misspelled field should raise ``AttributeError`` immediately, not silently
     return a fallback.
3. To add or change a model-structure parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields used by both model and data pipeline (``action_dim``, ``action_horizon``,
``proprio_dim``, etc.) are defined here once. ``FastWAMDataConfig`` does not
duplicate them; the data side reads them from the ``model_cfg`` instance
passed alongside.

Nested architecture configs
---------------------------
``video_dit_config``, ``action_dit_config``, ``video_scheduler``,
``action_scheduler``, and ``loss`` carry fixed architecture params for the
Wan2.2-5B backbone.  They are **not** exposed in the YAML; override per-key
after construction when needed for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


@dataclass(frozen=True)
class Cosmos3ModelConfig:
    """Cosmos model-structure config (maps 1:1 to YAML ``model:`` section)."""

    model_type: str = "cosmos3"
    qwen3_vl_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    qk_norm_for_text: bool = True
    qk_norm_for_diffusion: bool = True

    vision_gen: bool = True
    action_gen: bool = True
    sound_gen: bool = False
    latent_patch_size: int = 2
    latent_channel_size: int = 48
    latent_downsample_factor: int = 16
    position_embedding_type: str = "unified_3d_mrope"
    enable_fps_modulation: bool = True
    base_fps: int = 24
    unified_3d_mrope_reset_spatial_ids: bool = True
    unified_3d_mrope_temporal_modality_margin: int = 15000
    max_latent_h: int = 32
    max_latent_w: int = 32
    max_latent_t: int = 32
    joint_attn_implementation: str = "two_way"
    train_time_distribution: str = "logitnormal"
    train_time_weight_method: str = "uniform"
    shift: int = 10
    # VAE encoder.
    vae_path: str = "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    vae_temporal_compression: int = 4
    vae_spatial_compression: int = 16
    # Conditioning distribution: vision-only path uses [0.7, 0.2, 0.1] but
    # the action-policy path overrides condition_frame_indexes_vision per-sample.
    condition_frame_distribution: list = field(default_factory=lambda: [0.7, 0.2, 0.1])
    # Action heads.
    max_action_dim: int = 64
    num_embodiment_domains: int = 32
    action_loss_weight: float = 10.0
    encode_exact_durations: list = field(default_factory=lambda: [33])

    train_modules: list[str] = field(default_factory=lambda: [])
    keys_to_skip_loading: list[str] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        pass
