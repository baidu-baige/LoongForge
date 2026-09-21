# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.6 DataConfig — data-processing params (config, from YAML ``data:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/groot_n1_6.yaml``, ``data:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``data:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``data_cfg.embodiment_tag``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``.
3. To add or change a data-processing parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields that the model also needs (``action_horizon``, ``max_action_dim``,
``max_state_dim``, ``backbone_model_type``) are NOT duplicated here.  The data side
reads them from the ``model_cfg`` instance passed alongside this ``DataConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GrootN1d6DataConfig:
    """GR00T-N1.6 data-processing config (maps to YAML ``data:`` section)."""

    embodiment_tag: str = "libero_panda"
    groot_preprocess_mode: str = "sample"
    max_token_len: Optional[int] = None
    backbone_model_type: str = "eagle"

    # Preprocessing dims (data-side padding/truncation, distinct from model dims)
    preprocess_action_horizon: int = 16
    preprocess_max_action_dim: int = 29
    preprocess_max_state_dim: int = 29

    # Image geometry / augmentation
    image_crop_size: List[int] = field(default_factory=lambda: [224, 224])
    image_target_size: List[int] = field(default_factory=lambda: [224, 224])
    shortest_image_edge: Optional[int] = 256
    crop_fraction: Optional[float] = 0.95
    random_rotation_angle: Optional[int] = None
    color_jitter_params: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        }
    )
    use_albumentations_transforms: bool = True
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = False
    use_relative_action: bool = True
    use_processor_image_size: bool = False

    # Multi-frame observation sampling indices. None means single-frame (default).
    observation_delta_indices: Optional[List[int]] = None
