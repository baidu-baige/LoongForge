# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 DataConfig for the embodied trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GrootN1d7DataConfig:
    """GR00T-N1.7 data-processing config (maps to YAML ``data:`` section)."""

    embodiment_tag: str = "libero_sim"
    groot_preprocess_mode: str = "sharded"
    max_token_len: Optional[int] = None

    preprocess_action_horizon: Optional[int] = None

    state_dropout_prob: float = 0.2
    exclude_state: bool = False
    use_mean_std: bool = False
    use_percentiles: bool = True
    clip_outliers: bool = True
    apply_sincos_state_encoding: bool = False
    use_relative_action: bool = True

    image_crop_size: Optional[List[int]] = field(default_factory=lambda: [230, 230])
    image_target_size: Optional[List[int]] = field(default_factory=lambda: [256, 256])
    shortest_image_edge: Optional[int] = None
    crop_fraction: Optional[float] = None
    random_rotation_angle: Optional[int] = 0
    color_jitter_params: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        }
    )
    use_albumentations_transforms: bool = True
    extra_augmentation_config: Optional[Dict[str, Any]] = None
    formalize_language: bool = True

    shard_size: int = 1024
    episode_sampling_rate: float = 0.1
    num_shards_per_epoch: int = 100000
    allow_padding: bool = False

    use_image_transform: bool = False
    use_action_transform: bool = False
