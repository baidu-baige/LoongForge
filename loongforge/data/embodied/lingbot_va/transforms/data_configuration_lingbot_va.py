# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot-VA DataConfig for embodied typed configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class LingBotVADataConfig:
    """LingBot-VA data-processing config, mapped from YAML ``data:``."""

    dataset_path: Optional[str] = None
    empty_emb_path: Optional[str] = None
    obs_cam_keys: List[str] = field(default_factory=list)
    inverse_used_action_channel_ids: List[int] = field(default_factory=list)
    norm_q01: List[float] = field(default_factory=list)
    norm_q99: List[float] = field(default_factory=list)
    cfg_prob: float = 0.1
    env_type: str = "none"
    revision: str = "v2.1"
    video_backend: str = "pyav"
    action_dim: int = 30
    latent_dir_name: str = "latents"
    metadata_filename: str = "info.json"
    allow_missing_lerobot: bool = False
    pad_to_multiple: int = 1
