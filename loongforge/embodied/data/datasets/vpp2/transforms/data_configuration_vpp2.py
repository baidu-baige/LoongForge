# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Typed VPP2 data configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VPP2DataConfig:
    """Data fields that are not part of the VPP2 model architecture."""

    num_frames: int = 65
    action_video_freq_ratio: int = 4
    image_size: int = 224
    context_len: int = 128
    text_embedding_cache_dir: str | None = None
    text_embedding_cache_tag: str = "wan21i2v14b480p"
    norm_stats_path: str | None = None
    delta_action_dim_mask: list[bool] = field(
        default_factory=lambda: [True, True, True, True, True, True, False]
    )
    model_cfg: Any = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.num_frames < 2 or (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames must satisfy (num_frames - 1) % action_video_freq_ratio == 0"
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(0, self.num_frames, self.action_video_freq_ratio))
