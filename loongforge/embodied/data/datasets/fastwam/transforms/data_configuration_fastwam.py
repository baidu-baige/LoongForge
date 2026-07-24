# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM DataConfig — data-processing parameters (config, from YAML ``data:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/fastwam.yaml``, ``data:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``data:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``data_cfg.image_size``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``.
3. To add or change a data-processing parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields that the model also needs (``action_dim``, ``action_horizon``,
``proprio_dim``, etc.) are NOT duplicated here. The data side reads them from
the ``model_cfg`` instance passed alongside this ``DataConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loongforge.embodied.model.fastwam.modeling_configuration_fastwam import FastWAMModelConfig


@dataclass(frozen=True)
class FastWAMDataConfig:
    """FastWAM data-processing config (maps 1:1 to YAML ``data:`` section)."""

    # Image processing
    num_video_frames: int = 5
    image_size: int = 224
    image_normalize_mode: str = "siglip"
    image_resize_strategy: str = "center_crop"
    text_embedding_cache_dir: str | None = None
    use_image_transform: bool = True

    # Action processing
    normalization_mode: str = "q99"
    action_padding_strategy: str = "zero"
    action_video_freq_ratio: int = 4

    # Reference — not serialized, injected at construction time
    model_cfg: Any = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.num_video_frames < 2 or (self.num_video_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                f"num_video_frames must satisfy (T-1) % action_video_freq_ratio == 0 and T >= 2, "
                f"got num_video_frames={self.num_video_frames}, action_video_freq_ratio={self.action_video_freq_ratio}"
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        """Indices into the raw frame sequence for multi-frame observation sampling.

        e.g. num_video_frames=5, action_video_freq_ratio=4
        → num_frames=(5-1)*4+1=17, indices=[0,4,8,12,16]
        """
        ratio = self.action_video_freq_ratio
        num_frames = (self.num_video_frames - 1) * ratio + 1
        return list(range(0, num_frames, ratio))
