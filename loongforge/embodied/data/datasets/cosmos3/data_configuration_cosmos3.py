# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

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


@dataclass(frozen=True)
class Cosmos3DroidConfig:
    """Cosmos3 data-processing config (maps 1:1 to YAML ``data:`` section)."""

    action_chunk_length: int = 32
    max_text_tokens: int = 2048
    action_fps: float = 15.0
    cfg_dropout_rate: float = 0.0
    target_h: int = 480
    target_w: int = 480
    video_backend: str = "torchcodec"
    use_image_augmentation: bool = False

    def __post_init__(self) -> None:
        pass
