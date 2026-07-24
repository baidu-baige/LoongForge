# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pi05 DataConfig — data-processing parameters (config, from YAML ``data:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/pi05.yaml``, ``data:`` section) is the
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
Fields that the model also needs (``action_dim``, ``action_horizon``, etc.) are NOT
duplicated here.  The data side reads them from the ``model_cfg`` instance passed
alongside this ``DataConfig``.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Pi05DataConfig:
    """Pi05 data-processing config (maps 1:1 to YAML ``data:`` section)."""

    # Image processing
    image_size: int = 224
    image_normalize_mode: str = "identity"
    image_resize_strategy: str = "resize_with_pad"
    num_images: int = 2
    image_mask: Optional[List[bool]] = None

    # Tokenization
    max_token_len: int = 200

    # Action processing
    normalization_mode: str = "q99"
    use_image_transform: bool = True
    use_action_transform: bool = True
    action_apply_to: List[str] = field(default_factory=lambda: ["action"])
    action_use_statistics: bool = True
    action_padding_strategy: str = "zero"
    action_transform_horizon: Optional[int] = None
    action_transform_max_action_dim: Optional[int] = None

    # Multi-frame observation sampling indices. None means single-frame (default).
    observation_delta_indices: Optional[List[int]] = None
