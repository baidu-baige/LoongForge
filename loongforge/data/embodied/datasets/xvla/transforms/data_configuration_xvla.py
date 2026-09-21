# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""XVLA DataConfig — data-processing parameters (config, from YAML ``data:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/xvla.yaml``, ``data:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``data:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``data_cfg.num_image_views``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``.
3. To add or change a data-processing parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields that the model also needs (e.g. action head dims, action_horizon) live
on ``XvlaModelConfig`` and are NOT duplicated here.  The data side reads them
from the ``model_cfg`` instance passed alongside this ``DataConfig``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class XvlaDataConfig:
    """X-VLA data-processing config (maps 1:1 to YAML ``data:`` section).

    Image resize / normalization is fully handled by ``XVLAPreprocessor``
    (via the checkpoint's ``AutoImageProcessor``), so no generic image
    transform knobs are needed here.  Action normalization is not applied on
    the XVLA path either.  The only remaining data-side knob is the expected
    number of camera views.
    """

    num_image_views: int = 3
