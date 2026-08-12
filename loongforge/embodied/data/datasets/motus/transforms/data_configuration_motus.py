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

"""Motus DataConfig — data-processing params (config, from YAML ``data:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/motus.yaml``, ``data:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``data:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``data_cfg.embodiment_type``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``.
3. To add or change a data-processing parameter, edit only this dataclass.

Shared fields
-------------
Video geometry / sampling fields (``num_video_frames``, ``video_height``,
``video_width``, ``video_action_freq_ratio``, ``global_downsample_rate``,
``action_dim``, ``state_dim``) are NOT duplicated here — the data side reads them
from the ``model_cfg`` (``MotusModelConfig``) instance passed alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MotusDataConfig:
    """Motus data-processing config (maps 1:1 to YAML ``data:`` section)."""

    # ---- Dataset location (LeRobotMotusDataset needs repo_id + root) ----
    repo_id: str = ""
    root: str = ""

    # ---- Task selection ----
    task_mode: str = "single"  # "single" or "multi"
    # None = all tasks; str = single task; list[str] = specific tasks (multi mode).
    # Typed ``Any`` because OmegaConf structured configs reject Union-of-containers.
    task_name: Any = None
    max_episodes: Optional[int] = None
    image_aug: bool = False

    # ---- Normalization / embodiment ----
    embodiment_type: str = "aloha_agilex_2"

    # ---- Video decoding ----
    video_backend: str = "pyav"

    # ---- T5 language embedding (on-the-fly encode + cache fallback) ----
    enable_t5_fallback: bool = True
    t5_wan_path: str = ""
    t5_folder_name: str = "t5_embedding"
    t5_text_len: int = 512
