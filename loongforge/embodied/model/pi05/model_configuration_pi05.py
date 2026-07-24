# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from lerobot (https://github.com/huggingface/lerobot).
# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Pi05 ModelConfig — model-structure parameters (config, from YAML ``model:`` section).

YAML / config relationship
--------------------------
- The YAML file (``configs/models/embodied/pi05.yaml``, ``model:`` section) is the
  user-facing knob: edit it to override any field for a specific run.
- This dataclass provides all defaults and is the single source of truth.
  At startup, OmegaConf merges the YAML ``model:`` section on top of a structured
  default built from this class, then materialises the result into a frozen instance.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``model_cfg.action_dim``.
2. Never use ``getattr(cfg, "x", default)`` or ``cfg.get("x", default)``:
   - a default supplied there creates a second source of truth and hides the real one;
   - a misspelled field should raise ``AttributeError`` immediately, not silently return
     a fallback.
3. To add or change a model-structure parameter, edit only this dataclass
   (one authoritative definition).

Shared fields
-------------
Fields used by both model and data pipeline (``action_dim``, ``action_horizon``, etc.)
are defined here once.  ``Pi05DataConfig`` does not duplicate them; the data side reads
them from the ``model_cfg`` instance passed alongside.
Tokenizer path is supplied via ``training_args.tokenizer_path`` (``--tokenizer-path``)
or the ``TOKENIZER_PATH`` environment variable — not stored in this config.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pi05ModelConfig:
    """Pi05 model-structure config (maps 1:1 to YAML ``model:`` section)."""

    model_type: str = "pi05"

    # Task dimensions (shared with data side)
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 50
    max_action_dim: int = 32
    max_state_dim: int = 32

    # Training-time model switches
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    # Granularity of torch.compile inside PI05Pytorch.  Ignored when compile_model=False.
    #   "backbone"     — legacy behavior: single compile over PaliGemmaWithExpertModel.forward
    #   "multi_group"  — per-layer + final_norms + action-head bundle (recommended for DDP overlap)
    #   "per_layer"    — only per-layer _compute_layer_complete, everything else eager
    compile_scope: str = "backbone"
    compile_fullgraph: bool = True
    compile_dynamic: bool = False
