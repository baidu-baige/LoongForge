# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0 model configuration definitions.

Mirrors the pattern used by ``model/xvla/model_configuration_xvla.py``: an
OmegaConf schema dataclass (``GigaBrainModelConfig``) is the structured schema
that ``embodied.train.parser`` merges the YAML ``model:`` section into. Its
fields map 1:1 onto ``GigaBrain0Policy.__init__`` (see ``modeling_giga_brain_0.py``),
which mirrors the reference ``GigaBrain-0.1-3.5B-Base/config.json`` fields
exactly so a checkpoint trained with giga-train loads without remapping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GigaBrainModelConfig:
    """GigaBrain-0 model-structure config (maps 1:1 to YAML ``model:`` section
    and to ``GigaBrain0Policy.__init__`` kwargs / reference ``config.json``)."""

    model_type: str = "giga_brain"

    max_state_dim: int = 32
    max_action_dim: int = 32
    proj_width: int = 1024
    vlm_type: str = "paligemma2"
    vlm_hidden_size: int = 2304
    n_action_steps: int = 50
    num_steps: int = 10
    use_cache: bool = True
    vision_in_channels: int = 3
    enable_knowledge_insulation: bool = False
    enable_next_token_prediction: bool = True
    enable_learnable_traj_token: bool = False
    num_traj_tokens: int = 10
    max_traj_dim: int = 4
    traj_hidden_dim: int = 256
    num_embodiments: int = 1

    # Robot-type -> embodiment id, resolved at data-transform time (see
    # ``data/datasets/giga_brain/giga_brain_0_transforms.py::robot_type_mapping``).
    # Kept here (not just on the data side) so eval/inference code that only
    # has the model config can still resolve it if needed.
    robot_type: str = ""
