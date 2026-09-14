# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain-0 DataConfig — data-processing parameters (from YAML ``data:`` section).

Mirrors ``data/datasets/xvla/transforms/data_configuration_xvla.py``. Fields
map 1:1 onto ``GigaBrain0Transform.__init__`` kwargs (see
``giga_brain_0_transforms.py``), which are unchanged from the reference
``configs/giga_brain_0_agibot_a2d_finetune*.py`` dict-config fields
(``delta_action_cfg`` / ``norm_cfg`` / ``image_cfg`` / ``prompt_cfg``).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GigaBrainDeltaActionConfig:
    use_delta_joint_actions: bool = True
    # embodiment_id (str key, e.g. "1") -> per-dim bool mask (True = delta).
    mask: Dict[str, List[bool]] = field(default_factory=dict)


@dataclass
class GigaBrainNormConfig:
    # embodiment_id (str key) -> path to a norm_stats.json (see reference
    # AgiBotWorldLerobot*/agibot_a2d_norm_stats.json).
    norm_stats_path: Dict[str, str] = field(default_factory=dict)
    use_quantiles: bool = True
    enable_clamp: bool = False


@dataclass
class GigaBrainImageConfig:
    resize_imgs_with_padding: List[int] = field(default_factory=lambda: [224, 224])
    enable_image_aug: bool = True
    present_img_keys: List[str] = field(default_factory=lambda: [
        "observation.images.top_head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ])
    enable_depth_img: bool = False


@dataclass
class GigaBrainPromptSampleRatios:
    task_only: float = 1.0
    task_with_subtask: float = 0.0
    task_only_using_subtask_regression: float = 0.0
    task_only_using_fast_regression: float = 0.0
    task_with_subtask_using_fast_regression: float = 0.0


@dataclass
class GigaBrainPromptConfig:
    tokenizer_model_path: str = ""
    fast_tokenizer_path: str = ""
    max_length: int = 200
    discrete_state_input: bool = True
    encode_action_input: bool = False
    encode_sub_task_input: bool = False
    sample_ratios: GigaBrainPromptSampleRatios = field(default_factory=GigaBrainPromptSampleRatios)


@dataclass(frozen=True)
class GigaBrainDataConfig:
    """GigaBrain-0 data-processing config (maps 1:1 to YAML ``data:`` section)."""

    delta_action_cfg: GigaBrainDeltaActionConfig = field(default_factory=GigaBrainDeltaActionConfig)
    norm_cfg: GigaBrainNormConfig = field(default_factory=GigaBrainNormConfig)
    image_cfg: GigaBrainImageConfig = field(default_factory=GigaBrainImageConfig)
    prompt_cfg: GigaBrainPromptConfig = field(default_factory=GigaBrainPromptConfig)

    # LeRobotDataset path for --dataset-format giga_brain_datasets (see
    # data/datasets/giga_brain_dataset.py). Overridable via --dataset-path.
    data_path: str = ""
