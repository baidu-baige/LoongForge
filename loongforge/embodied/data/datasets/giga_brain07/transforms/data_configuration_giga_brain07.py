# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07 DataConfig — data-pipeline parameters (YAML ``data:`` section).

方案B: the actual dataset + preprocessing are the *giga originals*
(``giga_datasets.LeRobotDataset`` + ``GigaBrain07Transform``). This dataclass
carries every value needed to reconstruct the giga ``data_or_config`` +
``transform`` dicts exactly as in
``configs/giga_brain_0_07_a2d_finetune.py`` (A2D v3.0 finetune). The
``GigaBrain07Trainer._build_dataloaders`` override reads these fields and hands
them to the vendored giga ``load_dataset`` / ``build_transform`` so loss stays
aligned with the giga-brain-0 baseline.

Frozen dataclass; defaults are the single source of truth and are overridable
by the YAML ``data:`` section (paths via oc.env for env-driven runs).
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GigaBrain07DataConfig:
    """GigaBrain07 data config (maps 1:1 to YAML ``data:`` section)."""

    # ── Paths (known-good local A2D v3.0 paths; overridable via oc.env) ──
    data_path: str = (
        "/mnt/cfs/pxy/data/agibot-world/AgiBotWorldLerobot_v30/agibotworld/task_327"
    )
    norm_stats_path: str = (
        "/mnt/cfs/pxy/data/agibot-world/AgiBotWorldLerobot/agibot_a2d_norm_stats.json"
    )
    tokenizer_model_path: str = "/mnt/cfs/pxy/ckpt/paligemma-3b-pt-224"
    fast_tokenizer_path: str = "/mnt/cfs/pxy/ckpt/physical-intelligence/fast"

    # ── LeRobotDataset layout ──
    camera_keys: List[str] = field(
        default_factory=lambda: [
            "observation.images.top_head",
            "observation.images.hand_left",
            "observation.images.hand_right",
        ]
    )
    action_chunk: int = 50
    meta_name: str = "meta"
    video_backend: str = "torchcodec"
    torchcodec_decoder_cache_size: int = 128

    # ── Dataloader ──
    batch_size_per_gpu: int = 4
    num_workers: int = 0            # deterministic default (0); production can bump
    persistent_workers: bool = True
    prefetch_factor: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    sampler_seed: int = 6666

    # ── Transform (GigaBrain07Transform, proprio_anchor) ──
    is_train: bool = True
    enable_image_aug: bool = False  # deterministic default (off)
    state_input_mode: str = "proprio_anchor"
    observation_memory_size: int = 1
    agent_pos_state: int = 32
    use_quaternion_to_6d: bool = False
    robot_type: str = "a2d"
    embodiment_override: str = "AGIBOT_G1"
    embodiment_id: str = "1"
    vlm_type: str = "paligemma2"
    resize_hw: List[int] = field(default_factory=lambda: [224, 224])
    enable_depth_img: bool = False

    # a2d joint-space delta mask (20 dims matching observation.state)
    delta_mask: List[bool] = field(
        default_factory=lambda: [
            True, True, True, True, True, True, True, False, True, True,
            True, True, True, True, True, False, True, True, True, True,
        ]
    )
    use_delta_joint_actions: bool = True
    mask_unsupervised_action_dims_for_noise: bool = True
    use_quantiles: bool = True

    # prompt_cfg
    enable_control_mode_token: bool = True
    enable_end_effector_token: bool = True
    max_length: int = 120
    discrete_state_input: bool = False
    discrete_state_input_for_pose_embodiments: bool = False
    encode_action_input: bool = False
    encode_sub_task_input: bool = False
    encoded_action_horizon: int = 50
    sample_ratios: Dict[str, int] = field(
        default_factory=lambda: {
            "input_task": 1,
            "input_subtask": 0,
            "input_task_target_subtask": 0,
            "input_task_target_action": 0,
            "input_subtask_target_action": 0,
            "input_task_target_subtask_action": 0,
        }
    )
