# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.
#
# Copyright 2024 NVIDIA. All rights reserved.
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

"""Embodiment, action and modality configs for Gr00t N1.6, plus LeRobot stats conversion."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import torch


class ActionRepresentation(Enum):
    """Action representation types."""
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    """Action types."""
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    """Action formats."""
    DEFAULT = "default"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"


class EmbodimentTag(Enum):
    """Embodiment tags."""
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    GR1 = "gr1"
    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    UNITREE_G1 = "unitree_g1"
    LIBERO_PANDA = "libero_panda"
    OXE_GOOGLE = "oxe_google"
    OXE_WIDOWX = "oxe_widowx"
    NEW_EMBODIMENT = "new_embodiment"


class ActionConfig:
    """Action configuration class."""
    def __init__(
        self,
        rep: ActionRepresentation | str,
        type: ActionType | str,
        format: ActionFormat | str,
        state_key: str | None = None,
    ):
        if isinstance(rep, str):
            rep = ActionRepresentation[rep]
        if isinstance(type, str):
            type = ActionType[type]
        if isinstance(format, str):
            format = ActionFormat[format]

        self.rep = rep
        self.type = type
        self.format = format
        self.state_key = state_key


class ModalityConfig:
    """Modality configuration class."""
    def __init__(
        self,
        delta_indices: list[int],
        modality_keys: list[str],
        sin_cos_embedding_keys: list[str] | None = None,
        mean_std_embedding_keys: list[str] | None = None,
        action_configs: list[ActionConfig | dict] | None = None,
    ):
        self.delta_indices = delta_indices
        self.modality_keys = modality_keys
        self.sin_cos_embedding_keys = sin_cos_embedding_keys
        self.mean_std_embedding_keys = mean_std_embedding_keys
        if action_configs is not None:
            parsed_action_configs = []
            for action_config in action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        rep=ActionRepresentation[action_config["rep"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                        state_key=action_config.get("state_key", None),
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs
        else:
            self.action_configs = None


"""EMBODIMENT_TAG_TO_PROJECTOR_INDEX mapping"""
EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    "unitree_g1": 8,
    "libero_panda": 2,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "new_embodiment": 10,
}

"""SO100 modality metadata configuration"""
SO100_MODALITY_META = {
    "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
}

"""SO100 modality configuration"""
SO100_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

"""Libero Panda modality metadata configuration"""
LIBERO_PANDA_MODALITY_META = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
}

"""Libero Panda modality configuration"""
LIBERO_PANDA_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["image", "image2"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
}

"""Behavior R1 Pro modality metadata configuration"""
BEHAVIOR_R1_PRO_MODALITY_META = {
    "state": {
        "robot_pos": {"start": 0, "end": 3},
        "robot_ori_cos": {"start": 3, "end": 6},
        "robot_ori_sin": {"start": 6, "end": 9},
        "robot_2d_ori": {"start": 9, "end": 10},
        "robot_2d_ori_cos": {"start": 10, "end": 11},
        "robot_2d_ori_sin": {"start": 11, "end": 12},
        "robot_lin_vel": {"start": 12, "end": 15},
        "robot_ang_vel": {"start": 15, "end": 18},
        "arm_left_qpos": {"start": 18, "end": 25},
        "arm_left_qpos_sin": {"start": 25, "end": 32},
        "arm_left_qpos_cos": {"start": 32, "end": 39},
        "eef_left_pos": {"start": 39, "end": 42},
        "eef_left_quat": {"start": 42, "end": 46},
        "gripper_left_qpos": {"start": 46, "end": 48},
        "arm_right_qpos": {"start": 48, "end": 55},
        "arm_right_qpos_sin": {"start": 55, "end": 62},
        "arm_right_qpos_cos": {"start": 62, "end": 69},
        "eef_right_pos": {"start": 69, "end": 72},
        "eef_right_quat": {"start": 72, "end": 76},
        "gripper_right_qpos": {"start": 76, "end": 78},
        "trunk_qpos": {"start": 78, "end": 82},
    },
    "action": {
        "base": {"start": 0, "end": 3},
        "torso": {"start": 3, "end": 7},
        "left_arm": {"start": 7, "end": 14},
        "left_gripper": {"start": 14, "end": 15},
        "right_arm": {"start": 15, "end": 22},
        "right_gripper": {"start": 22, "end": 23},
    },
}

"""Behavior R1 Pro modality configuration"""
BEHAVIOR_R1_PRO_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["rgb.head_256_256", "rgb.left_wrist_256_256", "rgb.right_wrist_256_256"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "robot_pos",
            "robot_ori_cos",
            "robot_ori_sin",
            "robot_2d_ori",
            "robot_2d_ori_cos",
            "robot_2d_ori_sin",
            "robot_lin_vel",
            "robot_ang_vel",
            "arm_left_qpos",
            "arm_left_qpos_sin",
            "arm_left_qpos_cos",
            "eef_left_pos",
            "eef_left_quat",
            "gripper_left_qpos",
            "arm_right_qpos",
            "arm_right_qpos_sin",
            "arm_right_qpos_cos",
            "eef_right_pos",
            "eef_right_quat",
            "gripper_right_qpos",
            "trunk_qpos",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(32)),
        modality_keys=["base", "torso", "left_arm", "left_gripper", "right_arm", "right_gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="trunk_qpos",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm_left_qpos",
            ),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="arm_right_qpos",
            ),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
}

"""OXE WidowX (Bridge) modality metadata configuration"""
OXE_WIDOWX_MODALITY_META = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "pad": {"start": 6, "end": 7},
        "gripper": {"start": 7, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
}

"""OXE WidowX (Bridge) modality configuration"""
OXE_WIDOWX_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["image"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 8)),
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        # Matches the bridge checkpoint's processor_config.json: action horizon
        # is 8 (delta_indices [0..7]) and pos+rotation use mean-std normalization
        # (gripper stays min-max). Without mean-std the euler action is min-max'd
        # over the huge ±2pi range and blows up.
        mean_std_embedding_keys=["x", "y", "z", "roll", "pitch", "yaw"],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
}

"""Modality configurations mapping"""
MODALITY_CONFIGS = {
    "new_embodiment": SO100_MODALITY_CONFIG,
    "libero_panda": LIBERO_PANDA_MODALITY_CONFIG,
    "behavior_r1_pro": BEHAVIOR_R1_PRO_MODALITY_CONFIG,
    "oxe_widowx": OXE_WIDOWX_MODALITY_CONFIG,
}

"""Embodiment statistics configurations"""
EMBODIMENT_STAT_CONFIGS = {
    "new_embodiment": {"modality_meta": SO100_MODALITY_META, "modality_config": SO100_MODALITY_CONFIG},
    "libero_panda": {"modality_meta": LIBERO_PANDA_MODALITY_META, "modality_config": LIBERO_PANDA_MODALITY_CONFIG},
    "behavior_r1_pro": {
        "modality_meta": BEHAVIOR_R1_PRO_MODALITY_META,
        "modality_config": BEHAVIOR_R1_PRO_MODALITY_CONFIG,
    },
    "oxe_widowx": {
        "modality_meta": OXE_WIDOWX_MODALITY_META,
        "modality_config": OXE_WIDOWX_MODALITY_CONFIG,
    },
}


def _slice_stats_by_joint_group(stats_dict: dict[str, Any], start_idx: int, end_idx: int) -> dict[str, list]:
    """Slice statistics dictionary by joint group indices."""
    sliced_stats = {}
    for stat_type, values in stats_dict.items():
        if isinstance(values, torch.Tensor):
            sliced_stats[stat_type] = values[start_idx:end_idx].cpu().tolist()
        elif isinstance(values, (list, np.ndarray)):
            sliced_stats[stat_type] = list(np.array(values)[start_idx:end_idx])
        else:
            sliced_stats[stat_type] = values
    return sliced_stats


def _get_lerobot_stats_key(modality: str) -> str:
    """Get the corresponding key for LeRobot stats based on modality."""
    mapping = {"state": "observation.state", "action": "action", "relative_action": "relative_action"}
    return mapping.get(modality, modality)


def convert_lerobot_stats_to_processor_format(
    dataset_stats: dict[str, dict[str, Any]],
    embodiment_tag: str,
) -> dict[str, Any]:
    """
    Convert LeRobot statistics to processor format.

    Args:
        dataset_stats: Dictionary containing dataset statistics.
        embodiment_tag: String identifier for the embodiment.

    Returns:
        Dictionary containing processed statistics.

    Raises:
        ValueError: If embodiment_tag is not found in EMBODIMENT_STAT_CONFIGS.
    """
    if embodiment_tag not in EMBODIMENT_STAT_CONFIGS:
        available_embodiments = list(EMBODIMENT_STAT_CONFIGS.keys())
        raise ValueError(
            f"Embodiment '{embodiment_tag}' not found in EMBODIMENT_STAT_CONFIGS. "
            f"Available embodiments: {available_embodiments}."
        )
    config = EMBODIMENT_STAT_CONFIGS[embodiment_tag]
    modality_meta = config["modality_meta"]
    modality_config = config["modality_config"]

    statistics = {embodiment_tag: {}}

    for modality in ["state", "action"]:
        lerobot_key = _get_lerobot_stats_key(modality)
        stats_dict = dataset_stats[lerobot_key]
        statistics[embodiment_tag][modality] = {}
        modality_cfg = modality_config[modality]
        joint_groups = modality_cfg.modality_keys

        for joint_group in joint_groups:
            start_idx = modality_meta[modality][joint_group]["start"]
            end_idx = modality_meta[modality][joint_group]["end"]
            statistics[embodiment_tag][modality][joint_group] = _slice_stats_by_joint_group(
                stats_dict, start_idx, end_idx
            )

    action_modality = modality_config["action"]
    action_configs = action_modality.action_configs
    needs_relative_stats = any(cfg.rep == ActionRepresentation.RELATIVE for cfg in (action_configs or []))

    if needs_relative_stats:
        if "relative_action" not in dataset_stats:
            raise ValueError(
                f"Embodiment '{embodiment_tag}' requires relative_action statistics."
            )

        statistics[embodiment_tag]["relative_action"] = {}
        for joint_group, action_config in zip(action_modality.modality_keys, action_configs, strict=False):
            if action_config.rep != ActionRepresentation.RELATIVE:
                continue

            if joint_group not in dataset_stats["relative_action"]:
                raise KeyError(
                    f"Missing relative_action stats for joint group '{joint_group}'. "
                    f"Available joint groups: {list(dataset_stats['relative_action'].keys())}"
                )

            relative_stats = dataset_stats["relative_action"][joint_group]
            statistics[embodiment_tag]["relative_action"][joint_group] = {
                stat_type: values if isinstance(values, list) else values
                for stat_type, values in relative_stats.items()
            }

    return statistics
