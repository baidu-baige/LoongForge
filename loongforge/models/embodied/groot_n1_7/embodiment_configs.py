# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""Embodiment, modality configs and LeRobot stats conversion for GR00T-N1.7."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch

from loongforge.models.embodied.groot_n1_6.embodiment_configs import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}


LIBERO_SIM_MODALITY_META = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        # robot0_gripper_qpos is 2D (both finger joints); the official
        # LiberoEnv declares state.gripper with shape (2,) and the released
        # checkpoint statistics carry 2 values per stat for this group.
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


LIBERO_SIM_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["image", "wrist_image"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ]
        * 7,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


SIMPLER_ENV_WIDOWX_MODALITY_META = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        # Dead channel: the Bridge statistics carry q01 == q99 == 0.0 for
        # ``pad`` and the official ``WidowXBridgeEnv`` always feeds 0. It still
        # has to occupy index 6 so ``gripper`` lands on index 7.
        "pad": {"start": 6, "end": 7},
        # 1D normalized openness in [0, 1] (1.0 = fully open), unlike
        # ``libero_sim`` whose gripper slot is the 2D finger qpos.
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


# Transcribed from ``processor_config.json`` of
# ``nvidia/GR00T-N1.7-SimplerEnv-Bridge`` (``processor_kwargs.modality_configs
# ["simpler_env_widowx"]``): a single exterior view, an 8D state whose 7th slot
# is a dead pad channel, and an 8-step absolute action chunk.
SIMPLER_ENV_WIDOWX_MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["image_0"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(8)),
        modality_keys=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ]
        * 7,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


NEW_EMBODIMENT_MODALITY_META = {
    "state": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
    "action": {
        "single_arm": {"start": 0, "end": 5},
        "gripper": {"start": 5, "end": 6},
    },
}


NEW_EMBODIMENT_MODALITY_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["exterior_1_left", "wrist_left"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.language.language_instruction"],
    ),
}


MODALITY_CONFIGS = {
    "libero_sim": LIBERO_SIM_MODALITY_CONFIG,
    "simpler_env_widowx": SIMPLER_ENV_WIDOWX_MODALITY_CONFIG,
    "new_embodiment": NEW_EMBODIMENT_MODALITY_CONFIG,
}


EMBODIMENT_STAT_CONFIGS = {
    "libero_sim": {
        "modality_meta": LIBERO_SIM_MODALITY_META,
        "modality_config": LIBERO_SIM_MODALITY_CONFIG,
    },
    "simpler_env_widowx": {
        "modality_meta": SIMPLER_ENV_WIDOWX_MODALITY_META,
        "modality_config": SIMPLER_ENV_WIDOWX_MODALITY_CONFIG,
    },
    "new_embodiment": {
        "modality_meta": NEW_EMBODIMENT_MODALITY_META,
        "modality_config": NEW_EMBODIMENT_MODALITY_CONFIG,
    },
}


NEW_EMBODIMENT_GROUP_STAT_KEYS = {
    "state": {
        "single_arm": "observation.state.single_arm",
        "gripper": "observation.state.gripper",
    },
    "action": {
        "single_arm": "action.single_arm",
        "gripper": "action.gripper",
    },
}


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


def convert_lerobot_stats_to_groot_n1d7_format(
    dataset_stats: Dict[str, Any],
    embodiment_tag: str = "libero_sim",
    *,
    modality_meta: dict[str, dict[str, dict[str, int]]] | None = None,
    modality_config: dict[str, ModalityConfig] | None = None,
) -> dict:
    """Convert flat LoongForge LeRobot stats to processor-style GR00T stats."""
    if "statistics" in dataset_stats and isinstance(dataset_stats["statistics"], dict):
        dataset_stats = dataset_stats["statistics"]
    if modality_meta is None or modality_config is None:
        if embodiment_tag not in EMBODIMENT_STAT_CONFIGS:
            raise ValueError(f"Unsupported GR00T-N1.7 embodiment tag: {embodiment_tag}")
        modality_meta = EMBODIMENT_STAT_CONFIGS[embodiment_tag]["modality_meta"]
        modality_config = EMBODIMENT_STAT_CONFIGS[embodiment_tag]["modality_config"]
    if not modality_meta or not modality_config:
        raise ValueError(f"Unsupported GR00T-N1.7 embodiment tag: {embodiment_tag}")
    statistics = {embodiment_tag: {}}

    stats_key_map = {"state": "observation.state", "action": "action"}
    for modality in ("state", "action"):
        source_key = stats_key_map[modality]
        if source_key not in dataset_stats:
            raise KeyError(f"Missing dataset statistics key '{source_key}'")
        source_stats = dataset_stats[source_key]
        statistics[embodiment_tag][modality] = {}
        for joint_group in modality_config[modality].modality_keys:
            group_source_stats = _get_group_source_stats(
                dataset_stats,
                embodiment_tag,
                modality,
                joint_group,
            )
            if group_source_stats is not None:
                statistics[embodiment_tag][modality][joint_group] = _copy_stats(group_source_stats)
            else:
                meta = modality_meta[modality][joint_group]
                statistics[embodiment_tag][modality][joint_group] = _slice_stats(
                    source_stats,
                    meta["start"],
                    meta["end"],
                )

    if "relative_action" in dataset_stats:
        statistics[embodiment_tag]["relative_action"] = deepcopy(dataset_stats["relative_action"])
    return statistics


def _slice_stats(stats_dict: dict[str, Any], start_idx: int, end_idx: int) -> dict[str, list]:
    sliced = {}
    for stat_type, values in stats_dict.items():
        arr = _to_numpy(values)
        sliced[stat_type] = arr[start_idx:end_idx].tolist()
    return sliced


def _copy_stats(stats_dict: dict[str, Any]) -> dict[str, list]:
    copied = {}
    for stat_type, values in stats_dict.items():
        arr = _to_numpy(values)
        copied[stat_type] = arr.tolist()
    return copied


def _get_group_source_stats(
    dataset_stats: dict[str, Any],
    embodiment_tag: str,
    modality: str,
    joint_group: str,
) -> dict[str, Any] | None:
    if embodiment_tag != "new_embodiment":
        return None
    source_key = NEW_EMBODIMENT_GROUP_STAT_KEYS.get(modality, {}).get(joint_group)
    if source_key is None:
        return None
    value = dataset_stats.get(source_key)
    return value if isinstance(value, dict) else None
