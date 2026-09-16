# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.
# Copyright (c) Baidu Inc. All rights reserved.
"""DreamZero per-embodiment ModalityConfig definitions.

Embodiment IDs are consumed by WANPolicyHead's embodiment-id projector and must
stay in sync with the model checkpoint metadata.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class ModalityConfig(BaseModel):
    """Define relative sample indices and dataset keys for one modality."""

    delta_indices: list[int]
    eval_delta_indices: list[int] | None = None
    modality_keys: list[str]

    def model_post_init(self, *args, **kwargs):
        """Default evaluation sampling to the training indices."""
        super().model_post_init(*args, **kwargs)
        if self.eval_delta_indices is None:
            self.eval_delta_indices = self.delta_indices


# Embodiment tag -> DreamZero action-head projector ID.
EMBODIMENT_TAG_TO_ID: Dict[str, int] = {
    "oxe_droid": 17,
    "libero_sim": 21,
    "agibot": 26,
    "yam": 32,
    # Extra IDs are referenced by the shared DreamZero collator branches.
    # Keeping them here avoids KeyError when later embodiments fall through
    # earlier branch checks.
    "gr1_unified": 24,
    "mecka_hands": 27,
    "xdof": 22,
    "dream": 31,
    "lapa": 27,
}


def _ranges(n: int) -> List[int]:
    """Return [0, 1, ..., n-1]."""
    return list(range(n))


def _chunked_indices(chunk_size: int, horizon: int, per_chunk: int) -> List[int]:
    return [
        chunk_idx * horizon + offset
        for chunk_idx in range(chunk_size)
        for offset in range(per_chunk)
    ]


def build_droid_modality_configs(
    num_video_frames: int = 33,
    action_horizon: int = 24,
    state_horizon: int = 1,
    max_chunk_size: int = 1,
) -> Dict[str, ModalityConfig]:
    """DROID (oxe_droid): 8d state, 8d action, 3 cameras, 15 fps.

    The state/action schema intentionally uses joint_position(7) plus
    gripper_position(1). Other fields present in some DROID metadata are not
    part of this DreamZero recipe.

    `max_chunk_size` expands the action/state windows by that factor so the
    dataset emits one chunk per latent image block.
    """
    return {
        "video": ModalityConfig(
            delta_indices=_ranges(num_video_frames),
            modality_keys=[
                "video.exterior_image_1_left",
                "video.exterior_image_2_left",
                "video.wrist_image_left",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=_ranges(state_horizon * max_chunk_size),
            modality_keys=[
                "state.joint_position",
                "state.gripper_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=_ranges(action_horizon * max_chunk_size),
            modality_keys=[
                "action.joint_position",
                "action.gripper_position",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "annotation.language.language_instruction",
                "annotation.language.language_instruction_2",
                "annotation.language.language_instruction_3",
            ],
        ),
    }


def build_libero_sim_modality_configs(
    num_video_frames: int = 33,
    action_horizon: int = 16,
    state_horizon: int = 1,
    max_chunk_size: int = 1,
) -> Dict[str, ModalityConfig]:
    """LIBERO sim — 8d state, 7d action, 2 camera views.

    Projector id 21 is used for ``libero_sim``. The two views are formatted as
    a horizontal layout: exterior image on the left and wrist image on the
    right.
    """
    return {
        "video": ModalityConfig(
            delta_indices=_ranges(num_video_frames),
            modality_keys=[
                "video.image",
                "video.wrist_image",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=_chunked_indices(max_chunk_size, action_horizon, state_horizon),
            modality_keys=[
                "state.state",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=_chunked_indices(max_chunk_size, action_horizon, action_horizon),
            modality_keys=[
                "action.actions",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.task"],
        ),
    }


def build_agibot_modality_configs(
    num_video_frames: int = 33,
    action_horizon: int = 24,
    state_horizon: int = 1,
    max_chunk_size: int = 1,
) -> Dict[str, ModalityConfig]:
    """AGIBot — 20d state, 22d action, 3 cameras."""
    return {
        "video": ModalityConfig(
            delta_indices=_ranges(num_video_frames),
            modality_keys=[
                "video.top_head",
                "video.hand_left",
                "video.hand_right",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=_ranges(state_horizon * max_chunk_size),
            modality_keys=[
                "state.left_arm_joint_position",
                "state.right_arm_joint_position",
                "state.left_effector_position",
                "state.right_effector_position",
                "state.head_position",
                "state.waist_position",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=_ranges(action_horizon * max_chunk_size),
            modality_keys=[
                "action.left_arm_joint_position",
                "action.right_arm_joint_position",
                "action.left_effector_position",
                "action.right_effector_position",
                "action.head_position",
                "action.waist_position",
                "action.robot_velocity",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.language.action_text"],
        ),
    }


def build_yam_modality_configs(
    num_video_frames: int = 33,
    action_horizon: int = 24,
    state_horizon: int = 1,
    max_chunk_size: int = 1,
) -> Dict[str, ModalityConfig]:
    """YAM — 14d state, 14d action, 3 cameras, 30 fps."""
    return {
        "video": ModalityConfig(
            delta_indices=_ranges(num_video_frames),
            modality_keys=[
                "video.top_camera-images-rgb",
                "video.left_camera-images-rgb",
                "video.right_camera-images-rgb",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=_ranges(state_horizon * max_chunk_size),
            modality_keys=[
                "state.left_joint_pos",
                "state.left_gripper_pos",
                "state.right_joint_pos",
                "state.right_gripper_pos",
            ],
        ),
        "action": ModalityConfig(
            delta_indices=_ranges(action_horizon * max_chunk_size),
            modality_keys=[
                "action.left_joint_pos",
                "action.left_gripper_pos",
                "action.right_joint_pos",
                "action.right_gripper_pos",
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.task"],
        ),
    }


EMBODIMENT_BUILDERS = {
    "oxe_droid": build_droid_modality_configs,
    "libero_sim": build_libero_sim_modality_configs,
    "agibot": build_agibot_modality_configs,
    "yam": build_yam_modality_configs,
}
