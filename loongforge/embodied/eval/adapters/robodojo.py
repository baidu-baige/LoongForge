# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo (Isaac Sim, dual-arm ARX X5) benchmark adapter.

RoboDojo's official sim client hands the policy an observation dict with
``vision.<cam>.color`` (HWC uint8 RGB), ``state.<arm|ee>`` vectors and the
episode ``instruction``; it executes action dicts keyed by
``{left,right}_arm_joint_state`` (6) and ``{left,right}_ee_joint_state`` (1).

The 14-D joint vector used throughout this adapter follows XPolicyLab's
``pack_robot_state`` dual-arm order: left arm 6, left gripper 1, right arm 6,
right gripper 1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseBenchmarkAdapter

ROBODOJO_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ROBODOJO_JOINT_LAYOUT = (
    ("left_arm_joint_state", 6),
    ("left_ee_joint_state", 1),
    ("right_arm_joint_state", 6),
    ("right_ee_joint_state", 1),
)
ROBODOJO_JOINT_DIM = sum(dim for _, dim in ROBODOJO_JOINT_LAYOUT)
# RoboDojo control rate: one action is interpolated over 10 physics steps of 4 ms.
ROBODOJO_CONTROL_HZ = 25


def _camera_color(env_obs: Dict[str, Any], cam: str) -> np.ndarray:
    vision = env_obs.get("vision") or {}
    if cam not in vision or "color" not in vision[cam]:
        raise KeyError(f"RoboDojo observation is missing vision.{cam}.color")
    image = np.ascontiguousarray(vision[cam]["color"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"vision.{cam}.color must be HxWx3, got shape {image.shape}")
    return image


def pack_joint_state(state: Dict[str, Any]) -> np.ndarray:
    """Pack ``obs['state']`` into the 14-D dual-arm joint vector (float64, as sent)."""
    parts = []
    for key, dim in ROBODOJO_JOINT_LAYOUT:
        if key not in state:
            raise KeyError(f"RoboDojo observation is missing state.{key}")
        value = np.asarray(state[key], dtype=np.float64).reshape(-1)
        if value.size != dim:
            raise ValueError(f"state.{key} must have {dim} values, got {value.size}")
        parts.append(value)
    return np.concatenate(parts)


def joint14_to_action_dict(joint14: Sequence[float]) -> Dict[str, List[float]]:
    """Split a 14-D joint command into RoboDojo's joint-control action dict."""
    vector = np.asarray(joint14, dtype=np.float64).reshape(-1)
    if vector.size != ROBODOJO_JOINT_DIM:
        raise ValueError(f"joint action must have {ROBODOJO_JOINT_DIM} values, got {vector.size}")
    action: Dict[str, List[float]] = {}
    offset = 0
    for key, dim in ROBODOJO_JOINT_LAYOUT:
        action[key] = vector[offset:offset + dim].tolist()
        offset += dim
    return action


class RoboDojoAdapter(BaseBenchmarkAdapter):
    """Convert RoboDojo observations and joint actions to/from the canonical format."""

    action_space = "joint14"

    def __init__(
        self,
        task_name: str = "robodojo_task",
        env_cfg_type: str = "arx_x5",
        seed: int = 0,
        action_channel_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Keep the task identity and the model-output channels that form the 14-D joint action.

        ``action_channel_ids`` picks, in order, the 14 entries of the raw model
        action that map to the joint layout above; ``None`` means the model
        already emits the 14-D joint vector.
        """
        self.task_name = task_name
        self.env_cfg_type = env_cfg_type
        self.seed = int(seed)
        self.action_channel_ids = None if action_channel_ids is None else [int(i) for i in action_channel_ids]
        if self.action_channel_ids is not None and len(self.action_channel_ids) != ROBODOJO_JOINT_DIM:
            raise ValueError(
                f"action_channel_ids must list {ROBODOJO_JOINT_DIM} channels, got {len(self.action_channel_ids)}"
            )

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run obs_to_canonical."""
        head, left, right = (_camera_color(env_obs, cam) for cam in ROBODOJO_CAMERAS)
        joint = pack_joint_state(env_obs.get("state") or {})
        return {
            "instruction": str(context["instruction"]),
            "images": {
                "primary": head,
                "head": head,
                "left": left,
                "right": right,
                "wrist": None,
            },
            "state": {
                "eef_pos": None,
                "eef_rot_axis_angle": None,
                "gripper": None,
                "joint": joint.tolist(),
                "frame": "base",
                "units": {"joint": "rad", "gripper": "normalized_0_1"},
            },
            "state_raw": {"joint": joint},
            "meta": {
                "benchmark": "robodojo",
                "robot_setup": self.env_cfg_type,
                "control_hz": ROBODOJO_CONTROL_HZ,
                "episode_id": str(context.get("episode_id", "default")),
                "episode_step": int(context.get("episode_step", 0)),
                "runtime": "sim",
                "bimanual": True,
                "task_name": self.task_name,
                "seed": self.seed,
            },
        }

    def action_from_canonical(self, canonical_action: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, List[float]]:
        """Map one raw model action (1-D) to a RoboDojo joint action dict."""
        raw = np.asarray(canonical_action, dtype=np.float64).reshape(-1)
        if self.action_channel_ids is not None:
            if max(self.action_channel_ids) >= raw.size:
                raise ValueError(
                    f"action_channel_ids {self.action_channel_ids} exceed the model action size {raw.size}"
                )
            raw = raw[self.action_channel_ids]
        return joint14_to_action_dict(raw)

    def get_eval_context(self) -> Dict[str, Any]:
        """Return benchmark metadata."""
        return {
            "benchmark": "robodojo",
            "task_name": self.task_name,
            "env_cfg_type": self.env_cfg_type,
            "seed": self.seed,
            "control_hz": ROBODOJO_CONTROL_HZ,
        }
