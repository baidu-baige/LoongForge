# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab (NVIDIA Isaac Lab / DROID) benchmark adapter.

RoboLab exposes ``observation.image_obs.<camera>`` (batched uint8 HWC) and
``observation.proprio_obs.{arm_joint_pos, gripper_pos}``. The robot is a
7-DoF Franka with a Robotiq 2F-85; actions are absolute joint targets
(7 arm joints + 1 gripper in [0, 1], 0 = open).

Cameras are named after their prims: ``wrist_cam``, ``over_shoulder_left_camera``,
``over_shoulder_right_camera`` and ``head_camera``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.adapters.base import BaseBenchmarkAdapter

ROBOLAB_ARM_JOINTS = 7
ROBOLAB_ACTION_DIM = ROBOLAB_ARM_JOINTS + 1  # + gripper
ROBOLAB_DEFAULT_CONTROL_HZ = 15  # dt=1/120 s, decimation=8


def _to_numpy(value: Any, env_id: int = 0) -> np.ndarray:
    """Convert a batched tensor/array observation to NumPy, selecting one env."""
    if hasattr(value, "detach"):
        return value[env_id].detach().cpu().numpy()
    array = np.asarray(value)
    return array[env_id] if array.ndim > 0 else array


def _camera(env_obs: Dict[str, Any], name: str, env_id: int) -> Optional[np.ndarray]:
    """Return one env's uint8 HWC frame from ``image_obs``."""
    group = env_obs.get("image_obs") or {}
    value = group.get(name)
    if value is None:
        return None
    image = _to_numpy(value, env_id)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _proprio(env_obs: Dict[str, Any], name: str, env_id: int) -> Optional[np.ndarray]:
    """Return one env's flat float32 proprio term from ``proprio_obs``."""
    group = env_obs.get("proprio_obs") or {}
    value = group.get(name)
    if value is None:
        return None
    return np.asarray(_to_numpy(value, env_id), dtype=np.float32).reshape(-1)


class RobolabAdapter(BaseBenchmarkAdapter):
    """RoboLab obs/action adapter."""

    # Consumed by ``resolve_action_decoder_key`` and view packing.
    action_space: str = "joint_pos"
    default_fps: int = ROBOLAB_DEFAULT_CONTROL_HZ
    cameras: tuple = ("wrist", "left", "right", "head")

    def __init__(
        self,
        task_name: str = "",
        control_hz: int = ROBOLAB_DEFAULT_CONTROL_HZ,
        env_id: int = 0,
        **kwargs: Any,
    ) -> None:
        self.task_name = task_name
        self.control_hz = control_hz
        self.env_id = env_id

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one RoboLab observation into the canonical dict."""
        env_id = int(context.get("env_id", self.env_id))
        wrist = _camera(env_obs, "wrist_cam", env_id)
        left = _camera(env_obs, "over_shoulder_left_camera", env_id)
        right = _camera(env_obs, "over_shoulder_right_camera", env_id)
        head = _camera(env_obs, "head_camera", env_id)

        arm_joint_pos = _proprio(env_obs, "arm_joint_pos", env_id)
        gripper_pos = _proprio(env_obs, "gripper_pos", env_id)
        if arm_joint_pos is None or gripper_pos is None:
            raise KeyError(
                "RoboLab observation is missing proprio_obs.arm_joint_pos / proprio_obs.gripper_pos; "
                f"available groups: {sorted(env_obs.keys())}"
            )
        # DROID state/action layout: 7 arm joints followed by the gripper.
        joint = np.concatenate([arm_joint_pos[:ROBOLAB_ARM_JOINTS], gripper_pos[:1]]).astype(np.float32)

        return {
            "instruction": str(context.get("instruction", "")),
            "images": {
                # The wrist view is the only robot-mounted camera, so it is the
                # primary view for single-view models. Cosmos3 picks wrist/left/
                # right explicitly.
                "primary": wrist if wrist is not None else head,
                "wrist": wrist,
                "left": left,
                "right": right,
                "head": head,
            },
            "state": {
                "joint": arm_joint_pos.tolist(),
                "gripper": float(gripper_pos[0]),
                "frame": "base",
                "units": {"joint": "rad", "gripper": "normalized_0_1"},
            },
            "state_raw": {
                "joint": joint,
                "arm_joint_pos": arm_joint_pos,
                "gripper": gripper_pos,
            },
            "meta": {
                "benchmark": "robolab",
                "robot_setup": "droid_panda_jointpos",
                "control_hz": self.control_hz,
                "episode_id": str(context.get("episode_id", "default")),
                "episode_step": int(context.get("episode_step", 0)),
                "runtime": "sim",
                "bimanual": False,
                "chunk_policy": "full",
                "realtime_deadline_ms": None,
            },
        }

    def action_from_canonical(
        self, canonical_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Convert a canonical 8D action into ``env.step``'s joint target."""
        flat = np.asarray(canonical_action["actions"], dtype=np.float32).reshape(-1)
        if flat.size != ROBOLAB_ACTION_DIM:
            raise ValueError(
                f"RoboLab expects {ROBOLAB_ACTION_DIM}D joint actions "
                f"(7 arm joints + 1 gripper), got {flat.size}D"
            )
        # Own the buffer: the incoming array is a read-only view of the RPC
        # payload, and Isaac Lab writes into the action tensor it is handed.
        return np.array(flat, dtype=np.float32, copy=True)

    def get_eval_context(self) -> Dict[str, Any]:
        """Return benchmark metadata for reporting and diagnostics."""
        return {
            "benchmark": "robolab",
            "robot_setup": "droid_panda_jointpos",
            "control_hz": self.control_hz,
            "max_steps": {},
            "action_scale": {"joint": 1.0, "gripper": 1.0},
            "bimanual": False,
            "has_state_fields": ["joint", "arm_joint_pos", "gripper"],
            "episodes_per_task": 1,
            "runtime": "sim",
            "success_oracle_type": "termination_manager",
        }
