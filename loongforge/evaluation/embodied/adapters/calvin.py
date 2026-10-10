# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CALVIN benchmark adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.evaluation.embodied.adapters.base import BaseBenchmarkAdapter

CALVIN_MAX_STEPS_PER_SUBTASK = 360


def binarize_calvin_gripper(gripper_value: np.ndarray | float) -> np.ndarray:
    """Map model gripper scalar to CALVIN's {-1, +1} action convention."""
    value = float(np.asarray(gripper_value, dtype=np.float32).reshape(-1)[0])
    return np.asarray([1.0 if value > 0.0 else -1.0], dtype=np.float32)


class CalvinAdapter(BaseBenchmarkAdapter):
    """Convert between CALVIN observations/actions and canonical eval payloads.

    Proprio encoding is delegated to the per-model PayloadBuilder — the raw
    CALVIN ``robot_obs`` array is exposed under ``canonical["state_raw"]``
    and ``XVLAPayloadBuilder``'s ``ee6d_calvin`` encoding converts it to the
    20D proprio X-VLA expects (with closed-loop backfill across steps).
    """

    # Capability declarations for orchestrator M×N matching. CALVIN's
    # absolute-EE action space is ``pos(3)+quat(4)+gripper(1)`` — the
    # PayloadBuilder's ``action_encoding × adapter.action_space`` compose to
    # ``ee6d_to_calvin_abs`` (registered decoder key).
    action_space: str = "calvin_abs"
    default_fps: int = 30
    cameras: tuple = ("primary", "wrist")

    def __init__(
        self,
        robot_setup: str = "franka",
        control_hz: int = 30,
        max_steps_per_subtask: int = CALVIN_MAX_STEPS_PER_SUBTASK,
        continuous_gripper: bool = False,
    ) -> None:
        """Initialize the CALVIN adapter."""
        self.robot_setup = robot_setup
        self.control_hz = control_hz
        self.max_steps_per_subtask = max_steps_per_subtask
        self.continuous_gripper = continuous_gripper

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a CALVIN obs into a benchmark-neutral canonical dict."""
        rgb_obs = env_obs.get("rgb_obs") or {}
        primary = np.ascontiguousarray(rgb_obs["rgb_static"])
        wrist = np.ascontiguousarray(rgb_obs["rgb_gripper"])
        robot_obs = np.asarray(env_obs.get("robot_obs", []), dtype=np.float32).reshape(-1)

        eef_pos = robot_obs[:3] if robot_obs.size >= 3 else np.zeros(3, dtype=np.float32)
        eef_rot = robot_obs[3:6] if robot_obs.size >= 6 else np.zeros(3, dtype=np.float32)
        gripper = float(robot_obs[6]) if robot_obs.size >= 7 else None
        joint = robot_obs[8:15].tolist() if robot_obs.size >= 15 else None

        return {
            "instruction": str(context["instruction"]),
            "images": {
                "primary": primary,
                "wrist": wrist,
                "left": None,
                "right": None,
                "head": None,
            },
            "state": {
                "eef_pos": eef_pos.tolist(),
                "eef_rot_axis_angle": eef_rot.tolist(),
                "gripper": gripper,
                "joint": joint,
                "frame": "base",
                "units": {"pos": "m", "rot": "rad"},
            },
            # Raw robot_obs vector for per-model PayloadBuilder consumption.
            # X-VLA's ``ee6d_calvin`` encoding reads this array directly.
            "state_raw": {"robot_obs": robot_obs},
            "meta": {
                "benchmark": "calvin",
                "robot_setup": self.robot_setup,
                "control_hz": self.control_hz,
                "episode_id": str(context.get("episode_id", "default")),
                "episode_step": int(context.get("episode_step", 0)),
                "runtime": "sim",
                "bimanual": False,
                "chunk_policy": "full",
                "realtime_deadline_ms": None,
            },
        }

    def action_from_canonical(self, canonical_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Run action_from_canonical."""
        if "world_vector" in canonical_action:
            world_vector = np.asarray(canonical_action["world_vector"], dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(canonical_action["rotation_delta"], dtype=np.float32).reshape(-1)
            gripper_value = canonical_action["gripper"]
        elif "actions" in canonical_action:
            flat = np.asarray(canonical_action["actions"], dtype=np.float32).reshape(-1)
            world_vector = flat[:3]
            rotation_delta = flat[3:6]
            gripper_value = flat[6]
        else:
            raise ValueError("Canonical action must contain either structured fields or `actions` flat array")

        if world_vector.size != 3 or rotation_delta.size != 3:
            raise ValueError(f"Invalid CALVIN action shape: world={world_vector.shape}, rot={rotation_delta.shape}")

        gripper = (
            np.asarray([float(gripper_value)], dtype=np.float32)
            if self.continuous_gripper
            else binarize_calvin_gripper(gripper_value)
        )
        return np.concatenate([world_vector, rotation_delta, gripper], axis=0).astype(np.float32)

    def get_eval_context(self) -> Dict[str, Any]:
        """Run get_eval_context."""
        return {
            "benchmark": "calvin",
            "robot_setup": self.robot_setup,
            "control_hz": self.control_hz,
            "max_steps": {"per_subtask": self.max_steps_per_subtask},
            "action_scale": {
                "pos_scale": 1.0,
                "rot_scale": 1.0,
                "gripper_scale": 1.0,
                "gripper_bias": 0.0,
                "left": None,
                "right": None,
            },
            "bimanual": False,
            "has_state_fields": ["eef_pos", "eef_rot_axis_angle", "gripper", "joint"],
            "runtime": "sim",
            "success_oracle_type": "calvin_task_oracle",
            "sequence_length": 5,
        }
