# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin benchmark adapter.

The adapter intentionally has no dependency on the third-party RoboTwin repo. It
only converts RoboTwin-native observations/actions into the canonical protocol so
that official RoboTwin runners can call the standalone policy server.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.adapters.base import BaseBenchmarkAdapter

ROBOTWIN_ACTION_DIM = 14
ROBOTWIN_DEFAULT_MAX_STEPS = 400
ROBOTWIN_ACTION_REORDER = np.asarray([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.int64)
SUPPORTED_ACTION_MODES = {"abs", "delta", "rel"}


def _as_flat_float_array(value: Any, name: str) -> np.ndarray:
    """Run _as_flat_float_array."""
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != ROBOTWIN_ACTION_DIM:
        raise ValueError(f"RoboTwin {name} must be {ROBOTWIN_ACTION_DIM}D, got shape {array.shape}")
    return array


class RoboTwinAdapter(BaseBenchmarkAdapter):
    """Provide RoboTwinAdapter behavior."""

    def __init__(
        self,
        task_name: str = "robotwin_task",
        robot_setup: str = "bimanual_dual_arm",
        control_hz: int = 10,
        max_steps: int = ROBOTWIN_DEFAULT_MAX_STEPS,
        episodes_per_task: int = 1,
        action_mode: str = "abs",
        reorder_action: bool = True,
    ) -> None:
        """Run __init__."""
        if action_mode not in SUPPORTED_ACTION_MODES:
            raise ValueError(f"action_mode must be one of {sorted(SUPPORTED_ACTION_MODES)}, got {action_mode!r}")
        self.task_name = task_name
        self.robot_setup = robot_setup
        self.control_hz = control_hz
        self.max_steps = max_steps
        self.episodes_per_task = episodes_per_task
        self.action_mode = action_mode
        self.reorder_action = reorder_action

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run obs_to_canonical."""
        observation = env_obs["observation"]
        head = np.ascontiguousarray(observation["head_camera"]["rgb"])
        left = np.ascontiguousarray(observation["left_camera"]["rgb"])
        right = np.ascontiguousarray(observation["right_camera"]["rgb"])
        joint = _as_flat_float_array(env_obs["joint_action"]["vector"], "joint state")

        return {
            "instruction": str(context["instruction"]),
            "images": {
                "primary": head,
                "wrist": None,
                "left": left,
                "right": right,
                "head": head,
            },
            "state": {
                "eef_pos": None,
                "eef_rot_axis_angle": None,
                "gripper": None,
                "joint": joint.tolist(),
                "frame": "base",
                "units": {"joint": "normalized"},
            },
            "model_state": joint.copy(),
            "meta": {
                "benchmark": "robotwin",
                "robot_setup": self.robot_setup,
                "control_hz": self.control_hz,
                "episode_id": str(context.get("episode_id", "default")),
                "episode_step": int(context.get("episode_step", 0)),
                "runtime": "sim",
                "bimanual": True,
                "chunk_policy": "full",
                "realtime_deadline_ms": None,
                "task_name": self.task_name,
                "action_mode": self.action_mode,
            },
        }

    def action_from_canonical(self, canonical_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Run action_from_canonical."""
        if "actions" in canonical_action:
            action = _as_flat_float_array(canonical_action["actions"], "action")
        elif "left" in canonical_action and "right" in canonical_action:
            action = self._action_from_bimanual_fields(canonical_action)
        else:
            raise ValueError("Canonical RoboTwin action must contain `actions` or bimanual `left`/`right` fields")

        action_context = context or {}
        mode = str(action_context.get("action_mode", self.action_mode))
        if mode not in SUPPORTED_ACTION_MODES:
            raise ValueError(f"action_mode must be one of {sorted(SUPPORTED_ACTION_MODES)}, got {mode!r}")

        if mode in {"delta", "rel"}:
            if "current_joint" not in action_context:
                raise ValueError("RoboTwin delta/rel action conversion requires context['current_joint']")
            action = _as_flat_float_array(action_context["current_joint"], "current_joint") + action

        if bool(action_context.get("reorder_action", self.reorder_action)):
            action = action[ROBOTWIN_ACTION_REORDER]
        return action.astype(np.float32)

    @staticmethod
    def _action_from_bimanual_fields(canonical_action: Dict[str, Any]) -> np.ndarray:
        """Run _action_from_bimanual_fields."""

        def arm_to_7d(arm: Dict[str, Any], name: str) -> np.ndarray:
            """Run arm_to_7d."""
            world_vector = np.asarray(arm["world_vector"], dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(arm["rotation_delta"], dtype=np.float32).reshape(-1)
            gripper = np.asarray([arm["gripper"]], dtype=np.float32)
            if world_vector.size != 3 or rotation_delta.size != 3:
                raise ValueError(
                    f"Invalid RoboTwin {name} arm action shape: world={world_vector.shape}, rot={rotation_delta.shape}"
                )
            return np.concatenate([world_vector, rotation_delta, gripper], axis=0)

        left = arm_to_7d(canonical_action["left"], "left")
        right = arm_to_7d(canonical_action["right"], "right")
        return np.concatenate([left, right], axis=0).astype(np.float32)

    def get_eval_context(self) -> Dict[str, Any]:
        """Run get_eval_context."""
        return {
            "benchmark": "robotwin",
            "robot_setup": self.robot_setup,
            "control_hz": self.control_hz,
            "max_steps": {self.task_name: self.max_steps},
            "action_scale": {
                "pos_scale": 1.0,
                "rot_scale": 1.0,
                "gripper_scale": 1.0,
                "gripper_bias": 0.0,
                "left": None,
                "right": None,
            },
            "bimanual": True,
            "has_state_fields": ["joint"],
            "episodes_per_task": self.episodes_per_task,
            "runtime": "sim",
            "success_oracle_type": "script",
            "task_name": self.task_name,
            "action_mode": self.action_mode,
            "action_dim": ROBOTWIN_ACTION_DIM,
            "action_reorder": ROBOTWIN_ACTION_REORDER.tolist(),
        }
