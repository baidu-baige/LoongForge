# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO benchmark adapter.

This adapter is simulator-side and framework-agnostic. It converts between
LIBERO-native observations/actions and the Canonical protocol used by
`vla_eval`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.adapters.base import BaseBenchmarkAdapter

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    """Map model open-gripper scalar to LIBERO gripper command.

    LIBERO eval convention:
    - model value > 0.5 means open
    - LIBERO action gripper is -1 for open, +1 for close
    """
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    value = float(arr[0])
    return np.asarray([1.0 - 2.0 * (value > 0.5)], dtype=np.float32)


class LiberoAdapter(BaseBenchmarkAdapter):
    """LIBERO obs/action adapter.

    The adapter never encodes proprio state — it exposes raw end-effector
    fields under ``canonical["state_raw"]`` and the per-model PayloadBuilder
    is responsible for whatever encoding the model consumes
    (``ee6d`` / ``axis_angle`` / no state / ...).
    """

    # Capability declarations for orchestrator M×N matching. LIBERO expects
    # a 7D delta OSC action (pos + axis_angle + gripper) in canonical form,
    # decoded from whatever ``model.action_encoding`` the model emits.
    action_space: str = "axis_angle"
    default_fps: int = 20
    cameras: tuple = ("primary", "wrist")

    def __init__(
        self,
        suite_name: str = "libero_goal",
        robot_setup: str = "franka",
        control_hz: int = 20,
        episodes_per_task: int = 50,
        resolution: int = LIBERO_ENV_RESOLUTION,
        continuous_gripper: bool = False,
        prefer_controller_ee: bool = True,
    ) -> None:
        """Initialize the LIBERO adapter."""
        if suite_name not in SUITE_MAX_STEPS:
            raise ValueError(f"Unknown LIBERO suite {suite_name!r}; choose one of {sorted(SUITE_MAX_STEPS)}")
        self.suite_name = suite_name
        self.robot_setup = robot_setup
        self.control_hz = control_hz
        self.episodes_per_task = episodes_per_task
        self.resolution = resolution
        self.continuous_gripper = continuous_gripper
        self.prefer_controller_ee = prefer_controller_ee

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a LIBERO obs into a benchmark-neutral canonical dict."""
        primary = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])

        eef_pos = np.asarray(env_obs.get("robot0_eef_pos"), dtype=np.float32).reshape(-1)[:3]
        eef_quat = np.asarray(env_obs.get("robot0_eef_quat"), dtype=np.float32).reshape(-1)[:4]
        gripper_qpos = np.asarray(env_obs.get("robot0_gripper_qpos"), dtype=np.float32).reshape(-1)
        gripper = float(gripper_qpos[0]) if gripper_qpos.size else None

        # Prefer controller's ee_pos / ee_ori_mat if available (matches the
        # original X-VLA LIBERO client, which reads them from the robosuite
        # OSC controller rather than the raw obs).
        ctrl_ee_pos = context.get("ee_pos") if self.prefer_controller_ee else None
        ctrl_ee_ori_mat = context.get("ee_ori_mat") if self.prefer_controller_ee else None
        if ctrl_ee_pos is not None:
            eef_pos = ctrl_ee_pos[:3]

        state_raw = {
            "eef_pos": np.asarray(eef_pos, dtype=np.float32),
            "eef_quat": np.asarray(eef_quat, dtype=np.float32),
            "gripper": gripper,
            # Full 2-DoF finger qpos (Panda: [+finger, -finger]); models needing
            # the native 2D gripper state (e.g. GR00T libero_panda) read this.
            "gripper_qpos": np.asarray(gripper_qpos, dtype=np.float32),
            "ee_ori_mat": (
                np.asarray(ctrl_ee_ori_mat, dtype=np.float32) if ctrl_ee_ori_mat is not None else None
            ),
        }

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
                "eef_quat": eef_quat.tolist(),
                "gripper": gripper,
                "joint": None,
                "frame": "base",
                "units": {"pos": "m", "rot": "rad"},
            },
            "state_raw": state_raw,
            "meta": {
                "benchmark": "libero",
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
            raise ValueError(f"Invalid LIBERO action shape: world={world_vector.shape}, rot={rotation_delta.shape}")

        gripper = (
            np.asarray([float(gripper_value)], dtype=np.float32)
            if self.continuous_gripper
            else binarize_gripper_open(gripper_value)
        )
        return np.concatenate([world_vector, rotation_delta, gripper], axis=0).astype(np.float32).tolist()

    def get_eval_context(self) -> Dict[str, Any]:
        """Run get_eval_context."""
        return {
            "benchmark": "libero",
            "robot_setup": self.robot_setup,
            "control_hz": self.control_hz,
            "max_steps": {self.suite_name: SUITE_MAX_STEPS[self.suite_name]},
            "action_scale": {
                "pos_scale": 1.0,
                "rot_scale": 1.0,
                "gripper_scale": 1.0,
                "gripper_bias": 0.0,
                "left": None,
                "right": None,
            },
            "bimanual": False,
            "has_state_fields": ["eef_pos", "eef_quat", "gripper"],
            "episodes_per_task": self.episodes_per_task,
            "runtime": "sim",
            "success_oracle_type": "info_flag",
            "suite_name": self.suite_name,
            "num_steps_wait": 10,
            "dummy_action": LIBERO_DUMMY_ACTION,
            "resolution": self.resolution,
        }
