# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ManiSkill benchmark adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.adapters.base import BaseBenchmarkAdapter

MANISKILL_DEFAULT_MAX_STEPS = 200


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or array-like values to NumPy arrays."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_rgb(image: Any) -> np.ndarray:
    """Normalize a ManiSkill RGB tensor to an HWC NumPy image."""
    rgb = _to_numpy(image)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    return np.ascontiguousarray(rgb)


def _dummy_image() -> np.ndarray:
    """Return a black RGB image for state-only smoke tests."""
    return np.zeros((224, 224, 3), dtype=np.uint8)


def _extract_image(env_obs: Any, camera_name: str, allow_dummy_image: bool = False) -> np.ndarray:
    """Extract an RGB image from ManiSkill observation variants."""
    if not isinstance(env_obs, dict):
        if allow_dummy_image:
            return _dummy_image()
        raise KeyError(f"Could not find RGB image for camera {camera_name!r} in ManiSkill observation")

    if "image" in env_obs:
        image_obs = env_obs["image"]
        if camera_name in image_obs and "rgb" in image_obs[camera_name]:
            return _normalize_rgb(image_obs[camera_name]["rgb"])
        for camera_obs in image_obs.values():
            if isinstance(camera_obs, dict) and "rgb" in camera_obs:
                return _normalize_rgb(camera_obs["rgb"])

    if "sensor_data" in env_obs:
        sensor_data = env_obs["sensor_data"]
        if camera_name in sensor_data and "rgb" in sensor_data[camera_name]:
            return _normalize_rgb(sensor_data[camera_name]["rgb"])
        for camera_obs in sensor_data.values():
            if isinstance(camera_obs, dict) and "rgb" in camera_obs:
                return _normalize_rgb(camera_obs["rgb"])

    if allow_dummy_image:
        return _dummy_image()
    raise KeyError(f"Could not find RGB image for camera {camera_name!r} in ManiSkill observation")


def _extract_agent_state(env_obs: Any) -> Dict[str, Any]:
    """Extract agent state dict from ManiSkill observation variants."""
    if not isinstance(env_obs, dict):
        return {}
    agent = env_obs.get("agent", {})
    return agent if isinstance(agent, dict) else {}


class ManiSkillAdapter(BaseBenchmarkAdapter):
    """Convert ManiSkill observations/actions to the canonical eval schema."""

    def __init__(
        self,
        task_name: str = "PickCube-v1",
        robot_uid: str = "panda",
        control_hz: int = 5,
        max_steps: int = MANISKILL_DEFAULT_MAX_STEPS,
        camera_name: str = "base_camera",
        action_scale: float = 1.0,
        gripper_open_value: float = -1.0,
        gripper_close_value: float = 1.0,
        allow_dummy_image: bool = False,
    ) -> None:
        """Initialize the ManiSkill canonical-schema adapter."""
        self.task_name = task_name
        self.robot_uid = robot_uid
        self.control_hz = control_hz
        self.max_steps = max_steps
        self.camera_name = camera_name
        self.action_scale = action_scale
        self.gripper_open_value = gripper_open_value
        self.gripper_close_value = gripper_close_value
        self.allow_dummy_image = allow_dummy_image

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a ManiSkill observation to the canonical eval payload."""
        image = _extract_image(env_obs, self.camera_name, self.allow_dummy_image)
        agent_state = _extract_agent_state(env_obs)
        qpos = _to_numpy(agent_state.get("qpos", [])).astype(np.float32).reshape(-1)
        qvel = _to_numpy(agent_state.get("qvel", [])).astype(np.float32).reshape(-1)

        # Panda qpos is 9D (7 arm + 2 finger). RLinf/openpi ManiSkill stats use 8D:
        # 7 arm joints + single gripper width (mean of the two finger joints).
        if qpos.size >= 9:
            model_state = np.concatenate([qpos[:7], np.array([float(qpos[7:9].mean())], dtype=np.float32)])
        elif qpos.size:
            model_state = qpos
        else:
            model_state = None

        state: Dict[str, Any] = {
            "eef_pos": None,
            "eef_rot_axis_angle": None,
            "gripper": None,
            "joint": qpos.tolist() if qpos.size else None,
            "frame": "base",
            "units": {"pos": "m", "rot": "rad"},
        }
        if qvel.size:
            state["joint_vel"] = qvel.tolist()

        return {
            "instruction": str(context["instruction"]),
            "images": {
                "primary": image,
                "wrist": None,
                "left": None,
                "right": None,
                "head": None,
            },
            "state": state,
            "model_state": model_state.tolist() if model_state is not None else None,
            "meta": {
                "benchmark": "maniskill",
                "robot_setup": self.robot_uid,
                "control_hz": self.control_hz,
                "episode_id": str(context.get("episode_id", "default")),
                "episode_step": int(context.get("episode_step", 0)),
                "runtime": "sim",
                "bimanual": False,
                "chunk_policy": "full",
                "realtime_deadline_ms": None,
                "task_name": self.task_name,
                "camera_name": self.camera_name,
            },
        }

    def action_from_canonical(self, canonical_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Convert a canonical 7D action to the ManiSkill control action."""
        if "actions" in canonical_action:
            flat = np.asarray(canonical_action["actions"], dtype=np.float32).reshape(-1)
            world_vector = flat[:3]
            rotation_delta = flat[3:6]
            gripper_value = flat[6]
        else:
            world_vector = np.asarray(canonical_action["world_vector"], dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(
                canonical_action.get("rot_axangle", canonical_action.get("rotation_delta")), dtype=np.float32
            ).reshape(-1)
            gripper_value = canonical_action["gripper"]

        if world_vector.size != 3 or rotation_delta.size != 3:
            raise ValueError(f"Invalid ManiSkill action shape: world={world_vector.shape}, rot={rotation_delta.shape}")

        gripper = self._convert_gripper(gripper_value)
        return np.concatenate([world_vector, rotation_delta, gripper], axis=0).astype(np.float32) * self.action_scale

    def _convert_gripper(self, gripper_value: Any) -> np.ndarray:
        """Map a canonical gripper command to ManiSkill's scalar gripper value."""
        value = float(np.asarray(gripper_value, dtype=np.float32).reshape(-1)[0])
        gripper = self.gripper_close_value if value > 0.0 else self.gripper_open_value
        return np.asarray([gripper], dtype=np.float32)

    def get_eval_context(self) -> Dict[str, Any]:
        """Return metadata describing the ManiSkill eval setup."""
        return {
            "benchmark": "maniskill",
            "robot_setup": self.robot_uid,
            "control_hz": self.control_hz,
            "max_steps": {self.task_name: self.max_steps},
            "action_scale": {
                "pos_scale": self.action_scale,
                "rot_scale": self.action_scale,
                "gripper_scale": 1.0,
                "gripper_bias": 0.0,
                "left": None,
                "right": None,
            },
            "bimanual": False,
            "has_state_fields": ["joint"],
            "episodes_per_task": 1,
            "runtime": "sim",
            "success_oracle_type": "info_flag_or_reward",
            "task_name": self.task_name,
            "camera_name": self.camera_name,
        }
