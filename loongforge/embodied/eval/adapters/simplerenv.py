# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""SimplerEnv benchmark adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.adapters.base import BaseBenchmarkAdapter

SIMPLERENV_DEFAULT_MAX_STEPS = 120

TASK_TO_ENV_NAME = {
    "google_robot_pick_coke_can": "GraspSingleOpenedCokeCanInScene-v0",
    "google_robot_pick_horizontal_coke_can": "GraspSingleOpenedCokeCanInScene-v0",
    "google_robot_pick_vertical_coke_can": "GraspSingleOpenedCokeCanInScene-v0",
    "google_robot_pick_standing_coke_can": "GraspSingleOpenedCokeCanInScene-v0",
    "google_robot_pick_object": "GraspSingleRandomObjectInScene-v0",
    "google_robot_move_near": "MoveNearGoogleBakedTexInScene-v1",
    "google_robot_move_near_v0": "MoveNearGoogleBakedTexInScene-v0",
    "google_robot_move_near_v1": "MoveNearGoogleBakedTexInScene-v1",
    "google_robot_open_drawer": "OpenDrawerCustomInScene-v0",
    "google_robot_open_top_drawer": "OpenTopDrawerCustomInScene-v0",
    "google_robot_open_middle_drawer": "OpenMiddleDrawerCustomInScene-v0",
    "google_robot_open_bottom_drawer": "OpenBottomDrawerCustomInScene-v0",
    "google_robot_close_drawer": "CloseDrawerCustomInScene-v0",
    "google_robot_close_top_drawer": "CloseTopDrawerCustomInScene-v0",
    "google_robot_close_middle_drawer": "CloseMiddleDrawerCustomInScene-v0",
    "google_robot_close_bottom_drawer": "CloseBottomDrawerCustomInScene-v0",
    "google_robot_place_in_closed_drawer": "PlaceIntoClosedDrawerCustomInScene-v0",
    "google_robot_place_in_closed_top_drawer": "PlaceIntoClosedTopDrawerCustomInScene-v0",
    "google_robot_place_in_closed_middle_drawer": "PlaceIntoClosedMiddleDrawerCustomInScene-v0",
    "google_robot_place_in_closed_bottom_drawer": "PlaceIntoClosedBottomDrawerCustomInScene-v0",
    "google_robot_place_apple_in_closed_top_drawer": "PlaceIntoClosedTopDrawerCustomInScene-v0",
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v0",
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v0",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v0",
    # Registered by the official GR00T env list too, but upstream marks it
    # incomplete (TODO in both simpler_env/__init__.py and put_on_in_scene.py):
    # the target is an invisible dummy plane and xy_configs has a single grid
    # point, so object init is effectively fixed across episodes.
    "widowx_put_eggplant_in_sink": "PutEggplantInSinkScene-v0",
    # WidowX small-drawer tasks: registered only by NVIDIA's pinned SimplerEnv
    # fork, not by upstream simpler-env/SimplerEnv.
    "widowx_open_drawer": "OpenSmallDrawerCustomInScene-v0",
    "widowx_close_drawer": "CloseSmallDrawerCustomInScene-v0",
}


def _infer_policy_setup(robot_setup: str) -> str:
    """Run _infer_policy_setup."""
    if robot_setup.startswith("google_robot"):
        return "google_robot"
    if robot_setup.startswith("widowx"):
        return "widowx_bridge"
    raise ValueError(f"Unsupported SimplerEnv robot setup: {robot_setup!r}")


def _default_camera_name(robot_setup: str) -> str:
    """Run _default_camera_name."""
    policy_setup = _infer_policy_setup(robot_setup)
    if policy_setup == "google_robot":
        return "overhead_camera"
    return "3rd_view_camera"


def _extract_agent_state(env_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Run _extract_agent_state."""
    agent = env_obs.get("agent", {})
    if isinstance(agent, dict):
        return agent
    return {}


class SimplerEnvAdapter(BaseBenchmarkAdapter):
    """SimplerEnv obs/action adapter.

    Refactor note: this adapter no longer builds any proprio itself. The
    WidowX initial-state pose fields (``base_pose`` / ``tcp_pose``) are
    exposed under ``canonical["state_raw"]`` and the per-model PayloadBuilder
    (currently ``XVLAPayloadBuilder``'s ``ee6d_widowx`` encoding) is
    responsible for building the closed-loop-backfilled 20D proprio.
    """

    # Capability declarations for orchestrator M×N matching. The active
    # decoder is chosen by ``payload_builder.action_encoding × action_space``
    # — for xvla ee6d × ``simpler_abs_euler`` the composed registry key is
    # ``ee6d_to_simpler_abs_euler``.
    action_space: str = "simpler_abs_euler"
    default_fps: int = 5
    cameras: tuple = ("primary",)

    def __init__(
        self,
        task_name: str = "widowx_put_eggplant_in_basket",
        robot_setup: str = "widowx_sink_camera_setup",
        control_hz: int = 5,
        max_steps: int = SIMPLERENV_DEFAULT_MAX_STEPS,
        camera_name: Optional[str] = None,
        action_scale: float = 1.0,
        rotation_mode: str = "euler",
    ) -> None:
        """Run __init__."""
        if task_name not in TASK_TO_ENV_NAME:
            raise ValueError(f"Unknown SimplerEnv task {task_name!r}; choose one of {sorted(TASK_TO_ENV_NAME)}")
        self.task_name = task_name
        self.env_name = TASK_TO_ENV_NAME[task_name]
        self.robot_setup = robot_setup
        self.policy_setup = _infer_policy_setup(robot_setup)
        self.control_hz = control_hz
        self.max_steps = max_steps
        if rotation_mode not in {"euler", "axis_angle"}:
            raise ValueError(f"rotation_mode must be 'euler' or 'axis_angle', got {rotation_mode!r}")
        self.camera_name = camera_name or _default_camera_name(robot_setup)
        self.action_scale = action_scale
        self.rotation_mode = rotation_mode

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a SimplerEnv obs into a benchmark-neutral canonical dict."""
        image = np.ascontiguousarray(env_obs["image"][self.camera_name]["rgb"])
        agent_state = _extract_agent_state(env_obs)
        qpos = np.asarray(agent_state.get("qpos", []), dtype=np.float32).reshape(-1)
        qvel = np.asarray(agent_state.get("qvel", []), dtype=np.float32).reshape(-1)

        # base_pose / tcp_pose are needed by ``XVLAPayloadBuilder``'s
        # ``ee6d_widowx`` encoding to compute the initial 20D proprio; expose
        # them under ``state_raw`` (raw, no encoding) so the per-model builder
        # can consume them without importing SAPIEN here in the adapter.
        base_pose = agent_state.get("base_pose") if isinstance(agent_state, dict) else None
        tcp_pose = None
        extra = env_obs.get("extra") if isinstance(env_obs, dict) else None
        if isinstance(extra, dict):
            tcp_pose = extra.get("tcp_pose")
        # Official GR00T WidowX wrapper builds proprio from ``agent.eef_pos``
        # ([x,y,z, quat_wxyz(4), gripper]); expose it raw for the per-model
        # PayloadBuilder (GR00T oxe_widowx) to encode.
        eef_pos = agent_state.get("eef_pos") if isinstance(agent_state, dict) else None

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
            "state_raw": {
                "base_pose": np.asarray(base_pose, dtype=np.float32) if base_pose is not None else None,
                "tcp_pose": np.asarray(tcp_pose, dtype=np.float32) if tcp_pose is not None else None,
                "eef_pos": np.asarray(eef_pos, dtype=np.float32) if eef_pos is not None else None,
                "joint": qpos if qpos.size else None,
            },
            "meta": {
                "benchmark": "simplerenv",
                "robot_setup": self.robot_setup,
                "control_hz": self.control_hz,
                "episode_id": str(context.get("episode_id", "default")),
                "episode_step": int(context.get("episode_step", 0)),
                "runtime": "sim",
                "bimanual": False,
                "chunk_policy": "full",
                "realtime_deadline_ms": None,
                "task_name": self.task_name,
                "env_name": self.env_name,
                "camera_name": self.camera_name,
            },
        }

    def action_from_canonical(self, canonical_action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Run action_from_canonical."""
        if "world_vector" in canonical_action:
            world_vector = np.asarray(canonical_action["world_vector"], dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(
                canonical_action.get("rot_axangle", canonical_action.get("rotation_delta")), dtype=np.float32
            ).reshape(-1)
            gripper_value = canonical_action["gripper"]
        elif "actions" in canonical_action:
            flat = np.asarray(canonical_action["actions"], dtype=np.float32).reshape(-1)
            world_vector = flat[:3]
            rotation_delta = flat[3:6]
            gripper_value = flat[6]
        else:
            raise ValueError("Canonical action must contain either structured fields or `actions` flat array")

        if world_vector.size != 3 or rotation_delta.size != 3:
            raise ValueError(f"Invalid SimplerEnv action shape: world={world_vector.shape}, rot={rotation_delta.shape}")

        gripper = self._convert_gripper(gripper_value)
        rotation_axis_angle = self._convert_rotation(rotation_delta)
        return np.concatenate(
            [world_vector * self.action_scale, rotation_axis_angle * self.action_scale, gripper], axis=0
        ).astype(np.float32)

    def _convert_rotation(self, rotation_delta: np.ndarray) -> np.ndarray:
        """Run _convert_rotation."""
        if self.rotation_mode == "axis_angle":
            return rotation_delta.astype(np.float32)
        # Lazy import: ``transforms3d`` is only available in the simplerenv
        # env; keeping the top-level module import-clean lets loongforge-env
        # unit tests exercise the adapter (only ``action_from_canonical``
        # actually uses this helper).
        from transforms3d.euler import euler2axangle

        roll, pitch, yaw = rotation_delta.astype(np.float64)
        axis, angle = euler2axangle(roll, pitch, yaw)
        return np.asarray(axis * angle, dtype=np.float32)

    def _convert_gripper(self, gripper_value: Any) -> np.ndarray:
        """Run _convert_gripper."""
        value = float(np.asarray(gripper_value, dtype=np.float32).reshape(-1)[0])
        if self.policy_setup == "google_robot":
            return np.asarray([value], dtype=np.float32)
        return np.asarray([2.0 * (value > 0.5) - 1.0], dtype=np.float32)

    def get_eval_context(self) -> Dict[str, Any]:
        """Run get_eval_context."""
        return {
            "benchmark": "simplerenv",
            "robot_setup": self.robot_setup,
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
            "success_oracle_type": "info_flag",
            "task_name": self.task_name,
            "env_name": self.env_name,
            "policy_setup": self.policy_setup,
            "camera_name": self.camera_name,
            "rotation_mode": self.rotation_mode,
        }
