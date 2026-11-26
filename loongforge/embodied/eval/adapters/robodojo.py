# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo (Isaac Sim, dual-arm ARX X5) benchmark adapter.

RoboDojo's official sim client hands the policy an observation dict with
``vision.<cam>.color`` (HWC uint8 RGB), ``state.<arm|ee>`` vectors and the
episode ``instruction``. It executes either joint action dicts keyed by
``{left,right}_arm_joint_state`` (6) and ``{left,right}_ee_joint_state`` (1), or
ee action dicts keyed by ``{left,right}_ee_pose`` (7: xyz + wxyz quaternion) and
``{left,right}_ee_joint_state`` (1).

The 14-D joint vector used throughout this adapter follows XPolicyLab's
``pack_robot_state`` dual-arm order: left arm 6, left gripper 1, right arm 6,
right gripper 1. The 20-D ``ee20`` action is X-VLA's ee6d layout: per arm xyz 3,
interleaved 6-D rotation 6, gripper 1 (left then right).
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
ROBODOJO_ACTION_SPACES = ("joint14", "ee20")
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


def ee_state(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Measured ee poses (xyz + wxyz quaternion) and gripper joints, float32 as XPolicyLab reads them."""
    out = {}
    for side in ("left", "right"):
        pose = np.asarray(state[f"{side}_ee_pose"], dtype=np.float32).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"state.{side}_ee_pose must have 7 values, got {pose.size}")
        out[f"{side}_ee_pose"] = pose
        out[f"{side}_gripper"] = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)
    return out


def rotate6d_to_quat(vec6: np.ndarray) -> np.ndarray:
    """Interleaved 6-D rotation -> wxyz quaternion, as the official RoboDojo X_VLA adapter computes it.

    The 6 values are the first two columns of the rotation matrix flattened row-wise, so the
    columns are ``vec6[0::2]`` and ``vec6[1::2]``; Gram-Schmidt, then matrix -> quaternion.
    """
    vec6 = np.asarray(vec6, dtype=np.float32)
    a1 = vec6[..., 0:5:2]
    a2 = vec6[..., 1:6:2]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    rot = np.stack((b1, b2, b3), axis=-1)
    m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]
    trace = m00 + m11 + m22
    quat = np.empty(rot.shape[:-2] + (4,), dtype=np.float32)  # xyzw while building
    pos = trace > 0
    s = np.sqrt(np.maximum(trace[pos] + 1.0, 1e-8)) * 2
    quat[pos, 3] = 0.25 * s
    quat[pos, 0] = (m21[pos] - m12[pos]) / s
    quat[pos, 1] = (m02[pos] - m20[pos]) / s
    quat[pos, 2] = (m10[pos] - m01[pos]) / s
    c1 = (~pos) & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.maximum(1.0 + m00[c1] - m11[c1] - m22[c1], 1e-8)) * 2
    quat[c1, 3] = (m21[c1] - m12[c1]) / s
    quat[c1, 0] = 0.25 * s
    quat[c1, 1] = (m01[c1] + m10[c1]) / s
    quat[c1, 2] = (m02[c1] + m20[c1]) / s
    c2 = (~pos) & (~c1) & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11[c2] - m00[c2] - m22[c2], 1e-8)) * 2
    quat[c2, 3] = (m02[c2] - m20[c2]) / s
    quat[c2, 0] = (m01[c2] + m10[c2]) / s
    quat[c2, 1] = 0.25 * s
    quat[c2, 2] = (m12[c2] + m21[c2]) / s
    c3 = (~pos) & (~c1) & (~c2)
    s = np.sqrt(np.maximum(1.0 + m22[c3] - m00[c3] - m11[c3], 1e-8)) * 2
    quat[c3, 3] = (m10[c3] - m01[c3]) / s
    quat[c3, 0] = (m02[c3] + m20[c3]) / s
    quat[c3, 1] = (m12[c3] + m21[c3]) / s
    quat[c3, 2] = 0.25 * s
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    return np.concatenate([quat[..., 3:4], quat[..., :3]], axis=-1).astype(np.float32)


def ee20_to_action_dict(ee20: Sequence[float]) -> Dict[str, List[float]]:
    """Split a 20-D ee6d command into RoboDojo's ee-control action dict (grippers passed through)."""
    vector = np.asarray(ee20, dtype=np.float32).reshape(-1)
    if vector.size != 20:
        raise ValueError(f"ee20 action must have 20 values, got {vector.size}")
    action: Dict[str, List[float]] = {}
    for side, base in (("left", 0), ("right", 10)):
        quat = rotate6d_to_quat(vector[base + 3:base + 9])
        action[f"{side}_ee_pose"] = np.concatenate([vector[base:base + 3], quat]).astype(np.float32).tolist()
        action[f"{side}_ee_joint_state"] = [float(vector[base + 9])]
    return action


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
    """Convert RoboDojo observations and joint / ee actions to/from the canonical format."""

    def __init__(
        self,
        task_name: str = "robodojo_task",
        env_cfg_type: str = "arx_x5",
        seed: int = 0,
        action_channel_ids: Optional[Sequence[int]] = None,
        action_space: str = "joint14",
    ) -> None:
        """Keep the task identity, the env action space and the model-output channels that form it.

        ``action_space`` is ``joint14`` (joint control) or ``ee20`` (X-VLA ee6d -> ee control).
        ``action_channel_ids`` picks, in order, the entries of the raw model action that form
        that action (14 for ``joint14``, 20 for ``ee20``); ``None`` means the model already emits it.
        """
        if action_space not in ROBODOJO_ACTION_SPACES:
            raise ValueError(f"action_space must be one of {ROBODOJO_ACTION_SPACES}, got {action_space!r}")
        self.task_name = task_name
        self.env_cfg_type = env_cfg_type
        self.seed = int(seed)
        self.action_space = action_space
        expected = ROBODOJO_JOINT_DIM if action_space == "joint14" else 20
        self.action_channel_ids = None if action_channel_ids is None else [int(i) for i in action_channel_ids]
        if self.action_channel_ids is not None and len(self.action_channel_ids) != expected:
            raise ValueError(
                f"action_channel_ids must list {expected} channels, got {len(self.action_channel_ids)}"
            )

    def obs_to_canonical(self, env_obs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run obs_to_canonical."""
        head, left, right = (_camera_color(env_obs, cam) for cam in ROBODOJO_CAMERAS)
        state = env_obs.get("state") or {}
        joint = pack_joint_state(state)
        state_raw: Dict[str, Any] = {"joint": joint}
        if self.action_space == "ee20":
            state_raw.update(ee_state(state))
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
            "state_raw": state_raw,
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
        """Map one raw model action (1-D) to a RoboDojo joint or ee action dict."""
        raw = np.asarray(canonical_action, dtype=np.float64).reshape(-1)
        if self.action_channel_ids is not None:
            if max(self.action_channel_ids) >= raw.size:
                raise ValueError(
                    f"action_channel_ids {self.action_channel_ids} exceed the model action size {raw.size}"
                )
            raw = raw[self.action_channel_ids]
        if self.action_space == "ee20":
            return ee20_to_action_dict(raw)
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
