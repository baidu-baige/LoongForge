# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM PayloadBuilder.

FastWAM consumes a per-replan multi-view observation plus a per-checkpoint
proprio state (LIBERO 8D eef / RoboTwin 14D joint). Image packing, prompt
text, normalization and action denormalization all live model-side in
``FastWAMPolicy.predict_action``; this builder only packs the canonical dict
into the ``predict_action`` kwargs.

Per-benchmark geometry (matched to the checkpoint via ``state_encoding``):
- ``libero_ee8``: 2 views (agentview + wrist), 224 square, horizontal concat.
- ``robotwin_joint14``: 3 views (head + left/right wrist), T-layout pack to
  384x320 (official ``deploy_policy._build_robotwin_image_tensor``), and the
  14D dual-arm joint vector passed through as-is (state == observation
  ``joint_action.vector``; the official client does no joint remapping).

Replan protocol (official FastWAM rollout): the model emits a 32-step chunk
and ``replan_steps`` are executed open-loop before the next call (LIBERO 10 /
RoboTwin 24), so the default chunk-cached behaviour is kept
(``disable_action_cache`` stays False) and the horizon truncation is applied
server-side by the factory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder
from loongforge.embodied.eval.payload_builders.registry import register_payload_builder

# Instruction wrapper used at FastWAM training time (fastwam.datasets.lerobot.
# robot_video_dataset.DEFAULT_PROMPT); eval must send the same distribution.
FASTWAM_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _pack_fastwam_images(
    images_by_cam: Dict[str, Optional[np.ndarray]], state_encoding: str
) -> List[np.ndarray]:
    """Return the view list FastWAM expects for the checkpoint's geometry.

    - ``libero_ee8``: ``[agentview, wrist]`` — the model concatenates the two
      views horizontally (agentview left, wrist right) inside
      ``predict_action``; the order must match that layout and the
      training-time camera pair (image | wrist_image).
    - ``robotwin_joint14``: ``[head, left_wrist, right_wrist]`` — packed into
      the official T layout (head on top, [left|right] below) model-side.
    """
    if state_encoding == "robotwin_joint14":
        head = images_by_cam.get("head")
        if head is None:
            head = images_by_cam.get("primary")
        if head is None:
            raise ValueError("robotwin images must contain 'head' (head_camera)")
        left = images_by_cam.get("left")
        right = images_by_cam.get("right")
        if left is None or right is None:
            raise ValueError("robotwin images must contain 'left' and 'right' wrist cams")
        return [np.asarray(head), np.asarray(left), np.asarray(right)]

    agent = images_by_cam.get("primary")
    if agent is None:
        agent = images_by_cam.get("head")
    if agent is None:
        raise ValueError("images_by_cam must contain 'primary' or 'head' (agentview)")
    wrist = images_by_cam.get("wrist")
    if wrist is None:
        wrist = images_by_cam.get("right")
    if wrist is None:
        raise ValueError("images_by_cam must contain 'wrist' for FastWAM LIBERO eval")
    return [np.asarray(agent), np.asarray(wrist)]


def _quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """(x,y,z,w) quaternion -> rotation vector.

    Exact port of the official eval's ``quat2axisangle`` (copied from
    robosuite transform_utils): robosuite/LIBERO ``robot0_eef_quat`` is
    **(x, y, z, w)** ordered, and NO hemisphere canonicalization is applied
    (w < 0 yields angle > pi with flipped axis — that representation is part
    of the training-data distribution and must be preserved).
    """
    import math

    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)[:4].copy()
    q[3] = min(max(q[3], -1.0), 1.0)
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


def encode_libero_state_ee8(state_raw: Dict[str, Any]) -> np.ndarray:
    """Canonical LIBERO proprio -> FastWAM 8D state (pos3 + axis-angle3 + gripper_qpos2).

    Matches the official eval state extraction (euler rpy -> quaternion ->
    axis-angle) and the checkpoint's 8D ``observation.state`` layout that the
    ``*_dataset_stats.json`` global_min/max were computed over.
    """
    eef_pos = np.asarray(state_raw.get("eef_pos"), dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(state_raw.get("eef_quat"), dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(state_raw.get("gripper_qpos"), dtype=np.float32).reshape(-1)
    if eef_pos.size != 3 or eef_quat.size != 4 or gripper_qpos.size != 2:
        raise ValueError(
            "FastWAM libero state requires eef_pos[3], eef_quat[4] and gripper_qpos[2]; "
            f"got sizes {eef_pos.size}/{eef_quat.size}/{gripper_qpos.size}"
        )
    axis_angle = _quat_to_axisangle(eef_quat)
    return np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32)


def encode_robotwin_state_joint14(state_raw: Dict[str, Any]) -> np.ndarray:
    """Canonical RoboTwin proprio -> FastWAM 14D joint state (pass-through).

    Official ``deploy_policy`` feeds ``observation["joint_action"]["vector"]``
    to the proprio encoder without remapping; the 14D checkpoint's stats were
    computed over the same layout (left arm 7 + right arm 7).
    """
    joint = state_raw.get("joint")
    if joint is None:
        raise ValueError("robotwin state_raw must contain 'joint' (joint_action.vector)")
    joint = np.asarray(joint, dtype=np.float32).reshape(-1)
    if joint.size != 14:
        raise ValueError(f"robotwin_joint14 state expects 14D, got {joint.size}D")
    return joint


@register_payload_builder("fastwam")
class FastWAMPayloadBuilder(PayloadBuilder):
    """FastWAM client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    #
    # Supported ``state_encoding`` values:
    #   ``libero_ee8`` — eef_pos + quat->axis-angle + gripper_qpos -> 8D
    #   ``robotwin_joint14`` — dual-arm joint vector pass-through -> 14D
    #     (T-cam 3-view pack; z-score stats; 24-step replan, factory-side)
    state_encoding: str = "libero_ee8"
    # LIBERO canonical action (pos + axis_angle + gripper) matches the model's
    # decoded 7D action; the composed decoder key is identity.
    action_encoding: str = "axis_angle"
    action_dim: int = 7
    action_horizon: int = 32
    # Open-loop chunk execution: the server replans once the cached chunk is
    # exhausted (or truncated to chunk_execute_steps server-side), which is
    # the official replan protocol — no per-step model call needed.
    disable_action_cache: bool = False

    def _encode_state(self, canonical: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode ``canonical.state_raw`` per ``self.state_encoding``."""
        state_raw = canonical.get("state_raw") or {}
        if self.state_encoding == "libero_ee8":
            return encode_libero_state_ee8(state_raw)
        if self.state_encoding == "robotwin_joint14":
            return encode_robotwin_state_joint14(state_raw)
        raise ValueError(f"Unsupported fastwam state_encoding: {self.state_encoding!r}")

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by the FastWAM eval wrapper."""
        images = _pack_fastwam_images(canonical["images"], self.state_encoding)
        return {
            "images": images,
            "instructions": [FASTWAM_PROMPT_TEMPLATE.format(task=canonical["instruction"])],
            "state": self._encode_state(canonical),
        }
