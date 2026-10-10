# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.6 PayloadBuilder.

GR00T-N1.6 consumes a per-embodiment *raw* proprio state (its own
``StateActionProcessor`` normalizes it inside ``predict_action``) plus the
benchmark camera views. This builder encodes the LIBERO adapter's
``state_raw`` (ee pos + quat + gripper) into the 8D ``libero_panda`` raw state
layout ``[x, y, z, roll, pitch, yaw, gripper, gripper]`` and packs images as
``[primary, wrist]`` (the two ``libero_panda`` video views ``image`` /
``image2``).

Action encoding is ``axis_angle`` so, against LIBERO's ``axis_angle`` action
space, the orchestrator composes an identity ActionDecoder — GR00T-N1.6's
decoded LIBERO action ``[x, y, z, roll, pitch, yaw, gripper]`` is already what
the LIBERO adapter's ``action_from_canonical`` consumes as a flat 7D array.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from loongforge.evaluation.embodied.payload_builders.base import PayloadBuilder
from loongforge.evaluation.embodied.payload_builders.pi05 import _pack_images
from loongforge.evaluation.embodied.payload_builders.registry import register_payload_builder

logger = logging.getLogger(__name__)

# WidowX finger joint limits (fixed by the ManiSkill2_real2sim WidowX URDF),
# used to reproduce ``WidowX.get_gripper_closedness()`` without SAPIEN access.
# The official agent reads these from ``robot.get_qlimits()`` at runtime; here they
# are constants, so a URDF change would silently rescale the whole gripper channel.
# ``_warn_finger_qpos_out_of_limits`` makes that desync loud instead.
WIDOWX_FINGER_QMIN = 0.015
WIDOWX_FINGER_QMAX = 0.037

_finger_qpos_warned = False


def _warn_finger_qpos_out_of_limits(finger_qpos: np.ndarray) -> None:
    """Warn once if finger qpos leaves the hardcoded ``[QMIN, QMAX]`` window.

    A small excursion is normal (contact / solver overshoot); a persistent or
    large one means ``WIDOWX_FINGER_QMIN/QMAX`` no longer match the loaded URDF,
    which would silently mis-scale the gripper proprio channel.
    """
    global _finger_qpos_warned
    if _finger_qpos_warned:
        return
    span = WIDOWX_FINGER_QMAX - WIDOWX_FINGER_QMIN
    lo = WIDOWX_FINGER_QMIN - 0.1 * span
    hi = WIDOWX_FINGER_QMAX + 0.1 * span
    if np.any(finger_qpos < lo) or np.any(finger_qpos > hi):
        _finger_qpos_warned = True
        logger.warning(
            "WidowX finger qpos %s outside hardcoded limits [%.4f, %.4f]; "
            "check WIDOWX_FINGER_QMIN/QMAX against the loaded URDF "
            "(gripper proprio would be mis-scaled). Warning shown once.",
            np.asarray(finger_qpos).tolist(),
            WIDOWX_FINGER_QMIN,
            WIDOWX_FINGER_QMAX,
        )


def _encode_libero_ee_euler(state_raw: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build the 8D ``libero_panda`` raw state from LIBERO ee fields.

    Layout (matches ``LIBERO_PANDA_MODALITY_META``):
    ``[x, y, z, roll, pitch, yaw, gripper_finger0, gripper_finger1]``. Rotation
    is derived from ``eef_quat`` (xyzw) as intrinsic-xyz Euler. The 2D gripper
    slot uses the native Panda finger qpos ``[+finger, -finger]`` when the
    adapter provides ``gripper_qpos``; otherwise it falls back to ``[g, -g]``.
    """
    from scipy.spatial.transform import Rotation

    eef_pos = np.asarray(state_raw.get("eef_pos"), dtype=np.float32).reshape(-1)[:3]
    eef_quat = np.asarray(state_raw.get("eef_quat"), dtype=np.float32).reshape(-1)[:4]
    euler = Rotation.from_quat(eef_quat).as_euler("xyz").astype(np.float32)

    gripper_qpos = state_raw.get("gripper_qpos")
    if gripper_qpos is not None:
        g = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
        grip2 = g[:2] if g.size >= 2 else np.array([g[0], -g[0]], dtype=np.float32)
    else:
        grip_val = state_raw.get("gripper")
        gv = float(grip_val) if grip_val is not None else 0.0
        grip2 = np.array([gv, -gv], dtype=np.float32)
    return np.array(
        [eef_pos[0], eef_pos[1], eef_pos[2], euler[0], euler[1], euler[2], grip2[0], grip2[1]],
        dtype=np.float32,
    )


def _encode_simpler_widowx(state_raw: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build the 8D ``oxe_widowx`` raw state, matching the official GR00T
    ``WidowXBridgeEnv`` wrapper (Isaac-GR00T gr00t/eval/sim/SimplerEnv).

    Official: ``proprio = obs.agent.eef_pos`` = ``[x,y,z, quat_wxyz(4), gripper]``;
    ``rpy = mat2euler(quat2mat(quat) @ default_rot.T)`` with
    ``default_rot = [[0,0,1],[0,1,0],[-1,0,0]]``; state =
    ``[x, y, z, roll, pitch, yaw, pad=0, gripper]``.

    Our ManiSkill2 obs does not expose ``agent.eef_pos``, so the ee pose is
    reconstructed in the robot base frame from ``base_pose`` / ``tcp_pose``
    (verified to reproduce the official small-euler / in-range state). Falls
    back to a raw ``eef_pos`` field if the adapter ever provides one.
    """
    from transforms3d import euler as te, quaternions as tq

    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    eef = state_raw.get("eef_pos")
    if eef is not None:
        eef = np.asarray(eef, dtype=np.float64).reshape(-1)
        if eef.size >= 8:
            rm = tq.quat2mat(eef[3:7])
            rpy = te.mat2euler(rm @ default_rot.T)
            return np.array([eef[0], eef[1], eef[2], rpy[0], rpy[1], rpy[2], 0.0, eef[7]], dtype=np.float32)

    base_pose = state_raw.get("base_pose")
    tcp_pose = state_raw.get("tcp_pose")
    if base_pose is None or tcp_pose is None:
        return None
    try:
        from sapien.core import Pose

        base = np.asarray(base_pose, dtype=np.float64).reshape(-1)
        tcp = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
        ee = Pose(p=base[:3], q=base[3:]).inv() * Pose(p=tcp[:3], q=tcp[3:])
        pos = np.asarray(ee.p, dtype=np.float64)
        rpy = te.mat2euler(tq.quat2mat(np.asarray(ee.q, dtype=np.float64)) @ default_rot.T)
    except Exception:
        return None

    # Gripper openness, matching the official GR00T WidowX wrapper, which reads
    # ``agent.eef_pos[7] = 1 - get_gripper_closedness()`` (a normalized openness
    # in [0, 1]; 1.0 = fully open). Upstream ManiSkill2_real2sim does not expose
    # ``eef_pos``, so reproduce ``WidowX.get_gripper_closedness()`` here from the
    # two finger qpos against their (fixed) joint limits.
    joint = state_raw.get("joint")
    grip = 1.0
    if joint is not None:
        j = np.asarray(joint, dtype=np.float64).reshape(-1)
        if j.size >= 2:
            _warn_finger_qpos_out_of_limits(j[-2:])
            closedness = np.mean((WIDOWX_FINGER_QMAX - j[-2:]) / (WIDOWX_FINGER_QMAX - WIDOWX_FINGER_QMIN))
            grip = float(1.0 - max(closedness, 0.0))
    return np.array(
        [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], 0.0, grip],
        dtype=np.float32,
    )


@register_payload_builder("gr00tn1d6")
class GrootN1d6PayloadBuilder(PayloadBuilder):
    """GR00T-N1.6 client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    #
    # Supported ``state_encoding`` values:
    #   ``libero_ee_euler`` — LIBERO 8D raw state [pos(3), euler(3), grip, grip]
    #   ``simpler_widowx``  — SimplerEnv WidowX 8D raw state [pos(3), euler(3), pad, grip]
    #   ``""``              — no state kwarg emitted
    state_encoding: str = "libero_ee_euler"
    action_encoding: str = "axis_angle"  # x,y,z + roll,pitch,yaw + gripper
    action_dim: int = 7
    action_horizon: int = 16

    def _encode_state(self, canonical: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode ``canonical.state_raw`` per ``self.state_encoding``."""
        if not self.state_encoding:
            return None
        state_raw = canonical.get("state_raw") or {}
        if self.state_encoding == "libero_ee_euler":
            return _encode_libero_ee_euler(state_raw)
        if self.state_encoding == "simpler_widowx":
            return _encode_simpler_widowx(state_raw)
        raise ValueError(f"Unsupported gr00tn1d6 state_encoding: {self.state_encoding!r}")

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``GrootN1d6Policy.predict_action``."""
        images = _pack_images(canonical["images"])
        if self.state_encoding == "simpler_widowx":
            # Official WidowXBridgeEnv resizes the view to 256x256 before the
            # model. The remaining letterbox-pad + 95%-center-crop step (matching
            # Gr00tN1d6Processor.eval_image_transform) is applied inside
            # GrootN1d6Policy.predict_action, not here — see modeling_groot_n1_6.py.
            import cv2

            images = [
                cv2.resize(np.asarray(img), (256, 256)) if img is not None else img
                for img in images
            ]
        return {
            "images": images,
            "instructions": [str(canonical["instruction"])],
            "state": self._encode_state(canonical),
        }
