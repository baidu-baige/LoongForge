# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-N1.7 PayloadBuilder.

GR00T-N1.7 consumes a per-embodiment *raw* proprio state (its own
``StateActionProcessor`` normalizes it inside ``predict_action``) plus the
benchmark camera views.

The ``libero_sim`` embodiment layout differs from N1.6's ``libero_panda`` in two
ways that were verified against the official Isaac-GR00T LIBERO client
(``gr00t/eval/sim/LIBERO/libero_env.py::_process_observation``) and against the
checkpoint's own ``statistics.json``:

* rotation is **axis-angle** (``quat2axisangle(robot0_eef_quat)``), not
  intrinsic-xyz Euler as N1.6 uses;
* the gripper slot carries the raw 2-DoF ``robot0_gripper_qpos`` at ``[6:8]``
  (the checkpoint's ``libero_sim.state.gripper`` statistics have 2 values per
  stat, and the official env declares ``state.gripper`` with ``shape=(2,)``).

Images are packed as ``[primary, wrist]`` — the two ``libero_sim`` video views
``image`` / ``wrist_image``. The 180-degree flip the official client applies to
both cameras is already done by the LIBERO adapter, and the crop/resize
pipeline runs inside ``predict_action``, so nothing image-side happens here.

Action encoding is ``axis_angle``, matching the LIBERO adapter's ``axis_angle``
action space, so the orchestrator composes an identity ActionDecoder: the
decoded chunk ``[x, y, z, rx, ry, rz, gripper]`` is already what
``LiberoAdapter.action_from_canonical`` consumes as a flat 7D array.

The ``simpler_env_widowx`` embodiment (SimplerEnv Bridge) uses a **single**
exterior view and an 8D state whose index 6 is a dead ``pad`` channel and whose
index 7 is a 1D gripper openness in ``[0, 1]``; set ``state_encoding:
simpler_widowx`` plus ``action_encoding: simpler_abs_euler`` in the YAML for that
path. Do **not** set ``action_horizon`` there: YAML ``model:`` keys are also
merged into ``GrootN1d7Config``, whose ``action_horizon`` is the DiT
flow-matching sequence length (40 in the released checkpoints), and shortening it
changes every DiT output. The decoded chunk length comes from the embodiment's
action ``delta_indices``, not from this field.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder
from loongforge.embodied.eval.payload_builders.groot_n1_6 import _encode_simpler_widowx
from loongforge.embodied.eval.payload_builders.pi05 import _pack_images
from loongforge.embodied.eval.payload_builders.registry import register_payload_builder

logger = logging.getLogger(__name__)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3D axis-angle vector.

    Mirrors ``quat2axisangle`` in the official Isaac-GR00T LIBERO env (which is
    robosuite's implementation): the returned vector is the rotation axis scaled
    by the rotation angle in radians. Uses the same near-identity early return so
    a numerically degenerate quaternion cannot produce NaNs.
    """
    import math

    quat = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
    # clip to guard against |w| slightly above 1 from float error
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(w)) / den).astype(np.float32)


def _encode_libero_ee_axis_angle(state_raw: Dict[str, Any]) -> Optional[np.ndarray]:
    """Build the 8D ``libero_sim`` raw state from LIBERO ee fields.

    Layout (matches ``LIBERO_SIM_MODALITY_META``):
    ``[x, y, z, rx, ry, rz, gripper_finger0, gripper_finger1]``, where the
    rotation triple is the axis-angle of ``eef_quat`` and the gripper slot is the
    native 2-DoF finger qpos. Returns ``None`` when the adapter did not provide
    the ee pose, so the caller can surface a missing-proprio failure instead of
    silently sending a zero state.
    """
    eef_pos = state_raw.get("eef_pos")
    eef_quat = state_raw.get("eef_quat")
    if eef_pos is None or eef_quat is None:
        return None

    pos = np.asarray(eef_pos, dtype=np.float32).reshape(-1)[:3]
    axis_angle = _quat2axisangle(eef_quat)

    gripper_qpos = state_raw.get("gripper_qpos")
    if gripper_qpos is not None:
        finger = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    else:
        finger = np.empty(0, dtype=np.float32)
    if finger.size >= 2:
        grip2 = finger[:2]
    elif finger.size == 1:
        grip2 = np.array([finger[0], -finger[0]], dtype=np.float32)
    else:
        grip_val = state_raw.get("gripper")
        gv = float(grip_val) if grip_val is not None else 0.0
        grip2 = np.array([gv, -gv], dtype=np.float32)

    return np.array(
        [pos[0], pos[1], pos[2], axis_angle[0], axis_angle[1], axis_angle[2], grip2[0], grip2[1]],
        dtype=np.float32,
    )


@register_payload_builder("gr00tn1d7")
class GrootN1d7PayloadBuilder(PayloadBuilder):
    """GR00T-N1.7 client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    #
    # Supported ``state_encoding`` values:
    #   ``libero_ee_axis_angle`` — LIBERO 8D raw state [pos(3), axis_angle(3), grip, grip]
    #   ``simpler_widowx``       — SimplerEnv Bridge 8D raw state
    #                              [pos(3), euler(3), pad=0, gripper_openness]
    #   ``""``                   — no state kwarg emitted
    state_encoding: str = "libero_ee_axis_angle"
    action_encoding: str = "axis_angle"  # x,y,z + axis-angle(3) + gripper
    action_dim: int = 7
    # libero_sim's action ModalityConfig uses delta_indices=range(16); the model's
    # own action_horizon (40) is the DiT sequence length, not the decoded length.
    # simpler_env_widowx uses delta_indices=range(8) — override in YAML.
    action_horizon: int = 16

    def _encode_state(self, canonical: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode ``canonical.state_raw`` per ``self.state_encoding``."""
        if not self.state_encoding:
            return None
        state_raw = canonical.get("state_raw") or {}
        if self.state_encoding == "libero_ee_axis_angle":
            return _encode_libero_ee_axis_angle(state_raw)
        if self.state_encoding == "simpler_widowx":
            # ``simpler_env_widowx`` and N1.6's ``oxe_widowx`` declare the same 8D
            # layout and the official WidowXBridgeEnv wrapper is byte-identical
            # between the two releases (``[x,y,z] + mat2euler(quat2mat(q) @
            # default_rot.T) + pad=0 + eef_pos[7]``), so reuse the N1.6 encoder
            # rather than forking it -- see DRAWER_调查报告 §7.2 for the gripper trap
            # that fork would re-introduce.
            return _encode_simpler_widowx(state_raw)
        raise ValueError(f"Unsupported Gr00tN1d7 state_encoding: {self.state_encoding!r}")

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``GrootN1d7Policy.predict_action``."""
        state = self._encode_state(canonical)
        if self.state_encoding and state is None:
            raise ValueError(
                "Gr00tN1d7 state_encoding="
                f"{self.state_encoding!r} could not be built: canonical state_raw is "
                "missing the fields it needs (LIBERO: eef_pos/eef_quat; SimplerEnv: "
                "base_pose/tcp_pose or eef_pos). Sending a zero proprio would "
                "silently degrade the policy, so this is a hard failure."
            )
        images = _pack_images(canonical["images"])
        if self.state_encoding == "simpler_widowx":
            # Official WidowXBridgeEnv._process_observation resizes the single
            # exterior view to 256x256 (cv2, default INTER_LINEAR) before the
            # model; the letterbox-pad + shortest-edge + 95%-center-crop stage
            # runs inside predict_action, not here.
            import cv2

            images = [
                cv2.resize(np.asarray(img), (256, 256)) if img is not None else img
                for img in images
            ]
        return {
            "images": images,
            "instructions": [str(canonical["instruction"])],
            "state": state,
        }
