# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero PayloadBuilder.

DreamZero consumes a per-step 2-view observation (exterior + wrist) plus an
8D proprio state (eef_pos + eef_quat + gripper). Temporal history, the video
grid, prompt formatting, normalization and action denormalization all live
server-side (see ``factories/dreamzero_factory.py``); this builder only packs
the canonical dict into the ``predict_action`` kwargs and declares that the
model owns its action queue (``disable_action_cache``) so the server wrapper
observes every env step and maintains the 33-frame history window.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.embodied.eval.payload_builders.base import PayloadBuilder
from loongforge.embodied.eval.payload_builders.registry import register_payload_builder


def _pack_dreamzero_images(images_by_cam: Dict[str, Optional[np.ndarray]]) -> List[np.ndarray]:
    """Return the ``[exterior, wrist]`` view list DreamZero expects.

    The LIBERO modality config concatenates the two views horizontally
    (exterior left, wrist right) server-side; the order here must match that
    layout and the training-time prompt text.
    """
    exterior = images_by_cam.get("primary")
    if exterior is None:
        exterior = images_by_cam.get("head")
    if exterior is None:
        raise ValueError("images_by_cam must contain 'primary' or 'head' (exterior view)")
    wrist = images_by_cam.get("wrist")
    if wrist is None:
        wrist = images_by_cam.get("right")
    if wrist is None:
        raise ValueError("images_by_cam must contain 'wrist' for DreamZero LIBERO eval")
    return [np.asarray(exterior), np.asarray(wrist)]


def encode_libero_state(state_raw: Dict[str, Any]) -> np.ndarray:
    """Canonical LIBERO proprio -> DreamZero 8D state (pos3 + quat4 + gripper1).

    Matches the ``state.state`` layout of the ``libero_sim`` modality config
    used at training time (8D state).
    """
    eef_pos = np.asarray(state_raw.get("eef_pos"), dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(state_raw.get("eef_quat"), dtype=np.float32).reshape(-1)
    gripper = state_raw.get("gripper")
    if eef_pos.size != 3 or eef_quat.size != 4 or gripper is None:
        raise ValueError(
            "DreamZero libero state requires eef_pos[3], eef_quat[4] and gripper; "
            f"got sizes {eef_pos.size}/{eef_quat.size}/{gripper}"
        )
    return np.concatenate(
        [eef_pos, eef_quat, np.asarray([gripper], dtype=np.float32)]
    ).astype(np.float32)


@register_payload_builder("dreamzero")
class DreamZeroPayloadBuilder(PayloadBuilder):
    """DreamZero client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    #
    # Supported ``state_encoding`` values:
    #   ``libero_ee`` — eef_pos + eef_quat + gripper -> 8D (LIBERO default)
    state_encoding: str = "libero_ee"
    # LIBERO canonical action (pos + axis_angle + gripper) matches the model's
    # decoded 7D action; the composed decoder key is identity.
    action_encoding: str = "axis_angle"
    action_dim: int = 7
    action_horizon: int = 16
    # DreamZero is a closed-loop stateful policy: the server wrapper must be
    # called every env step to grow the 33-frame history window and pops one
    # action from its own queue per step.
    disable_action_cache: bool = True

    def _encode_state(self, canonical: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode ``canonical.state_raw`` per ``self.state_encoding``."""
        state_raw = canonical.get("state_raw") or {}
        if self.state_encoding == "libero_ee":
            return encode_libero_state(state_raw)
        raise ValueError(f"Unsupported dreamzero state_encoding: {self.state_encoding!r}")

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by the DreamZero eval wrapper."""
        images = _pack_dreamzero_images(canonical["images"])
        return {
            "images": images,
            "instructions": [str(canonical["instruction"])],
            "state": self._encode_state(canonical),
        }
