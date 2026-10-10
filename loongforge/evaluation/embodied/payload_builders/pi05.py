# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pi0.5 PayloadBuilder.

Pi0.5 uses an ee-space canonical action (pos + axis_angle + gripper) and does
not consume a proprio state on eval — the training-side collator stubs it out.
This PayloadBuilder therefore keeps ``state`` optional and mostly emits image
and instruction fields, plus the ``unnorm_key`` dataset statistics selector.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from loongforge.evaluation.embodied.adapters.robotwin import ROBOTWIN_ACTION_DIM
from loongforge.evaluation.embodied.payload_builders.base import PayloadBuilder
from loongforge.evaluation.embodied.payload_builders.registry import register_payload_builder

# openpi Aloha/RoboTwin proprio helpers (adapt_to_pi input side). These used to
# live in ``adapters/robotwin.py``; they now live next to their only consumer
# (this PayloadBuilder) so the proprio conversion no longer imports behavior
# from the adapter. The tiny ``_normalize_range`` / ``_unnormalize_range`` and
# ``_PI05_JOINT_FLIP_MASK`` duplicated in ``action_decoders/joint.py`` (action
# side) are intentionally copied rather than shared, to avoid a shared module.
_PI05_JOINT_FLIP_MASK = np.asarray([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)


def _normalize_range(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Linear normalize ``x`` from [min_val, max_val] into [0, 1]."""
    return (x - min_val) / (max_val - min_val)


def _unnormalize_range(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Inverse of :func:`_normalize_range`."""
    return x * (max_val - min_val) + min_val


def gripper_to_angular(value: np.ndarray) -> np.ndarray:
    """Aloha linear gripper -> pi angular space (openpi aloha_policy)."""
    value = _unnormalize_range(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position: np.ndarray, arm_length: float, horn_radius: float) -> np.ndarray:
        ratio = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(ratio, -1.0, 1.0))

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize_range(value, min_val=0.5476, max_val=1.6296)


def adapt_to_pi_decode_state(state: np.ndarray) -> np.ndarray:
    """Env joint state -> pi internal space (input side of adapt_to_pi)."""
    out = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    if out.size < ROBOTWIN_ACTION_DIM:
        raise ValueError(f"pi05_aloha_14d state must be {ROBOTWIN_ACTION_DIM}D, got {out.size}D")
    out = out[:ROBOTWIN_ACTION_DIM]
    out = _PI05_JOINT_FLIP_MASK * out
    out[[6, 13]] = gripper_to_angular(out[[6, 13]])
    return out.astype(np.float32)


def _pack_images(images_by_cam: Dict[str, Optional[np.ndarray]]) -> List[np.ndarray]:
    """Convert per-camera dict to the ``[primary(+wrist|left|right)]`` list Pi05 expects.

    Bimanual benchmarks (RoboTwin) send ``head + left + right``. Single-arm
    benchmarks send ``primary + wrist``. LIBERO uses ``primary + wrist``.
    """
    primary = images_by_cam.get("primary")
    if primary is None:
        primary = images_by_cam.get("head")
    if primary is None:
        raise ValueError("images_by_cam must contain 'primary' or 'head'")
    packed: List[np.ndarray] = [np.asarray(primary)]

    left = images_by_cam.get("left")
    right = images_by_cam.get("right")
    if left is not None and right is not None:
        packed.append(np.asarray(left))
        packed.append(np.asarray(right))
        return packed

    wrist = images_by_cam.get("wrist")
    if wrist is None:
        wrist = right if right is not None else left
    if wrist is not None:
        packed.append(np.asarray(wrist))
    return packed


@register_payload_builder("pi05")
class Pi05PayloadBuilder(PayloadBuilder):
    """Pi0.5 client-side payload assembly."""

    # Capability declarations (YAML-overridable via type annotations).
    #
    # Supported ``state_encoding`` values:
    #   ``""``            — no state kwarg (LIBERO / CALVIN / SimplerEnv default)
    #   ``passthrough``   — send ``canonical["state_raw"]["joint"]`` as-is; used
    #                       for ManiSkill where the adapter emits 8D Panda qpos
    #   ``aloha_pi``      — RoboTwin: env joint -> pi space (openpi adapt_to_pi)
    state_encoding: str = ""
    action_encoding: str = "axis_angle"  # pos(3)+axis_angle(3)+grip(1)
    action_dim: int = 7
    action_horizon: int = 50

    # Non-annotated attributes are internal (not YAML-overridable).
    unnorm_key = ""

    def __init__(
        self,
        yaml_model=None,
        yaml_server=None,
        yaml_benchmark=None,
    ) -> None:
        """Read pi05-specific instance fields on top of annotated defaults."""
        super().__init__(yaml_model=yaml_model, yaml_server=yaml_server, yaml_benchmark=yaml_benchmark)
        self.unnorm_key = str(self._yaml_model.get("unnorm_key", "") or "")

    def _encode_state(self, canonical: Dict[str, Any]) -> Optional[np.ndarray]:
        """Encode ``canonical.state_raw`` per ``self.state_encoding``."""
        if not self.state_encoding:
            return None
        state_raw = canonical.get("state_raw") or {}
        if self.state_encoding == "passthrough":
            joint = state_raw.get("joint")
            return np.asarray(joint, dtype=np.float32) if joint is not None else None
        if self.state_encoding == "aloha_pi":
            # RoboTwin openpi adapt_to_pi: env joint -> pi internal space.
            joint = state_raw.get("joint")
            return adapt_to_pi_decode_state(np.asarray(joint, dtype=np.float32)) if joint is not None else None
        raise ValueError(f"Unsupported pi05 state_encoding: {self.state_encoding!r}")

    def build(self, canonical: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return the kwargs consumed by ``Pi05.predict_action``."""
        images = _pack_images(canonical["images"])
        kwargs: Dict[str, Any] = {
            "images": images,
            "instructions": [str(canonical["instruction"])],
            "state": self._encode_state(canonical),
        }
        if self.unnorm_key:
            kwargs["unnorm_key"] = self.unnorm_key
        return kwargs
