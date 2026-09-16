# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ActionDecoders for models that output **joint(-delta)** actions.

Currently only pi0.5 RoboTwin (openpi aloha protocol). Add future joint-space
models' decoders here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.evaluation.action_decoders.base import ActionDecoder, register_action_decoder
from loongforge.evaluation.adapters.robotwin import ROBOTWIN_ACTION_DIM

# openpi Aloha/RoboTwin action helpers (adapt_to_pi output side + delta->abs).
# These used to live in ``adapters/robotwin.py``; they now live next to their
# only consumer (the RoboTwin pi0.5 decoder) so the action side no longer
# imports behavior from the adapter. ``_normalize_range``, ``_unnormalize_range``
# and ``_PI05_JOINT_FLIP_MASK`` are intentionally duplicated from
# ``payload_builders/pi05.py`` (proprio side) rather than shared, to avoid a
# shared module.
_PI05_JOINT_FLIP_MASK = np.asarray([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
_PI05_DELTA_JOINT_MASK = np.asarray(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)


def _normalize_range(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Linear normalize ``x`` from [min_val, max_val] into [0, 1]."""
    return (x - min_val) / (max_val - min_val)


def _unnormalize_range(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Inverse of :func:`_normalize_range`."""
    return x * (max_val - min_val) + min_val


def gripper_from_angular(value: np.ndarray) -> np.ndarray:
    """pi angular gripper -> Aloha/RoboTwin env gripper (openpi aloha_policy)."""
    value = value + 0.5476
    return _normalize_range(value, min_val=-0.6213, max_val=1.4910)


def adapt_to_pi_encode_actions(actions: np.ndarray) -> np.ndarray:
    """pi internal absolute actions -> env joint commands (output side of adapt_to_pi)."""
    out = np.asarray(actions, dtype=np.float32).reshape(-1).copy()
    if out.size < ROBOTWIN_ACTION_DIM:
        raise ValueError(f"pi05_aloha_14d action must be {ROBOTWIN_ACTION_DIM}D, got {out.size}D")
    out = out[:ROBOTWIN_ACTION_DIM]
    out = _PI05_JOINT_FLIP_MASK * out
    out[[6, 13]] = gripper_from_angular(out[[6, 13]])
    return out.astype(np.float32)


def delta_to_absolute_actions(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Delta joints -> absolute using state at inference (openpi AbsoluteActions).

    Mask make_bool_mask(6, -1, 6, -1): joints relative to state; grippers absolute.
    """
    out = np.asarray(actions, dtype=np.float32).reshape(-1).copy()
    st = np.asarray(state, dtype=np.float32).reshape(-1)
    if out.size < ROBOTWIN_ACTION_DIM or st.size < ROBOTWIN_ACTION_DIM:
        raise ValueError(
            f"pi05_aloha_14d delta->abs requires {ROBOTWIN_ACTION_DIM}D action/state, "
            f"got action={out.size}D state={st.size}D"
        )
    out = out[:ROBOTWIN_ACTION_DIM]
    st = st[:ROBOTWIN_ACTION_DIM]
    out = np.where(_PI05_DELTA_JOINT_MASK, out + st, out)
    return out.astype(np.float32)


@register_action_decoder("pi05_aloha_robotwin")
class RoboTwinPi05AlohaDecoder(ActionDecoder):
    """Stateful pi0.5 RoboTwin ``pi05_aloha_14d``: delta joints -> abs -> env joints.

    openpi AbsoluteActions anchors delta joints to the pi-space state captured
    at chunk-inference time; cached steps reuse that anchor. ``ctx`` supplies
    ``pi_state`` (the pi-space proprio the Pi05PayloadBuilder produced via
    ``adapt_to_pi_decode_state``) and ``is_fresh_chunk`` (True on a fresh model
    forward, False on a server chunk-cache hit). Stateful -> overrides ``reset()``.
    """

    def __init__(self) -> None:
        """Initialize with no chunk anchor."""
        self._chunk_pi_state: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear the per-chunk pi-space anchor at episode boundaries."""
        self._chunk_pi_state = None

    def __call__(self, actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        """Decode unnormalized delta joints into absolute env joint commands."""
        chunk = np.asarray(actions, dtype=np.float32).reshape(-1, actions.shape[-1])
        pi_state = np.asarray(ctx["pi_state"], dtype=np.float32)
        if ctx.get("is_fresh_chunk", True) or self._chunk_pi_state is None:
            self._chunk_pi_state = pi_state.copy()
        out = []
        for row in chunk:
            abs_action = delta_to_absolute_actions(row[:ROBOTWIN_ACTION_DIM], self._chunk_pi_state)
            out.append(adapt_to_pi_encode_actions(abs_action))
        return np.stack(out).astype(np.float32)
