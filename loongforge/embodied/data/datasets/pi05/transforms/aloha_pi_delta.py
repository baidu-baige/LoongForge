# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin pi0.5 Aloha absolute-joint -> pi-space delta action transform.

Training-side inverse of the eval-time ActionDecoder ``pi05_aloha_robotwin``
(``loongforge/embodied/eval/action_decoders/joint.py``). RoboTwin Unified stores
**absolute 14D joint** actions in env space, but the pi0.5 ``pi05_aloha_14d``
eval bridge expects the model to emit **pi-space joint deltas** (arm joints
relative to the chunk-anchor proprio, grippers absolute), which the decoder turns
back into env joints via ``delta_to_absolute_actions`` + ``adapt_to_pi_encode_actions``.

This transform makes the training target the exact inverse of that decode, so the
learned action space matches inference:

    pi_abs      = adapt_to_pi_decode_actions(env_action)          # inverse of adapt_to_pi_encode
    pi_state    = adapt_to_pi_decode_state(observation.state)     # chunk anchor (frame 0), == eval proprio
    model_delta = where(DELTA_MASK, pi_abs - pi_state, pi_abs)    # arm relative, gripper absolute

It ALSO rewrites ``observation.state`` in-place to that same ``pi_state`` (the
absolute pi-space proprio the eval bridge feeds the model). This keeps the
train-time state discretization (StateDiscretizationTransform) in the SAME space
as eval, whose ``predict_action`` discretizes ``adapt_to_pi_decode_state(env_state)``
against the pi-space state stats. Action delta is computed from the RAW env state
FIRST, then the state key is overwritten -- no double conversion.

Round-trip: ``eval_decode(this_transform(a, s)) == a`` exactly -- arm dims because FLIP is
+/-1 (self-inverse), gripper dims because the range map is an analytic linear
inverse and grippers never enter the delta (DELTA_MASK is False there). The state
output equals the eval proprio decode ``adapt_to_pi_decode_state``.

The small constants/masks are intentionally duplicated from the eval side (the
eval code itself duplicates them across proprio/action to avoid a shared module);
the verification test asserts they stay in lock-step with the real decoder.

IMPORTANT (data prep): when this is enabled, the ``action`` AND ``observation.state``
normalization stats consumed by ``ActionTransform`` / ``StateDiscretizationTransform``
MUST be in pi-space (delta for action, decoded proprio for state). Absolute-action
q01/q99 will mis-scale the deltas; absolute-state q01/q99 will mis-scale the proprio.
Use the pi-space stats view (meta/stats.json action + observation.state replaced with
the eval assets ``pi05_robotwin2_dataset_stats.json`` values).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from loongforge.embodied.data.datasets.transforms.base import BaseTransform

# Aloha/RoboTwin 14D layout: [Larm(0-5), Lgrip(6), Rarm(7-12), Rgrip(13)].
# Duplicated from eval action_decoders/joint.py + payload_builders/pi05.py.
_PI05_JOINT_FLIP_MASK = np.asarray([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)
_PI05_DELTA_JOINT_MASK = np.asarray(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)
_ROBOTWIN_ACTION_DIM = 14
# --- gripper: action side (eval action_decoders/joint.py:gripper_from_angular) ---
_GRIP_ANGULAR_MIN = -0.6213
_GRIP_ANGULAR_MAX = 1.4910
_GRIP_ANGULAR_BIAS = 0.5476

# --- gripper: proprio/state side (eval payload_builders/pi05.py:gripper_to_angular) ---
_GRIP_LINEAR_MIN = 0.01844
_GRIP_LINEAR_MAX = 0.05800
_GRIP_RADIAN_MIN = 0.5476
_GRIP_RADIAN_MAX = 1.6296
_ARM_LENGTH = 0.036
_HORN_RADIUS = 0.022


def _normalize_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x - lo) / (hi - lo)


def _unnormalize_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x * (hi - lo) + lo


def _gripper_from_angular_inv(value: np.ndarray) -> np.ndarray:
    """env gripper -> pi angular gripper; exact inverse of joint.py:gripper_from_angular."""
    return _unnormalize_range(value, _GRIP_ANGULAR_MIN, _GRIP_ANGULAR_MAX) - _GRIP_ANGULAR_BIAS


def _gripper_to_angular(value: np.ndarray) -> np.ndarray:
    """Aloha linear gripper -> pi angular; == eval payload_builders/pi05.py:gripper_to_angular."""
    value = _unnormalize_range(value, _GRIP_LINEAR_MIN, _GRIP_LINEAR_MAX)
    ratio = (_HORN_RADIUS ** 2 + value ** 2 - _ARM_LENGTH ** 2) / (2 * _HORN_RADIUS * value)
    value = np.arcsin(np.clip(ratio, -1.0, 1.0))
    return _normalize_range(value, _GRIP_RADIAN_MIN, _GRIP_RADIAN_MAX)


def adapt_to_pi_decode_actions(env_action: np.ndarray) -> np.ndarray:
    """env joint command -> pi internal absolute action (inverse of adapt_to_pi_encode_actions)."""
    flat = np.asarray(env_action, dtype=np.float32).reshape(-1)[:_ROBOTWIN_ACTION_DIM]
    out = _PI05_JOINT_FLIP_MASK * flat
    out[[6, 13]] = _gripper_from_angular_inv(flat[[6, 13]])
    return out.astype(np.float32)


def adapt_to_pi_decode_state(env_state: np.ndarray) -> np.ndarray:
    """env joint state -> pi internal space; == eval payload_builders/pi05.py:adapt_to_pi_decode_state."""
    flat = np.asarray(env_state, dtype=np.float32).reshape(-1)[:_ROBOTWIN_ACTION_DIM]
    out = _PI05_JOINT_FLIP_MASK * flat
    out[[6, 13]] = _gripper_to_angular(flat[[6, 13]])
    return out.astype(np.float32)


def env_actions_to_pi_delta(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """[T,14] env absolute actions + [14] env state (chunk anchor) -> [T,14] pi-space delta."""
    chunk = np.asarray(actions, dtype=np.float32).reshape(-1, actions.shape[-1])[:, :_ROBOTWIN_ACTION_DIM]
    pi_state = adapt_to_pi_decode_state(state)
    out = []
    for row in chunk:
        pi_abs = adapt_to_pi_decode_actions(row)
        out.append(np.where(_PI05_DELTA_JOINT_MASK, pi_abs - pi_state, pi_abs))
    return np.stack(out).astype(np.float32)


class AlohaPiDeltaActionTransform(BaseTransform):
    """Convert RoboTwin env-space absolute 14D joints into pi-space deltas + proprio.

    Must run BEFORE ``ActionTransform`` (normalization/padding) and BEFORE
    ``StateDiscretizationTransform``: it reads the raw ``action`` chunk and the
    chunk-anchor proprio ``observation.state`` (frame 0), writes the pi-space
    delta back into ``action`` (arm joints relative to the anchor; grippers
    absolute in pi-angular space), and -- when ``convert_state`` is set --
    overwrites ``observation.state`` with ``adapt_to_pi_decode_state`` so the
    discretized proprio matches the eval bridge exactly.

    The action delta is computed from the RAW env state first; the state key is
    overwritten afterwards, so there is no double conversion within a sample.
    """

    def __init__(
        self,
        action_key: str = "action",
        state_key: str = "observation.state",
        convert_state: bool = True,
        training: bool = True,
    ):
        super().__init__(apply_to=[action_key, state_key], training=training)
        self.action_key = action_key
        self.state_key = state_key
        self.convert_state = convert_state

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        action = data.get(self.action_key)
        state = data.get(self.state_key)
        if action is None or state is None:
            return data
        was_tensor = isinstance(action, torch.Tensor)
        a = action.detach().cpu().numpy() if was_tensor else np.asarray(action, dtype=np.float32)
        state_is_tensor = isinstance(state, torch.Tensor)
        s = state.detach().cpu().numpy() if state_is_tensor else np.asarray(state, dtype=np.float32)
        a2 = a.reshape(1, -1) if a.ndim == 1 else a
        # Compute the delta from the RAW env state (anchor) BEFORE touching state.
        delta = env_actions_to_pi_delta(a2, s)
        if a.ndim == 1:
            delta = delta[0]
        data[self.action_key] = torch.from_numpy(delta).float() if was_tensor else delta

        # Rewrite proprio into the SAME pi-space the eval bridge feeds the model
        # (adapt_to_pi_decode_state), so train/eval state discretization align.
        if self.convert_state:
            pi_state = adapt_to_pi_decode_state(s)
            if s.size == pi_state.size:
                pi_state = pi_state.reshape(s.shape)
            data[self.state_key] = (
                torch.from_numpy(pi_state).float() if state_is_tensor else pi_state
            )
        return data
