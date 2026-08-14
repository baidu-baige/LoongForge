# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# The RoboTwin composition below is derived from the official LingBot-VA
# repository (github.com/Robbyant/lingbot-va), Apache-2.0:
# ``evaluation/robotwin/eval_polict_client_openpi.py`` -> ``add_eef_pose`` /
# ``add_init_pose``. Where this file and the official client disagree, the
# official client is authoritative.

"""ActionDecoders for models that emit **per-arm position + quaternion + gripper**.

Currently only LingBot-VA RoboTwin: the model predicts a dual-arm end-effector
pose *relative to the pose the arms held at episode start*, which the official
client composes back onto that initial pose before stepping the env.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.action_decoders.base import ActionDecoder, register_action_decoder

# Per arm: xyz(3) + quat(4) + gripper(1); two arms.
_ARM_DIM = 8
ROBOTWIN_EE_QUAT_ACTION_DIM = 2 * _ARM_DIM


def _compose_arm_pose(relative: np.ndarray, initial: np.ndarray) -> np.ndarray:
    """Compose one arm's relative pose onto the episode's initial pose.

    Mirrors the official ``add_eef_pose``: translations add, rotations compose as
    ``init_R * relative_R``, and the gripper passes through untouched. The
    quaternion slice ``[3:7]`` is fed to ``scipy``'s ``from_quat`` exactly as
    upstream does, so whatever component order RoboTwin's ``endpose`` uses is
    preserved rather than reinterpreted here.
    """
    from scipy.spatial.transform import Rotation

    rel = np.asarray(relative, dtype=np.float64).reshape(-1)
    init = np.asarray(initial, dtype=np.float64).reshape(-1)
    rotation = Rotation.from_quat(init[3:7]) * Rotation.from_quat(rel[3:7])
    return np.concatenate([rel[:3] + init[:3], rotation.as_quat().reshape(-1), rel[7:8]])


def _flatten_endpose(endpose: Any) -> np.ndarray:
    """Flatten RoboTwin's endpose dict the way the official client builds its anchor.

    Official: ``left_endpose + [left_gripper] + right_endpose + [right_gripper]``.
    An already-flat 16D array is accepted as-is.
    """
    if isinstance(endpose, dict):
        return np.concatenate(
            [
                np.asarray(endpose["left_endpose"], dtype=np.float64).reshape(-1),
                np.asarray([endpose["left_gripper"]], dtype=np.float64),
                np.asarray(endpose["right_endpose"], dtype=np.float64).reshape(-1),
                np.asarray([endpose["right_gripper"]], dtype=np.float64),
            ]
        )
    return np.asarray(endpose, dtype=np.float64).reshape(-1)


@register_action_decoder("lingbot_va_robotwin_ee_dual")
class RoboTwinLingBotVAEeQuatDecoder(ActionDecoder):
    """LingBot-VA RoboTwin: 16D relative dual-arm ee pose -> 16D absolute env ee action.

    The anchor is the endpose observed on the episode's first step, captured here
    because the official client captures it from ``TASK_ENV.get_obs()`` before the
    rollout starts. Stateful -> overrides :meth:`reset`.

    The gripper is passed through as the model produced it (its dataset quantiles
    are ``q01=0`` / ``q99=1``), unlike the X-VLA RoboTwin decoder which binarizes
    to ``+/-1``. That difference is upstream's, not a simplification here.
    """

    def __init__(self) -> None:
        """Initialize with no episode anchor."""
        self._initial_endpose: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Drop the anchor so the next step re-captures it."""
        self._initial_endpose = None

    def __call__(self, actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        """Compose a relative-pose chunk onto the episode's initial endpose."""
        chunk = np.asarray(actions, dtype=np.float32).reshape(-1, actions.shape[-1])
        if chunk.shape[-1] < ROBOTWIN_EE_QUAT_ACTION_DIM:
            raise ValueError(
                f"lingbot_va RoboTwin decoder requires {ROBOTWIN_EE_QUAT_ACTION_DIM}D actions, "
                f"got {chunk.shape[-1]}D"
            )
        if self._initial_endpose is None:
            endpose = ctx.get("endpose")
            if endpose is None:
                raise ValueError(
                    "lingbot_va RoboTwin decoder needs ctx['endpose'] on the first step of an "
                    "episode: the model predicts a pose relative to where the arms started, so "
                    "without the anchor the commands would be interpreted as absolute and the "
                    "arms would jump to near the world origin."
                )
            anchor = _flatten_endpose(endpose)
            if anchor.size != ROBOTWIN_EE_QUAT_ACTION_DIM:
                raise ValueError(
                    f"ctx['endpose'] must be {ROBOTWIN_EE_QUAT_ACTION_DIM}D "
                    f"(left xyz+quat+gripper, then right), got {anchor.size}D"
                )
            self._initial_endpose = anchor

        anchor = self._initial_endpose
        out = [
            np.concatenate(
                [
                    _compose_arm_pose(row[:_ARM_DIM], anchor[:_ARM_DIM]),
                    _compose_arm_pose(row[_ARM_DIM:ROBOTWIN_EE_QUAT_ACTION_DIM], anchor[_ARM_DIM:]),
                ]
            )
            for row in chunk
        ]
        return np.stack(out).astype(np.float32)
