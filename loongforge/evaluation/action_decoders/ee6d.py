# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stateless decoders for models that output **ee6d** actions.

ee6d layout (per step): ``pos(3) + rot6d(6) + grip(1)`` (+ optional padding).
Each transform is a plain ``fn(actions, ctx) -> actions`` registered with
``@register_action_fn`` and wrapped in a ``FunctionDecoder`` at build time
(stateless — they ignore ``ctx``).

Registered keys:
* form-A (auto-composed ``{action_encoding}_to_{action_space}``):
  ``ee6d_to_axis_angle`` / ``ee6d_to_euler`` / ``ee6d_to_quat`` /
  ``ee6d_to_calvin_abs`` / ``ee6d_to_simpler_abs_euler``
* form-B RoboTwin (bridge-wired): ``ee6d_robotwin_ee_dual``
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from loongforge.evaluation.action_decoders.base import register_action_fn
from loongforge.evaluation.action_decoders.rotation import (
    rot6d_interleaved_to_matrix,
    rot6d_interleaved_to_quat,
    rot6d_to_axis_angle,
)
from loongforge.evaluation.adapters.robotwin import ROBOTWIN_EE6D_ACTION_DIM


def _robotwin_ee6d_row_to_ee(raw_row: np.ndarray) -> np.ndarray:
    """Convert a single 20D ee6d model action row to the 16D RoboTwin ee action.

    Official client: per arm xyz(3) + rot6d_interleaved->quat(4) +
    (1 - 2*(grip > 0.7))(1), executed via
    ``TASK_ENV.take_action(action, action_type='ee')``.
    """
    raw = np.asarray(raw_row, dtype=np.float32).reshape(-1)

    def arm(offset: int) -> np.ndarray:
        pos = raw[offset : offset + 3]
        quat = rot6d_interleaved_to_quat(raw[offset + 3 : offset + 9]).reshape(-1)
        grip = np.asarray([1.0 - 2.0 * float(raw[offset + 9] > 0.7)], dtype=np.float32)
        return np.concatenate([pos, quat, grip])

    return np.concatenate([arm(0), arm(10)]).astype(np.float32)


@register_action_fn("ee6d_to_axis_angle")
def ee6d_to_axis_angle(actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    """ee6d chunk [H, >=10] -> axis-angle 7D [H, 7] (LIBERO / ManiSkill).

    pos(3) + rot6d(6) + grip(1) -> pos(3) + axis_angle(3) + grip(1);
    gripper binarized > 0.5 -> +1.0 else -1.0 (LIBERO convention).
    """
    pos = actions[:, :3]
    axis_angle = rot6d_to_axis_angle(actions[:, 3:9])
    grip = np.where(actions[:, 9:10] > 0.5, 1.0, -1.0)
    return np.concatenate([pos, axis_angle, grip], axis=-1).astype(np.float32)


@register_action_fn("ee6d_to_euler")
def ee6d_to_euler(actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    """ee6d chunk [H, >=10] -> Euler 7D [H, 7]; gripper > 0.25 -> +1 else -1."""
    from scipy.spatial.transform import Rotation

    pos = actions[:, :3]
    euler = Rotation.from_rotvec(rot6d_to_axis_angle(actions[:, 3:9])).as_euler("xyz")
    grip = np.where(actions[:, 9:10] > 0.25, 1.0, -1.0)
    return np.concatenate([pos, euler, grip], axis=-1).astype(np.float32)


@register_action_fn("ee6d_to_quat")
def ee6d_to_quat(actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    """ee6d chunk [H, >=10] -> quaternion 8D [H, 8]; gripper > 0.5 -> +1 else -1."""
    from scipy.spatial.transform import Rotation

    pos = actions[:, :3]
    quat = Rotation.from_rotvec(rot6d_to_axis_angle(actions[:, 3:9])).as_quat()
    grip = np.where(actions[:, 9:10] > 0.5, 1.0, -1.0)
    return np.concatenate([pos, quat, grip], axis=-1).astype(np.float32)


@register_action_fn("ee6d_to_calvin_abs")
def ee6d_to_calvin_abs(actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    """ee6d chunk [H, >=10] -> CALVIN absolute pose 8D [H, 8].

    Matches the official calvin client: pos(3) +
    rot6d_interleaved->quat(4) + (grip < 0.8 -> +1 else -1).
    """
    pos = actions[:, :3]
    quat = rot6d_interleaved_to_quat(actions[:, 3:9])
    grip = np.where(actions[:, 9:10] < 0.8, 1.0, -1.0)
    return np.concatenate([pos, quat, grip], axis=-1).astype(np.float32)


@register_action_fn("ee6d_to_simpler_abs_euler")
def ee6d_to_simpler_abs_euler(actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    """ee6d chunk [H, >=10] -> SimplerEnv WidowX 7D [H, 7].

    Matches the official simpler WidowX client: pos(3) +
    (rot6d_interleaved->euler_xyz + [0, pi/2, 0])(3) + (grip < 0.91 -> +1 else -1).
    """
    from scipy.spatial.transform import Rotation

    pos = actions[:, :3]
    rot6d = np.asarray(actions[:, 3:9], dtype=np.float64)
    euler = Rotation.from_matrix(rot6d_interleaved_to_matrix(rot6d)).as_euler("xyz")
    euler = euler + np.array([0.0, math.pi / 2.0, 0.0])
    grip = np.where(actions[:, 9:10] < 0.91, 1.0, -1.0)
    return np.concatenate([pos, euler, grip], axis=-1).astype(np.float32)


@register_action_fn("ee6d_robotwin_ee_dual")
def ee6d_robotwin_ee_dual(actions: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    """RoboTwin dual-arm: 20D ee6d chunk -> 16D env ee chunk (stateless).

    Per arm: xyz(3) + rot6d_interleaved->quat(4) + (1 - 2*(grip > 0.7))(1),
    executed via ``TASK_ENV.take_action(action, action_type='ee')``. The
    closed-loop endpose backfill lives in ``XVLAPayloadBuilder`` via
    ``note_env_action`` — not here.
    """
    chunk = np.asarray(actions, dtype=np.float32).reshape(-1, actions.shape[-1])
    if chunk.shape[-1] < ROBOTWIN_EE6D_ACTION_DIM:
        raise ValueError(
            f"ee6d_dual decoder requires {ROBOTWIN_EE6D_ACTION_DIM}D actions, got {chunk.shape[-1]}D"
        )
    return np.stack(
        [_robotwin_ee6d_row_to_ee(row[:ROBOTWIN_EE6D_ACTION_DIM]) for row in chunk]
    ).astype(np.float32)
