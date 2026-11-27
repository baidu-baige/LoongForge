# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Rotation conversions. Quaternions are (x, y, z, w); RPY is fixed-axis roll, pitch, yaw about x, y, z."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_GIMBAL_LOCK_TOL = 1e-9


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rpy_to_matrix`; at pitch of +-90 degrees yaw takes the whole turn."""
    sp = min(1.0, max(-1.0, -float(rot[2, 0])))
    if abs(sp) < 1.0 - _GIMBAL_LOCK_TOL:
        return np.array([math.atan2(rot[2, 1], rot[2, 2]), math.asin(sp), math.atan2(rot[1, 0], rot[0, 0])])
    return np.array([0.0, math.asin(sp), math.atan2(-rot[0, 1], rot[1, 1])])


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a unit (x, y, z, w) quaternion."""
    x, y, z, w = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion of a rotation matrix, with w >= 0."""
    m = np.asarray(rot, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        quat = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        quat = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        quat = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        quat = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    quat = np.array(quat)
    quat /= np.linalg.norm(quat)
    return -quat if quat[3] < 0 else quat


def rotation_error(quat_to: Sequence[float], quat_from: Sequence[float]) -> np.ndarray:
    """Shortest world-frame rotation (axis-angle vector) from ``quat_from`` to ``quat_to``."""
    delta = quat_to_matrix(quat_to) @ quat_to_matrix(quat_from).T
    x, y, z, w = matrix_to_quat(delta)
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(sin_half, w)
    return np.array([x, y, z]) / sin_half * angle
