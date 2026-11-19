# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Camera trajectory conventions shared by Path B and its optimizer bridge."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def validate_trajectory(c2w, n):
    poses = np.asarray(c2w, dtype=np.float64)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("camera trajectory must be finite (frames,4,4)")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("invalid homogeneous camera poses")
    r = poses[:, :3, :3]
    if (not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-4)):
        raise ValueError("camera rotations must be proper orthonormal matrices")
    return poses


def interpolate_c2w(poses, indices, n, max_gap=30):
    """Rebase to the first camera and interpolate registered video views.

    Missing endpoints and long unregistered spans fail explicitly. They must
    not silently become a stationary camera or an extrapolated trajectory.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if (not len(indices) or indices[0] != 0 or indices[-1] != n - 1
            or np.any(np.diff(indices) <= 0)):
        raise ValueError("registered cameras must span the full video")
    if np.any(np.diff(indices) > max_gap):
        raise ValueError("camera registration has an unsupported long gap")
    poses = validate_trajectory(poses, len(indices))
    relative = np.linalg.inv(poses[0]) @ poses
    full = np.tile(np.eye(4), (n, 1, 1))
    if len(indices) == 1:
        return full.astype(np.float32)
    full[:, :3, :3] = Slerp(indices, Rotation.from_matrix(relative[:, :3, :3]))(
        np.arange(n)).as_matrix()
    for axis in range(3):
        full[:, axis, 3] = np.interp(np.arange(n), indices, relative[:, axis, 3])
    return validate_trajectory(full, n).astype(np.float32)


def complete_c2w(poses, indices, n, max_gap=30):
    """Return solver poses and an explicit validity mask for partial tracking.

    Short internal gaps are interpolated. Unlocalized ends and long internal
    gaps are invalid. Finite placeholders there are for the optimizer only;
    exporters must replace them with NaN and invalidate the hand observations.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if (not len(indices) or indices[0] < 0 or indices[-1] >= n
            or np.any(np.diff(indices) <= 0)):
        raise ValueError("invalid registered camera indices")
    poses = validate_trajectory(poses, len(indices))
    if indices[0] == 0 and indices[-1] == n - 1 and np.all(np.diff(indices) <= max_gap):
        return interpolate_c2w(poses, indices, n, max_gap), np.ones(n, bool)
    relative = np.linalg.inv(poses[0]) @ poses
    full = np.tile(np.eye(4), (n, 1, 1))
    valid = np.zeros(n, bool)
    valid[indices] = True
    if len(indices) > 1:
        query = np.clip(np.arange(n), indices[0], indices[-1])
        full[:, :3, :3] = Slerp(indices, Rotation.from_matrix(relative[:, :3, :3]))(query).as_matrix()
        for axis in range(3):
            full[:, axis, 3] = np.interp(query, indices, relative[:, axis, 3])
        for left, right in zip(indices[:-1], indices[1:]):
            if right - left <= max_gap:
                valid[left:right + 1] = True
    return validate_trajectory(full, n).astype(np.float32), valid
