# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pure 6D-rotation math (leaf module, no eval-package imports).

Dependency-free (numpy + scipy only) so both the ee6d ActionDecoders and the
RoboTwin adapter can import it without any circular-import risk. Kept separate
from ``base.py`` (the ActionDecoder abstraction + registry) because rotation
math is an unrelated concern shared across components.
"""

from __future__ import annotations

import numpy as np


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """Convert concatenated 6D rotation to axis-angle. Input: [N, 6], Output: [N, 3].

    Concatenated layout [col0(3), col1(3)] (LIBERO ``Mat_to_Rotate6D``).
    """
    from scipy.spatial.transform import Rotation

    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    # Gram-Schmidt orthonormalization
    eps = 1e-8
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + eps)
    b3 = np.cross(b1, b2)

    rot_mat = np.stack([b1, b2, b3], axis=-1)  # [N, 3, 3]
    return Rotation.from_matrix(rot_mat).as_rotvec().astype(np.float32)


def rot6d_interleaved_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert interleaved 6D rotation to matrices. Input: [N, 6], Output: [N, 3, 3].

    Interleaved layout (official X-VLA calvin/simpler/robotwin clients):
    ``mat[:, :2].reshape(6)`` = [R00, R01, R10, R11, R20, R21] — even indices
    form the first column, odd indices the second. Differs from the
    concatenated layout used by the LIBERO client.
    """
    a1 = rot6d[:, 0:5:2]
    a2 = rot6d[:, 1:6:2]

    eps = 1e-8
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + eps)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # [N, 3, 3]


def rot6d_interleaved_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """Convert interleaved 6D rotation to xyzw quaternions. Input: [N, 6], Output: [N, 4]."""
    from scipy.spatial.transform import Rotation

    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    return Rotation.from_matrix(rot6d_interleaved_to_matrix(rot6d)).as_quat().astype(np.float32)
