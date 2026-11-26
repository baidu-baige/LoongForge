# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Match MuJoCo camera output to the input image's full pinhole calibration."""

import cv2
import numpy as np


def covering_focal(K, image_shape):
    """Choose a symmetric render frustum containing every calibrated image ray."""
    h, w = image_shape
    corners = np.array([[0.0, w, 0.0, w], [0.0, 0.0, h, h], [1.0, 1.0, 1.0, 1.0]])
    rays = np.linalg.solve(np.asarray(K, dtype=float), corners)
    rays /= rays[2]
    return float(min(w / (2 * np.max(np.abs(rays[0]))), h / (2 * np.max(np.abs(rays[1])))))


def camera_homography(K, image_shape, render_shape, source_fy):
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Finite camera intrinsics with positive focal lengths required")
    h, w = image_shape
    rh, rw = render_shape
    scale = np.diag([rw / w, rh / h, 1.0])
    source = np.array([[source_fy, 0, w / 2], [0, source_fy, h / 2], [0, 0, 1.0]])
    return scale @ K @ np.linalg.inv(source) @ np.linalg.inv(scale)


def warp_camera(array, homography, categorical=False):
    dtype = array.dtype
    src = array.astype(np.uint8) if dtype == np.bool_ else array
    result = cv2.warpPerspective(
        src,
        homography,
        (array.shape[1], array.shape[0]),
        flags=cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return result.astype(bool) if dtype == np.bool_ else result
