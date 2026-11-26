# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Original calibrated camera in the gravity-aligned and mounted robot frames."""

import numpy as np
from scipy.spatial.transform import Rotation


def world_to_arm(points, gravity_wxyz, world_offset, wrist_anchor, workspace):
    up = Rotation.from_quat(np.asarray(gravity_wxyz)[[1, 2, 3, 0]])
    yaw = Rotation.from_euler("z", workspace[3], degrees=True)
    sim = up.apply(np.asarray(points).reshape(-1, 3)) - world_offset
    return (yaw.apply(sim - wrist_anchor) + np.asarray(workspace[:3])).reshape(np.asarray(points).shape)


def camera_in_arm(head_pose, gravity_wxyz, world_offset, wrist_anchor, workspace):
    up = Rotation.from_quat(np.asarray(gravity_wxyz)[[1, 2, 3, 0]])
    yaw = Rotation.from_euler("z", workspace[3], degrees=True)
    cv_rotation = (yaw * up * Rotation.from_quat(np.asarray(head_pose)[[4, 5, 6, 3]])).as_matrix()
    pos = world_to_arm(np.asarray(head_pose[:3]), gravity_wxyz, world_offset, wrist_anchor, workspace)
    mj_rotation = cv_rotation @ np.diag([1.0, -1.0, -1.0])
    return pos, Rotation.from_matrix(mj_rotation).as_quat()[[3, 0, 1, 2]], cv_rotation


def project(points, position, cv_rotation, K):
    camera = (np.asarray(points) - position) @ cv_rotation
    pixels = camera @ np.asarray(K).T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]


def render_to_calibrated_pixels(K, image_shape, focal):
    """MuJoCo raster pixel centers are indexed at half a pixel from viewport edges."""
    h, w = image_shape
    render_K = np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1.0]])
    return np.asarray(K) @ np.linalg.inv(render_K)
