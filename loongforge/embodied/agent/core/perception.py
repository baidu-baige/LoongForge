# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pixel-to-world geometry from a camera's metric depth and calibration."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from loongforge.embodied.agent.core.types import CameraFrame


def pixel_to_world(
    frame: CameraFrame, pixels: Sequence[Sequence[int]], neighborhood: int = 1
) -> List[Optional[List[float]]]:
    """Back-project ``[row, col]`` pixels of an upright frame to world points (metres).

    Each pixel uses the median valid depth of its ``(2n+1)^2`` window, so a
    point on an object edge does not pick up the background. A pixel outside
    the image or without valid depth maps to None. The result is a point on the
    visible surface, not an object's centre.
    """
    if frame.depth is None or frame.intrinsic is None or frame.cam2world is None:
        raise ValueError("this frame has no depth or calibration; observe with depth=True")
    depth = frame.depth
    height, width = depth.shape
    fx, fy = frame.intrinsic[0, 0], frame.intrinsic[1, 1]
    cx, cy = frame.intrinsic[0, 2], frame.intrinsic[1, 2]
    points: List[Optional[List[float]]] = []
    for row, col in pixels:
        row, col = int(row), int(col)
        if not (0 <= row < height and 0 <= col < width):
            points.append(None)
            continue
        window = depth[
            max(0, row - neighborhood) : row + neighborhood + 1, max(0, col - neighborhood) : col + neighborhood + 1
        ]
        valid = window[np.isfinite(window) & (window > 0)]
        if valid.size == 0:
            points.append(None)
            continue
        z = float(np.median(valid))
        cam = np.array([(col - cx) * z / fx, (row - cy) * z / fy, z, 1.0])
        points.append([round(float(v), 4) for v in (frame.cam2world @ cam)[:3]])
    return points


def world_to_pixel(frame: CameraFrame, point: Sequence[float]) -> Optional[List[int]]:
    """Project a world point into an upright frame as ``[row, col]``; None if behind the camera."""
    if frame.intrinsic is None or frame.cam2world is None:
        raise ValueError("this frame has no calibration; observe with depth=True")
    cam = np.linalg.inv(frame.cam2world) @ np.array([*point, 1.0])
    if cam[2] <= 0:
        return None
    col = frame.intrinsic[0, 0] * cam[0] / cam[2] + frame.intrinsic[0, 2]
    row = frame.intrinsic[1, 1] * cam[1] / cam[2] + frame.intrinsic[1, 2]
    return [int(round(row)), int(round(col))]
