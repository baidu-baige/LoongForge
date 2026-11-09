# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Output writers for the gauss reconstruction step."""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def write_scene_depth_npz(path, depth: np.ndarray, meta: Optional[Dict] = None,
                          vis: bool = True) -> None:
    """Write a ``scene_depth.npz`` with the DA3-compatible interface.

    ``depth`` is (N, H, W) float32 meters over the *effective* scene area.  The
    metadata keys mirror :func:`steps.retarget.rendering.load_scene_depth`
    (``height/width/crop_top/crop_bottom``) plus ``backend="gauss"`` so the
    retarget step can skip the DA3 scale/shift alignment.
    """
    meta = dict(meta or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth": np.asarray(depth, dtype=np.float16),
        "model": str(meta.get("model", "dvlt")),
        "unit": "meters",
        "backend": "gauss",
        "height": int(meta.get("height", depth.shape[1])),
        "width": int(meta.get("width", depth.shape[2])),
        "crop_top": int(meta.get("crop_top", 0)),
        "crop_bottom": int(meta.get("crop_bottom", 0)),
    }
    K = meta.get("K")
    if K is not None:
        payload["K"] = np.asarray(K, dtype=np.float64)
    np.savez_compressed(path, **payload)
    if vis:
        write_vis_video(path.with_name("depth_vis.mp4"),
                        np.asarray(depth, dtype=np.float32))


def write_dvlt_cache(path, payload: Dict) -> None:
    """Cache raw DVLT outputs and scale/alignment metadata for offline analysis."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_meta(path, meta: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_vis_video(path, depth_maps: np.ndarray, fps: float = 30.0) -> None:
    """Write (N, H, W) float depth -> turbo-colormap mp4 (matches DA3 style)."""
    from ..depth_estimate import colormap_turbo
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{depth_maps.shape[2]}x{depth_maps.shape[1]}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = sp.Popen(cmd, stdin=sp.PIPE)
    vmin = float(np.nanpercentile(depth_maps, 1))
    vmax = float(np.nanpercentile(depth_maps, 99))
    for d in depth_maps:
        norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
        rgb = colormap_turbo(norm)
        proc.stdin.write(np.nan_to_num(rgb, nan=0).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


def write_recon_vis(path, frames_rgb, depth_maps, step: int = 5) -> None:
    """Side-by-side (original | depth) mp4, sampled every ``step`` frames."""
    from ..depth_estimate import colormap_turbo
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{frames_rgb.shape[2] * 2}x{frames_rgb.shape[1]}", "-r", "15",
        "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = sp.Popen(cmd, stdin=sp.PIPE)
    vmin = float(np.nanpercentile(depth_maps, 1))
    vmax = float(np.nanpercentile(depth_maps, 99))
    for i in range(0, len(frames_rgb), step):
        d = depth_maps[i]
        norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
        vis = colormap_turbo(norm).astype(np.uint8)
        if vis.shape[:2] != frames_rgb.shape[1:3]:
            import cv2
            vis = cv2.resize(vis, (frames_rgb.shape[2], frames_rgb.shape[1]))
        side = np.concatenate([frames_rgb[i], vis], axis=1)
        proc.stdin.write(np.ascontiguousarray(side).tobytes())
    proc.stdin.close()
    proc.wait()