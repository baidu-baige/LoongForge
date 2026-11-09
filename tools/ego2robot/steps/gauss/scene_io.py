# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode data access for the gauss reconstruction step.

All frame indices are the **raw zarr indices** (0..n-1), identical to what the
retarget step consumes through ``retarget.episode.load_episode_world``, so the
reconstructed scene depth lines up with the rendered robot depth frame by
frame.  The ``01_load`` head-relative outputs are intentionally *not* used.
"""

from __future__ import annotations

import os
import re
import subprocess as sp
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Same fallback as the retarget renderer (retarget/cli.py and episode.py).
DEFAULT_FY = 490.1961
# Legacy EgoDex bg.mp4 carries 4 px H.264 padding top and bottom (368 -> 360).
BG_H, BG_W = 368, 640
CROP_TOP = 4
CROP_BOTTOM = 4
SCENE_H, SCENE_W = 360, 640


def resolve_n_frames(state_dir: Optional[str], zarr_dir: str, ep: str) -> int:
    """Number of aligned frames for ``ep`` (matches the retarget step's N).

    The retarget step uses ``N = len(state)`` where ``state`` comes from the
    Step 2 alignment output ``{state_dir}/{ep}.npz``; duplicate that logic so
    every array we produce is frame-aligned with the rendered robot frames.
    When no state file is available (standalone gauss-recon before align),
    fall back to the raw zarr frame count.
    """
    if state_dir:
        state_path = Path(state_dir) / f"{ep}.npz"
        if state_path.exists():
            with np.load(state_path, allow_pickle=False) as z:
                for key in z.files:
                    arr = z[key]
                    if arr.ndim >= 1 and arr.shape[0] > 0:
                        return int(arr.shape[0])
    import zarr
    z = zarr.open_group(str(Path(zarr_dir) / ep), mode="r")
    n_img = int(z["images.front_1"].shape[0])
    n_pose = int(z["obs_head_pose"].shape[0]) if "obs_head_pose" in z else n_img
    return min(n_img, n_pose)


def read_head_poses(zarr_dir: str, ep: str, n: int) -> np.ndarray:
    """Read the SLAM camera poses (n, 7) = [t xyz, q wxyz] camera-to-world.

    Same source and indexing as retarget's ``load_episode_world``.
    """
    import zarr
    z = zarr.open_group(str(Path(zarr_dir) / ep), mode="r")
    return np.asarray(z["obs_head_pose"][:n], dtype=np.float64)


def read_frames_sampled(
    zarr_dir: str,
    ep: str,
    n: int,
    sample_fps: Optional[float] = 2.0,
    fps: float = 30.0,
) -> Tuple[List, np.ndarray]:
    """Decode JPEG frames from the zarr and subsample to ~``sample_fps``.

    Returns ``(frames, indices)`` where ``frames`` is a list of PIL images and
    ``indices`` are the raw zarr frame indices of each sampled frame.  With
    ``sample_fps=None`` every frame is kept.
    """
    import zarr
    from PIL import Image

    z = zarr.open_group(str(Path(zarr_dir) / ep), mode="r")
    total = int(z["images.front_1"].shape[0])
    n = min(n, total)
    if sample_fps is None or sample_fps <= 0:
        stride = 1
    else:
        src_fps = float(fps) if fps and fps > 0 else 30.0
        stride = max(1, int(round(src_fps / float(sample_fps))))
    indices = np.arange(0, n, stride, dtype=np.int64)
    blobs = read_vlen_frames(str(Path(zarr_dir) / ep), "images.front_1", indices)
    frames = [Image.fromarray(_decode_jpeg(b)) for b in blobs]
    return frames, indices


def read_vlen_frames(ep_dir: str, arr_name: str, indices) -> List[bytes]:
    """Read JPEG blobs from a zarr v3 ``sharding_indexed`` + ``vlen-bytes`` array.

    zarr-python 3.4.x mis-decodes variable-length bytes under sharding (returns
    an 8-byte index entry instead of the element payload), so this reader walks
    the shard layout by hand:

      shard file = [inner-chunk zstd streams...]
                   [index: uint64 LE (n_outer, 2) = (offset, length) per chunk]
                   [crc32c (4 bytes)]        <- ignored (read-only trust)

    Each inner chunk decodes as a numcodecs ``VLenBytes`` frame (8-byte length
    prefix + payload).  Works for any ``vlen-bytes`` array produced by zarr >= 3.
    """
    import json
    from numcodecs.vlen import VLenBytes
    import zstandard

    arr_dir = Path(ep_dir) / arr_name
    meta = json.loads((arr_dir / "zarr.json").read_text())
    n_outer = int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
    index_size = 16 * n_outer + 4  # (offset, length) uint64 pairs + crc32c
    vlen = VLenBytes()

    # Map frame index -> (shard_file, inner_chunk_index).
    # Shard files are named by the chunk-grid index: c/0, c/1, ... (array is 1-D
    # with the default "/" separator).  Missing files surface as FileNotFoundError
    # below, which is fine -- every shard frame index must exist.
    def _shard_file(shard_i: int) -> Path:
        return arr_dir / "c" / str(shard_i)

    out: List[bytes] = [b""] * len(indices)
    pos = {int(i): k for k, i in enumerate(indices)}
    cache = {}
    for frame_i in sorted(pos):
        shard_i, inner = divmod(frame_i, n_outer)
        if shard_i not in cache:
            raw = _shard_file(shard_i).read_bytes()
            idx = np.frombuffer(raw[-index_size:-4], dtype="<u8").reshape(-1, 2)
            cache[shard_i] = (raw, idx)
        raw, idx = cache[shard_i]
        off, ln = int(idx[inner][0]), int(idx[inner][1])
        decoded = zstandard.decompress(raw[off:off + ln])
        objs = vlen.decode(decoded)
        out[pos[frame_i]] = bytes(objs[0]) if len(objs) else b""
    return out


def _decode_jpeg(blob: bytes) -> np.ndarray:
    """Decode a JPEG blob to RGB (simplejpeg -> OpenCV -> PIL)."""
    import simplejpeg
    try:
        return simplejpeg.decode_jpeg(blob, colorspace="RGB")
    except Exception:
        pass
    try:
        import cv2
        arr = np.frombuffer(blob, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    from io import BytesIO
    from PIL import Image
    return np.asarray(Image.open(BytesIO(blob)).convert("RGB"))


def resolve_fy(zarr_dir: str, ep: str, n: int, fy_cli: Optional[float]) -> float:
    """Effective vertical focal length for this episode (pixels).

    Resolution order (mirrors retarget):
      1. explicit ``--fy``
      2. zarr attrs intrinsics (``fallback_episode_attrs`` borrows from sibling
         episodes when missing)
      3. ``DEFAULT_FY``
    """
    if fy_cli is not None and float(fy_cli) > 0:
        return float(fy_cli)
    from .. import config
    ep_attrs, _ = config.fallback_episode_attrs(str(Path(zarr_dir) / ep))
    fy = config.dataset_fy_for_height(ep_attrs or {}, SCENE_H)
    return float(fy) if fy is not None else DEFAULT_FY


def bg_effective_size(bg_path: str) -> Tuple[int, int, int, int]:
    """(scene_h, scene_w, crop_top, crop_bottom) of an inpainted bg.mp4.

    Mirrors ``depth_estimate.read_video_frames``: a legacy 368x640 video owns
    4 px of H.264 padding top and bottom, so its effective scene is 360x640.
    Every other source (e.g. 640x480 Path B) is used at native resolution.
    """
    probe = sp.run(
        ["ffmpeg", "-hide_banner", "-i", str(bg_path), "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", probe.stderr or "")
    if m:
        v_w, v_h = int(m.group(1)), int(m.group(2))
    else:
        v_w, v_h = BG_W, BG_H
    if v_h == BG_H and CROP_TOP > 0:
        return SCENE_H, SCENE_W, CROP_TOP, CROP_BOTTOM
    return v_h, v_w, 0, 0


def make_k(fy: float, width: int, height: int) -> np.ndarray:
    """The exact pinhole model used by the retarget renderer.

    MuJoCo cameras have a single fovy and a centered principal point, matching
    ``K = [[fy,0,w/2],[0,fy,h/2],[0,0,1]]`` with
    ``fovy = 2*atan((h/2)/fy)`` (episode.py).  Rendering the gauss depth with
    this K makes the depth grid pixel-identical to the robot depth grid.
    """
    return np.array([[fy, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)