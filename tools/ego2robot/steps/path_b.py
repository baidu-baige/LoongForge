# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Path B: convert a video without hand annotations into Ego2Robot input.

The paper's Path B is ``video -> WiLoR -> DynHaMR -> retarget``.  The public
WiLoR checkpoint is distributed separately from its Python package, so this
step deliberately keeps model execution behind a small runner boundary.  The
default runner invokes the official WiLoR ``demo.py`` when a source checkout is
provided; ``--predictions`` is useful for validating the rest of the pipeline
offline and for rerunning temporal refinement without GPU inference.

The output is an EgoVerse-like Zarr episode.  This makes Path B a producer for
the existing ``load``, ``mask``, ``inpaint`` and ``retarget`` steps rather than
creating a second copy of the downstream pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

try:
    from .hand_camera import (
        assumed_camera, calibrate_predictions, camera_matrix, estimate_camera,
        stabilize_camera_estimate,
    )
    from . import camera_motion
    from .camera_geometry import validate_trajectory
except ImportError:  # Direct execution as python steps/path_b.py.
    from hand_camera import (
        assumed_camera, calibrate_predictions, camera_matrix, estimate_camera,
        stabilize_camera_estimate,
    )
    import camera_motion
    from camera_geometry import validate_trajectory

try:
    from zarr.codecs import VLenBytesCodec
except ImportError:  # pragma: no cover - zarr 2 compatibility
    VLenBytesCodec = None
try:
    from zarr.dtype import VariableLengthBytes
except ImportError:  # pragma: no cover - zarr 2 compatibility
    VariableLengthBytes = None


# The Path B protocol uses the OpenPose-compatible 21-point MANO layout used
# by WiLoR and Dyn-HaMR (wrist, thumb, index, middle, ring, pinky chains).
MANO_JOINTS = 21
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 0, 4, 8, 12

# Camera-frame geometry uses x right, y down, z forward, consistent with
# positive-focal OpenCV projection. The explicit static mode leaves this
# frame unchanged. Estimated c2w maps it into the first camera's world frame.
CAMERA_TO_WORLD = np.asarray(
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]], dtype=np.float32)
CAMERA_HEAD_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def read_video(path: str, max_frames: int = 0):
    """Read BGR frames and the source FPS without loading unbounded input."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while max_frames <= 0 or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    return frames, fps if np.isfinite(fps) and fps > 1e-3 else 30.0


def _first(mapping, names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _as_keypoints(value):
    """Normalize common WiLoR/DynHaMR keypoint layouts to ``(N,21,3)``."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[-1] == 63:
        arr = arr.reshape(-1, MANO_JOINTS, 3)
    elif arr.ndim == 3 and arr.shape[-2:] == (MANO_JOINTS, 3):
        pass
    elif arr.ndim == 3 and arr.shape[-2:] == (3, MANO_JOINTS):
        arr = arr.transpose(0, 2, 1)
    elif arr.ndim == 2 and arr.shape == (MANO_JOINTS, 3):
        arr = arr[None]
    else:
        return None
    return arr.astype(np.float32, copy=False)


def _load_prediction_file(path: Path):
    """Load one prediction file and return a list of detection dictionaries."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            raw = {k: data[k] for k in data.files}
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        return []

    keypoints = _first(raw, ("pred_keypoints_3d", "keypoints_3d", "joints_3d",
                             "pred_keypoints", "mano_joints", "keypoints"))
    kp = _as_keypoints(keypoints)
    if kp is None:
        return []
    n = len(kp)
    right = _first(raw, ("right", "is_right", "handedness"))
    if right is None:
        right = np.full(n, -1, dtype=np.int8)
    right = np.asarray(right).reshape(-1)
    if len(right) == 1 and n > 1:
        right = np.repeat(right, n)
    frame = _first(raw, ("frame", "frame_idx", "frame_index", "image_idx"))
    if frame is None:
        # WiLoR demo revisions differ: some write one NPZ per image and put
        # the index only in the filename, while others write one sequence NPZ.
        # Numeric stems preserve the former convention.
        try:
            frame = np.full(n, int(path.stem), dtype=np.int64)
        except ValueError:
            frame = np.arange(n, dtype=np.int64)
    frame = np.asarray(frame).reshape(-1)
    if len(frame) == 1 and n > 1:
        frame = np.repeat(frame, n)
    score = _first(raw, ("score", "scores", "confidence", "conf"))
    score = np.ones(n, dtype=np.float32) if score is None else np.asarray(score).reshape(-1)
    if len(score) == 1 and n > 1:
        score = np.repeat(score, n)
    detections = []
    for i in range(n):
        item = {"keypoints": kp[i], "right": right[i], "frame": int(frame[i]),
                "score": float(score[i])}
        extras = {k: np.asarray(raw[k])[i] for k in
                  ("keypoints_2d", "hand_pose", "global_orient", "betas", "cam_trans")
                  if k in raw}
        if extras:
            item["_wilor_extra"] = extras
        if "keypoints_2d" in extras:
            item["keypoints_2d"] = extras["keypoints_2d"]
        detections.append(item)
    return detections


def collect_predictions(root: Path):
    """Collect prediction files emitted by WiLoR/DynHaMR."""
    files = ([root] if root.is_file() else
             sorted(p for p in root.rglob("*") if p.suffix.lower() in {".npz", ".json"}))
    detections = []
    for path in files:
        detections.extend(_load_prediction_file(path))
    if not detections:
        raise RuntimeError(
            f"no 3D hand predictions found under {root}; expected NPZ/JSON with "
            "pred_keypoints_3d (or keypoints_3d/joints_3d)"
        )
    return detections


def save_rich_predictions(path, predictions):
    """Persist the unmodified video model output for controlled reruns."""
    extras = [item.get("_wilor_extra") for item in predictions]
    if not all(extra is not None for extra in extras):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, pred_keypoints_3d=np.stack([x["keypoints"] for x in predictions]),
        right=np.asarray([x["right"] for x in predictions]),
        frame=np.asarray([x["frame"] for x in predictions]),
        score=np.asarray([x["score"] for x in predictions]),
        **{key: np.asarray([extra[key] for extra in extras])
           for key in ("keypoints_2d", "hand_pose", "global_orient", "betas", "cam_trans")})


def run_wilor(frames, checkpoint: str, wilor_repo: str | None, detector: str | None,
              mano_dir: str | None, command: str | None, batch_size: int, work_dir: Path):
    """Run an official WiLoR checkout and collect its prediction files.

    ``--wilor_command`` can be used for checkout revisions whose CLI differs.
    Placeholders are ``{images}``, ``{output}``, ``{checkpoint}``, ``{detector}``,
    and ``{mano_dir}``.  The default matches the released WiLoR demo convention.
    """
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"WiLoR checkpoint not found: {checkpoint}")
    if not wilor_repo:
        raise RuntimeError(
            "WiLoR weights were found, but the WiLoR Python source is missing. "
            "Provide --wilor_repo pointing at the WiLoR checkout (the .ckpt "
            "file alone cannot be executed), or use --predictions for an "
            "offline protocol check."
        )
    repo = Path(wilor_repo)
    if not repo.is_dir():
        raise FileNotFoundError(f"WiLoR source directory not found: {repo}")
    detector_path = Path(detector) if detector else repo / "pretrained_models" / "detector.pt"
    if not detector_path.is_file():
        raise FileNotFoundError(
            f"WiLoR detector checkpoint not found: {detector_path}; "
            "pass --detector explicitly or place detector.pt in the checkout"
        )

    image_dir = work_dir / "frames"
    output_dir = work_dir / "wilor_output"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        if not cv2.imwrite(str(image_dir / f"{i:06d}.jpg"), frame):
            raise RuntimeError(f"failed to write temporary frame {i}")

    if command:
        formatted = command.format(images=str(image_dir), output=str(output_dir),
                                   checkpoint=str(checkpoint), detector=str(detector or ""),
                                   mano_dir=str(mano_dir or ""))
        cmd = shlex.split(formatted)
    else:
        adapter = Path(__file__).with_name("wilor_infer.py")
        cmd = [sys.executable, str(adapter), "--wilor_repo", str(repo),
               "--images", str(image_dir), "--output", str(output_dir),
               "--checkpoint", str(checkpoint), "--detector", str(detector_path),
               "--mano_dir", str(mano_dir or ""),
               "--batch_size", str(batch_size)]
    proc = subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True,
                          env={**__import__("os").environ, "PYTHONPATH": str(repo)})
    if proc.returncode:
        raise RuntimeError("WiLoR failed (last output):\n" +
                           (proc.stdout + "\n" + proc.stderr)[-6000:])
    detections = collect_predictions(output_dir)
    rich_path = output_dir / "wilor_predictions.npz"
    if rich_path.is_file():
        with np.load(rich_path, allow_pickle=True) as rich:
            rich_fields = {k: rich[k] for k in rich.files}
        for i, det in enumerate(detections):
            if i < len(rich_fields.get("hand_pose", ())):
                det["_wilor_extra"] = {
                    k: rich_fields[k][i] for k in
                    ("keypoints_2d", "hand_pose", "global_orient", "betas", "cam_trans")
                    if k in rich_fields
                }
    return detections


def apply_dynhamr(predictions, command: str | None, work_dir: Path, video: str | None = None):
    """Optionally run DynHaMR; otherwise use the deterministic smoother below."""
    if not command:
        return predictions
    source = work_dir / "wilor_predictions.npz"
    np.savez_compressed(source,
                        keypoints=np.stack([x["keypoints"] for x in predictions]),
                        right=np.asarray([x["right"] for x in predictions]),
                        frame=np.asarray([x["frame"] for x in predictions]),
                        score=np.asarray([x["score"] for x in predictions]))
    output = work_dir / "dynhamr_output"
    output.mkdir(exist_ok=True)
    formatted = command.format(input=str(source), output=str(output), video=str(video or ""))
    proc = subprocess.run(shlex.split(formatted), text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError("DynHaMR failed (last output):\n" +
                           (proc.stdout + "\n" + proc.stderr)[-6000:])
    return collect_predictions(output)


def apply_official_dynhamr(predictions, video: str, repo: str, mano_dir: str,
                           work_dir: Path, gpu: str, is_static: bool,
                           mean_params: str | None = None, image_size=None,
                           n_frames=None, output_dir=None, camera=None,
                           camera_trajectory=None):
    """Refine both observed hands with the official Dyn-HaMR optimizer."""
    extras = [x.get("_wilor_extra") for x in predictions]
    if not extras or any(x is None for x in extras):
        raise RuntimeError(
            "official Dyn-HaMR requires WiLoR MANO parameters; run Path B "
            "from video with --dynhamr_repo instead of --predictions"
        )
    source = work_dir / "wilor_predictions.npz"
    np.savez_compressed(source,
                        pred_keypoints_3d=np.stack([x["keypoints"] for x in predictions]),
                        right=np.asarray([x["right"] for x in predictions]),
                        frame=np.asarray([x["frame"] for x in predictions]),
                        score=np.asarray([x["score"] for x in predictions]),
                        **{k: np.asarray([x[k] for x in extras], np.float32)
                           for k in ("keypoints_2d", "hand_pose", "global_orient", "betas", "cam_trans")})
    dyn_output = Path(output_dir) if output_dir is not None else work_dir / "dynhamr_output"
    cmd = [sys.executable, str(Path(__file__).with_name("dynhamr_official.py")),
           "--repo", repo, "--video", video, "--output", str(dyn_output),
           "--predictions", str(source), "--mano_dir", mano_dir, "--gpu", str(gpu)]
    if mean_params:
        cmd.extend(["--mano_mean_params", mean_params])
    if n_frames is not None:
        cmd.extend(["--max_frames", str(n_frames)])
    if camera is not None:
        camera_file = work_dir / "video_camera.json"
        camera_file.write_text(json.dumps(camera))
        cmd.extend(["--video_camera", str(camera_file)])
    if camera_trajectory is not None:
        trajectory_file = work_dir / "video_camera_trajectory.npz"
        np.savez_compressed(trajectory_file, c2w=camera_trajectory,
                            source=np.asarray("video:COLMAP"), units=np.asarray("meters"))
        cmd.extend(["--video_camera_trajectory", str(trajectory_file)])
    if not is_static:
        cmd.append("--no-is_static")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError("official Dyn-HaMR failed (last output):\n" +
                           (proc.stdout + "\n" + proc.stderr)[-10000:])
    shutil.copyfile(source, dyn_output / "wilor_initialization.npz")
    refined = _load_prediction_file(dyn_output / "dynhamr_predictions.npz")
    if not refined:
        raise RuntimeError(f"Dyn-HaMR exported no predictions: {dyn_output}")
    if image_size is None:
        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    else:
        width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        width, height = 640, 368
    # The bridge explicitly exports camera-frame joints. Keep its optimized
    # translation; wrist alignment to WiLoR would undo the temporal refinement.
    source_by_key = {}
    for source in predictions:
        key = (bool(float(source["right"]) > 0.5), int(source["frame"]))
        if key not in source_by_key or source["score"] > source_by_key[key]["score"]:
            source_by_key[key] = source
    refined_sides = set()
    for item in refined:
        side = bool(float(item["right"]) > 0.5)
        refined_sides.add(side)
        source = source_by_key.get((side, int(item["frame"])))
        if source is not None:
            source_2d = source.get("_wilor_extra", {}).get("keypoints_2d")
            if source_2d is not None:
                item["keypoints_2d"] = np.asarray(source_2d, np.float32).reshape(21, 2)
    return refined + [x for x in predictions
                      if bool(float(x["right"]) > 0.5) not in refined_sides]


def _interpolate_short_gaps(arr, max_gap):
    """Fill only bounded short gaps; long gaps and sequence edges stay invalid."""
    valid = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))
    good = np.flatnonzero(valid)
    for left, right in zip(good[:-1], good[1:]):
        if 0 < right - left - 1 <= max_gap:
            weights = np.arange(1, right - left) / (right - left)
            weights = weights.reshape((-1,) + (1,) * (arr.ndim - 1))
            arr[left + 1:right] = (1 - weights) * arr[left] + weights * arr[right]
    return arr


def detections_to_2d_tracks(detections, n_frames, width, max_gap=5):
    """Collect optional full-image 2D observations in left/right tracks."""
    tracks = {side: np.full((n_frames, MANO_JOINTS, 2), np.nan, np.float32)
              for side in ("left", "right")}
    for det in sorted(detections, key=lambda d: (d["frame"], -d["score"])):
        t = int(det["frame"])
        value = det.get("keypoints_2d")
        if value is None:
            value = det.get("_wilor_extra", {}).get("keypoints_2d")
        if t < 0 or t >= n_frames or value is None or det["score"] <= 0:
            continue
        kp = np.asarray(value, dtype=np.float32).reshape(-1, 2)
        if kp.shape != (MANO_JOINTS, 2) or not np.isfinite(kp).all():
            continue
        side = "right" if _side(det["right"], np.pad(kp, ((0, 0), (0, 1))), width) else "left"
        if not np.isfinite(tracks[side][t]).all():
            tracks[side][t] = kp
    for arr in tracks.values():
        _interpolate_short_gaps(arr, max_gap)
    return tracks


def _side(value, kp, width):
    if isinstance(value, str):
        return value.lower().startswith(("r", "1"))
    if np.isfinite(value) and float(value) >= 0:
        return bool(float(value) > 0.5)
    return bool(float(np.mean(kp[:, 0])) >= width / 2.0)


def detections_to_tracks(detections, n_frames, width, max_gap=5):
    """Assign detections to stable tracks, accepting single-hand videos.

    Missing sides remain invalid. Downstream retargeting uses the confidence
    arrays to skip IK for a hand that was not observed.
    """
    tracks = {"left": np.full((n_frames, MANO_JOINTS, 3), np.nan, np.float32),
              "right": np.full((n_frames, MANO_JOINTS, 3), np.nan, np.float32)}
    scores = {"left": np.zeros(n_frames, np.float32), "right": np.zeros(n_frames, np.float32)}
    for det in sorted(detections, key=lambda d: (d["frame"], -d["score"])):
        t = det["frame"]
        if t < 0 or t >= n_frames or not np.isfinite(det["score"]) or det["score"] <= 0:
            continue
        kp = _as_keypoints(det["keypoints"])
        if kp is None or not np.isfinite(kp).all():
            continue
        kp = kp[0] if kp.ndim == 3 else kp
        side = "right" if _side(det["right"], kp, width) else "left"
        # Keep the highest confidence detection if a runner emits duplicates.
        if det["score"] >= scores[side][t]:
            tracks[side][t] = kp
            scores[side][t] = det["score"]

    valid_sides = []
    for side in tracks:
        arr = tracks[side]
        valid = np.isfinite(arr).all(axis=(1, 2))
        if valid.any():
            valid_sides.append(side)
            original_valid = valid.copy()
            tracks[side] = _interpolate_short_gaps(arr, max_gap)
            # Interpolated 3D observations are usable for temporal smoothing
            # and export. Keep them below the neighboring confidence so
            # downstream consumers can distinguish measured and filled frames.
            filled = np.isfinite(tracks[side]).all(axis=(1, 2)) & ~original_valid
            if np.any(filled):
                for t in np.flatnonzero(filled):
                    left = t - 1
                    while left >= 0 and not original_valid[left]:
                        left -= 1
                    right = t + 1
                    while right < n_frames and not original_valid[right]:
                        right += 1
                    neighbors = [scores[side][i] for i in (left, right)
                                 if 0 <= i < n_frames and original_valid[i]
                                 and scores[side][i] > 0]
                    if neighbors:
                        scores[side][t] = np.float32(0.5 * min(neighbors))

    if not valid_sides:
        raise RuntimeError("WiLoR produced no hand detections")
    return tracks, scores


def smooth_tracks(tracks, window=11, polyorder=3, scores=None,
                  tracks_2d=None, camera=None, c2w=None,
                  reprojection_gate_px=3.0):
    """Smooth finite tracks with an RGB-only reprojection guard.

    Savitzky-Golay is used only to propose a temporal solution. When full
    image 2D observations and a video camera are available, each proposed
    joint is accepted only if its reprojection remains consistent with the
    observed pixels. Lower-confidence observations receive more smoothing.
    """
    if window <= polyorder + 1 or window % 2 == 0:
        raise ValueError("smooth window must be odd and greater than polyorder")
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return tracks
    out = {}
    confidence = scores or {}
    k = None
    poses = None
    if tracks_2d is not None and camera is not None:
        k = camera_matrix(camera)
        poses = (np.asarray(c2w, dtype=np.float64)
                 if c2w is not None else np.tile(
                     np.eye(4), (len(next(iter(tracks.values()))), 1, 1)))
        if poses.shape[0] != len(next(iter(tracks.values()))):
            raise ValueError("c2w and hand tracks must have the same frame count")

    def reprojection(points, frame_ids):
        w2c = np.linalg.inv(poses[frame_ids])
        cam_points = (np.einsum("nij,nkj->nki", w2c[:, :3, :3], points)
                      + w2c[:, None, :3, 3])
        z = cam_points[..., 2]
        uv = np.full(cam_points.shape[:-1] + (2,), np.nan)
        valid = np.isfinite(cam_points).all(-1) & (z > .02)
        uv[valid] = (cam_points[valid] @ k.T)[:, :2] / z[valid, None]
        return uv

    for side, arr in tracks.items():
        out[side] = arr.copy()
        finite = np.isfinite(arr).all(axis=(1, 2))
        starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
        ends = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
        for start, end in zip(starts, ends):
            if end - start >= window:
                proposal = savgol_filter(
                    arr[start:end], window, polyorder, axis=0).astype(np.float32)
                if tracks_2d is None or side not in tracks_2d or k is None:
                    out[side][start:end] = proposal
                    continue
                frame_ids = np.arange(start, end)
                observed_uv = np.asarray(tracks_2d[side][start:end], dtype=np.float64)
                original_error = np.linalg.norm(
                    reprojection(arr[start:end], frame_ids) - observed_uv, axis=-1)
                proposed_error = np.linalg.norm(
                    reprojection(proposal, frame_ids) - observed_uv, axis=-1)
                valid = (np.isfinite(observed_uv).all(-1)
                         & np.isfinite(original_error)
                         & np.isfinite(proposed_error))
                # Keep the temporal proposal only where it remains compatible
                # with the RGB observation. The absolute cap prevents a large
                # 3D change from hiding behind a relative improvement.
                accept = valid & (proposed_error <= np.minimum(
                    original_error + float(reprojection_gate_px), 8.))
                if side in confidence:
                    score = np.asarray(confidence[side][start:end], dtype=float)
                    alpha = np.where(
                        score[..., None, None] < .5, 1.0,
                        np.clip(1.0 - score[..., None, None], .15, .75))
                else:
                    alpha = .35
                blended = arr[start:end] * (1. - alpha) + proposal * alpha
                out[side][start:end] = np.where(
                    accept[..., None], blended, arr[start:end]).astype(np.float32)
    return out


def _quat_identity(n):
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1.0
    return q


def _tcp_pose(kp, hand_sign):
    """Paper Eq. (1)-(3), returning ``pose7`` and gripper width."""
    vf = 0.7 * kp[:, INDEX_TIP] + 0.3 * kp[:, MIDDLE_TIP]
    thumb = kp[:, THUMB_TIP]
    wrist = kp[:, WRIST]
    p = 0.5 * (thumb + vf)
    width = np.linalg.norm(thumb - vf, axis=1)
    z = hand_sign * (thumb - vf) / np.maximum(width[:, None], 1e-8)
    d = vf - wrist
    y = np.cross(z, d)
    yn = np.linalg.norm(y, axis=1)
    bad = yn < 1e-6
    y /= np.maximum(yn[:, None], 1e-8)
    if np.any(bad):
        ref = np.tile([0.0, 0.0, 1.0], (len(kp), 1))
        ref[np.abs(np.sum(z * ref, axis=1)) > 0.99] = [0.0, 1.0, 0.0]
        y[bad] = np.cross(z[bad], ref[bad])
        y[bad] /= np.maximum(np.linalg.norm(y[bad], axis=1, keepdims=True), 1e-8)
    x = np.cross(y, z)
    matrices = np.stack([x, y, z], axis=2)
    valid = np.isfinite(matrices).all((1, 2)) & (width > 1e-8)
    quat = np.full((len(kp), 4), np.nan, np.float32)
    if valid.any():
        quat[valid] = Rotation.from_matrix(matrices[valid]).as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(quat)):
        if valid[i] and valid[i - 1] and np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    p[~valid] = np.nan
    return np.concatenate([p, quat], axis=1).astype(np.float32), width.astype(np.float32)


def _vlen_array(group, name, values):
    data = np.asarray(values, dtype=object)
    if VariableLengthBytes is not None:
        # Zarr 3 requires an explicit variable-length byte dtype for object
        # arrays. Create the array first, then assign data separately.
        array = group.create_array(name, shape=data.shape,
                                   dtype=VariableLengthBytes())
        array[:] = data
        return array
    if VLenBytesCodec is not None:  # pragma: no cover - zarr 2 compatibility
        return group.create_array(name, data=data, serializer=VLenBytesCodec())
    return group.create_array(name, data=data)


def transform_tracks(tracks, c2w, inverse=False):
    """Transform full hand tracks; NaN observations remain invalid."""
    poses = validate_trajectory(c2w, len(next(iter(tracks.values()))))
    if inverse:
        poses = np.linalg.inv(poses)
    return {side: (np.einsum("nij,nkj->nki", poses[:, :3, :3], arr)
                   + poses[:, None, :3, 3]).astype(np.float32)
            for side, arr in tracks.items()}


def write_episode(output_dir: Path, episode: str, frames, fps, tracks, scores,
                  tracks_2d=None, metadata=None, camera=None, c2w=None,
                  camera_valid=None, camera_observed=None):
    """Write the canonical Zarr episode consumed by the existing pipeline."""
    ep_dir = output_dir / episode
    ep_dir.mkdir(parents=True, exist_ok=True)
    n, h, w = len(frames), frames[0].shape[0], frames[0].shape[1]
    # Keep keypoints, wrist poses, and TCP poses in the same synthetic world
    # frame. Applying the transform before TCP construction also rotates the
    # hand orientation consistently with the rendered camera.
    world_tracks = {
        side: np.einsum("ij,nkj->nki", CAMERA_TO_WORLD,
                        np.asarray(tracks[side], dtype=np.float32))
        for side in ("left", "right")
    }
    if c2w is not None:
        world_tracks = transform_tracks(tracks, c2w)
    if camera_valid is not None:
        camera_valid = np.asarray(camera_valid, dtype=bool)
        if camera_valid.shape != (n,):
            raise ValueError("camera_valid must have one value per video frame")
        for side in world_tracks:
            world_tracks[side][~camera_valid] = np.nan
            scores[side][~camera_valid] = 0
    poses = {}
    for side, sign in [("left", -1.), ("right", 1.)]:
        poses[side], _ = _tcp_pose(world_tracks[side], sign)
        valid = (np.isfinite(poses[side]).all(1)
                 & np.isfinite(world_tracks[side]).all((1, 2)) & (scores[side] > 0))
        scores[side] = np.where(valid, scores[side], 0).astype(np.float32)
        poses[side][~valid] = np.nan
    observed_tracks = {side: world_tracks[side].copy() for side in world_tracks}
    for side in observed_tracks:
        observed_tracks[side][scores[side] <= 0.0] = np.nan
    left_kp = observed_tracks["left"].reshape(n, -1).astype(np.float32)
    right_kp = observed_tracks["right"].reshape(n, -1).astype(np.float32)
    left_ee, right_ee = poses["left"], poses["right"]
    left_wrist = np.concatenate([observed_tracks["left"][:, WRIST], _quat_identity(n)], 1)
    right_wrist = np.concatenate([observed_tracks["right"][:, WRIST], _quat_identity(n)], 1)
    left_wrist[scores["left"] <= 0] = np.nan
    right_wrist[scores["right"] <= 0] = np.nan
    camera = camera or assumed_camera(w, h)
    head = np.zeros((n, 7), np.float32)
    head[:, 3:7] = CAMERA_HEAD_QUAT
    if c2w is not None:
        head[:, :3] = c2w[:, :3, 3]
        head[:, 3:] = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        for i in range(1, n):
            if np.dot(head[i - 1, 3:], head[i, 3:]) < 0:
                head[i, 3:] *= -1
    if camera_valid is not None:
        head[~camera_valid] = np.nan
    attrs = {"total_frames": n, "fps": float(fps), "source": "Ego2Robot Path B",
             "path": "B", "coordinate_frame": "camera",
             "intrinsics": {k: camera[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h")},
             "camera_intrinsics_estimated": camera["estimated"],
             "intrinsics_source": camera["source"],
             "video_camera_estimate": camera,
             "camera_pose_source": "assumed:static",
             "camera_pose_estimated": c2w is not None}
    if c2w is not None:
        attrs.update(coordinate_frame="first_camera_world",
                     camera_pose_source="video:COLMAP", camera_translation_units="meters",
                     camera_translation_scale_estimated=True)
        reference = camera.get("motion_estimate", {}).get("camera_reference_frame_index", 0)
        attrs["camera_reference_frame_index"] = reference
        if reference != 0:
            attrs["coordinate_frame"] = "first_valid_camera_world"
    if metadata:
        attrs.update(metadata)
    group = zarr.open_group(str(ep_dir), mode="w", zarr_format=3)
    group.attrs.update(attrs)
    arrays = {
        "left.obs_keypoints": left_kp, "right.obs_keypoints": right_kp,
        "left.obs_wrist_pose": left_wrist,
        "right.obs_wrist_pose": right_wrist,
        "left.obs_ee_pose": left_ee, "right.obs_ee_pose": right_ee,
        "obs_head_pose": head,
        "obs_rgb_timestamps_ns": (np.arange(n) * (1e9 / fps)).astype(np.int64),
        "path_b_left_confidence": scores["left"], "path_b_right_confidence": scores["right"],
        "path_b_left_observed": scores["left"] > 0, "path_b_right_observed": scores["right"] > 0,
    }
    if camera_valid is not None:
        arrays["path_b_camera_pose_valid"] = camera_valid
    if camera_observed is not None:
        arrays["path_b_camera_pose_observed"] = np.asarray(camera_observed, dtype=bool)
    if tracks_2d is not None:
        for side in ("left", "right"):
            observed_2d = np.asarray(tracks_2d[side], dtype=np.float32).reshape(n, -1).copy()
            observed_2d[scores[side] <= 0] = np.nan
            arrays[f"{side}.obs_keypoints_2d"] = observed_2d
    for name, value in arrays.items():
        group.create_array(name, data=value, chunks="auto")
    encoded = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("failed to encode a video frame as JPEG")
        encoded.append(bytes(buf))
    _vlen_array(group, "images.front_1", encoded)
    annotations = [json.dumps({"task": "Path B video", "frame": i}).encode()
                   for i in range(n)]
    _vlen_array(group, "annotations", annotations)
    return ep_dir


def process_video(args):
    metric_backend = getattr(args, "metric_depth_backend", "unidepth")
    self_calibrate = getattr(args, "camera_self_calibration", False)
    if self_calibrate:
        if args.camera_motion != "colmap" or args.camera_estimator != "geocalib":
            raise ValueError("video self-calibration needs GeoCalib and COLMAP motion")
        if args.camera_focal_prior_max_deviation >= 0:
            raise ValueError("disable the legacy focal prior for video self-calibration")
        if args.camera_fit != "pnp":
            raise ValueError("video self-calibration currently requires --camera_fit pnp")
    hand_depth_stats = {"enabled": False, "reference_data_used": False}
    if args.hand_depth_alignment:
        camera_motion.validate_hand_depth_options(
            args.hand_depth_stride, args.hand_depth_patch_size,
            args.hand_depth_max_mad, args.hand_depth_max_delta)
    if getattr(args, "camera_motion", "colmap") == "colmap":
        required_model = (getattr(args, "unidepth_model", None)
                          if metric_backend == "unidepth"
                          else getattr(args, "da3_metric_model", None))
        if not required_model:
            flag = "--unidepth_model" if metric_backend == "unidepth" else "--da3_metric_model"
            raise ValueError(
                f"camera motion with {metric_backend} needs {flag}; "
                "use --camera_motion static only for a stationary-camera assumption")
    frames, fps = read_video(args.video, args.max_frames)
    estimator = getattr(args, "camera_estimator", "geocalib")
    c2w, camera_valid, camera_observed = None, None, None
    motion_camera = None
    if estimator == "geocalib":
        camera_repo = getattr(args, "geocalib_repo", None)
        if camera_repo is None and args.dynhamr_repo:
            camera_repo = str(Path(args.dynhamr_repo) / "third-party/vipe")
        raw_camera = estimate_camera(
            frames, camera_repo, getattr(args, "geocalib_weights", "pinhole"),
            getattr(args, "camera_samples", 8), getattr(args, "camera_device", None))
        focal_prior_deviation = getattr(args, "camera_focal_prior_max_deviation", -1.)
        camera = stabilize_camera_estimate(
                raw_camera,
                getattr(args, "camera_focal_prior_ratio", 0.4),
                focal_prior_deviation)
        motion_camera = raw_camera if camera.get("focal_prior_applied") else camera
        camera["motion_intrinsics"] = {
            key: motion_camera[key] for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")
        }
        camera["hand_intrinsics"] = {
            key: camera[key] for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")
        }
        camera["raw_video_camera_estimate"] = raw_camera
    else:
        camera = assumed_camera(frames[0].shape[1], frames[0].shape[0])
        motion_camera = camera
    print(f"Camera fx={camera['fl_x']:.4f}, fy={camera['fl_y']:.4f}, source={camera['source']}")
    camera_json = Path(args.output_dir) / f"{args.episode}_intrinsics.json"
    camera_json.parent.mkdir(parents=True, exist_ok=True)
    camera_json.write_text(json.dumps(camera, indent=2))
    with tempfile.TemporaryDirectory(prefix="ego2robot_path_b_") as temp:
        work = Path(temp)
        if args.predictions:
            predictions = collect_predictions(Path(args.predictions))
        else:
            predictions = run_wilor(frames, args.checkpoint, args.wilor_repo,
                                    args.detector, args.mano_dir, args.wilor_command,
                                    args.batch_size, work)
        save_rich_predictions(
            Path(args.output_dir) / f"{args.episode}_wilor_raw.npz", predictions)
        fit_stats = ({"deferred_until_shared_camera_selection": True} if self_calibrate else
                     calibrate_predictions(predictions, camera, getattr(args, "camera_fit", "pnp")))
        if not any(x["score"] > 0 for x in predictions):
            raise RuntimeError("no valid hand observations remain after camera fitting")
        motion = getattr(args, "camera_motion", "colmap")
        if motion == "colmap":
            if metric_backend == "unidepth":
                depth_repo = args.unidepth_repo
                depth_model = args.unidepth_model
            else:
                depth_repo = args.da3_repo
                depth_model = args.da3_metric_model
            if depth_repo is None and metric_backend == "unidepth" and args.dynhamr_repo:
                depth_repo = str(Path(args.dynhamr_repo) / "third-party/vipe")
            c2w, motion_arrays, motion_stats = camera_motion.estimate_motion(
                frames, motion_camera, predictions,
                Path(args.output_dir) / f"{args.episode}_camera_work",
                args.unidepth_repo, args.unidepth_model, args.camera_device,
                args.pose_stride, args.pose_max_keyframes, args.pose_depth_samples,
                args.pose_threads, args.pose_mask_margin,
                metric_depth_backend=metric_backend,
                metric_depth_repo=depth_repo,
                metric_depth_model=depth_model,
                metric_depth_process_res=args.da3_metric_process_res,
                min_registration_fraction=args.pose_min_registration_fraction,
                hand_depth_alignment=args.hand_depth_alignment,
                hand_camera=camera,
                hand_depth_stride=args.hand_depth_stride,
                hand_depth_patch_size=args.hand_depth_patch_size,
                hand_depth_max_mad=args.hand_depth_max_mad,
                hand_depth_max_delta=args.hand_depth_max_delta,
                intrinsics_policy=args.pose_intrinsics_policy,
                alternative_camera=camera if camera.get("focal_prior_applied") else None,
                self_calibrate=self_calibrate)
            if self_calibrate:
                camera.update(motion_stats["motion_intrinsics"])
                fit_stats = motion_stats["hand_camera_fit"]
            hand_depth_stats = motion_stats.get("hand_depth_alignment", hand_depth_stats)
            camera_output = Path(args.output_dir) / f"{args.episode}_camera.npz"
            camera_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(camera_output, **motion_arrays)
            camera["trajectory_file"] = str(camera_output.resolve())
            camera["motion_estimate"] = motion_stats
            camera["motion_intrinsics"] = {
                key: motion_stats.get("motion_intrinsics", motion_camera)[key]
                for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")
            }
            camera["hand_intrinsics"] = {
                key: camera[key] for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")
            }
            camera_valid = motion_arrays.get("pose_valid", np.ones(len(frames), bool))
            camera_observed = motion_arrays.get("pose_observed", camera_valid)
            for prediction in predictions:
                t = int(prediction["frame"])
                if 0 <= t < len(frames) and not camera_valid[t]:
                    prediction["score"] = 0.
            if not any(prediction["score"] > 0 for prediction in predictions):
                raise RuntimeError("no hand observations overlap a valid camera trajectory")
            print(f"COLMAP camera motion: {motion_stats['registered_keyframes']} keyframes, "
                  f"metric_depth={metric_backend}, "
                  f"scale={motion_stats['translation_scale']['meters_per_native_unit']:.6f}")
        elif args.hand_depth_alignment:
            depth_repo = args.unidepth_repo if metric_backend == "unidepth" else args.da3_repo
            depth_model_path = (args.unidepth_model if metric_backend == "unidepth"
                                else args.da3_metric_model)
            depth_model = camera_motion.load_metric_depth(
                depth_repo, depth_model_path, args.camera_device, backend=metric_backend)
            hand_depth_stats = camera_motion.align_hand_predictions_to_metric_depth(
                frames, predictions, depth_model, camera, metric_backend,
                process_res=args.da3_metric_process_res,
                frame_stride=args.hand_depth_stride,
                patch_size=args.hand_depth_patch_size,
                max_translation_mad=args.hand_depth_max_mad,
                max_translation_delta=args.hand_depth_max_delta)
            del depth_model
        if args.hand_depth_alignment:
            print(f"Hand depth alignment: {hand_depth_stats['hands_aligned']}/"
                  f"{hand_depth_stats['hands_considered']} observations accepted")
        if args.dynhamr_repo:
            mean_params = args.dynhamr_mean_params
            if mean_params is None and args.wilor_repo:
                candidate = Path(args.wilor_repo) / "mano_data" / "mano_mean_params.npz"
                if candidate.is_file():
                    mean_params = str(candidate)
            predictions = apply_official_dynhamr(
                predictions, args.video, args.dynhamr_repo,
                args.dynhamr_mano_dir or args.mano_dir, work,
                args.dynhamr_gpu, args.dynhamr_is_static if c2w is None else False,
                mean_params,
                image_size=(frames[0].shape[1], frames[0].shape[0]),
                n_frames=len(frames),
                output_dir=Path(args.output_dir) / f"{args.episode}_dynhamr", camera=camera,
                camera_trajectory=c2w)
        else:
            predictions = apply_dynhamr(predictions, args.dynhamr_command, work, args.video)
        max_gap = getattr(args, "max_gap", 5)
        if max_gap < 0:
            raise ValueError("max_gap must be nonnegative")
        tracks_2d = detections_to_2d_tracks(predictions, len(frames), frames[0].shape[1], max_gap)
        tracks, scores = detections_to_tracks(predictions, len(frames), frames[0].shape[1], max_gap)
        smoothing_mode = getattr(args, "hand_smoothing_mode", "savgol")
        smoothing_kwargs = {}
        if smoothing_mode == "reprojection_guard":
            smoothing_kwargs = dict(
                scores=scores, tracks_2d=tracks_2d, camera=camera,
                c2w=c2w if c2w is not None else None,
                reprojection_gate_px=args.hand_smoothing_reprojection_gate)
        if c2w is not None:
            # Temporal derivatives must be measured in a shared frame.
            tracks = transform_tracks(smooth_tracks(
                transform_tracks(tracks, c2w), args.smooth_window, args.smooth_polyorder,
                **smoothing_kwargs),
                c2w, inverse=True)
        else:
            tracks = smooth_tracks(
                tracks, args.smooth_window, args.smooth_polyorder,
                **smoothing_kwargs)
        # Keep the temporal solution; per-frame TCP re-anchoring would add
        # motion under a different projection objective.
    output = write_episode(Path(args.output_dir), args.episode, frames, fps, tracks, scores,
                           tracks_2d=tracks_2d, camera=camera, c2w=c2w,
                           camera_valid=camera_valid, camera_observed=camera_observed,
                           metadata={"wilor_checkpoint": str(args.checkpoint),
                            "source_video": str(Path(args.video).resolve()),
                            "reference_data_used_for_reconstruction": False,
                            "camera_motion": motion,
                            "camera_fit": getattr(args, "camera_fit", "pnp"),
                            "camera_fit_stats": fit_stats, "max_interpolation_gap": max_gap,
                            "hand_smoothing_mode": smoothing_mode,
                            "hand_depth_alignment": hand_depth_stats,
                            "temporal_refiner": "DynHaMR" if (args.dynhamr_command or args.dynhamr_repo) else "savgol",
                            "detected_hands": [side for side in ("left", "right")
                                               if float(np.max(scores[side])) > 0.0]})
    camera_json.write_text(json.dumps(camera, indent=2))
    world_tracks = {
        side: np.einsum("ij,nkj->nki", CAMERA_TO_WORLD,
                        np.asarray(tracks[side], dtype=np.float32))
        for side in ("left", "right")
    }
    if c2w is not None:
        world_tracks = transform_tracks(tracks, c2w)
    observed_tracks = {side: world_tracks[side].copy() for side in world_tracks}
    for side in observed_tracks:
        observed_tracks[side][scores[side] <= 0.0] = np.nan
    pred_out = Path(args.output_dir) / f"{args.episode}_predictions.npz"
    exported_c2w = None if c2w is None else c2w.copy()
    if camera_valid is not None:
        exported_c2w[~camera_valid] = np.nan
    np.savez_compressed(pred_out, left_keypoints=observed_tracks["left"], right_keypoints=observed_tracks["right"],
                        left_confidence=scores["left"], right_confidence=scores["right"],
                        fps=np.asarray(fps),
                        coordinate_frame=np.asarray(
                            ("first_camera_world" if camera["motion_estimate"].get(
                                "camera_reference_frame_index", 0) == 0 else "first_valid_camera_world")
                            if c2w is not None else "camera"),
                        **({"c2w": exported_c2w, "camera_pose_valid": camera_valid,
                            "camera_reference_frame_index": np.asarray(
                                camera["motion_estimate"].get("camera_reference_frame_index", 0))}
                           if c2w is not None else {}))
    print(f"Path B complete: {output}")
    print(f"Predictions: {pred_out}")


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Path B: video -> WiLoR/DynHaMR -> EgoVerse-like Zarr")
    ap.add_argument("--video", required=True, help="input RGB video")
    ap.add_argument("--output_dir", required=True, help="directory containing the generated episode")
    ap.add_argument("--episode", default="video_000", help="episode directory name")
    ap.add_argument("--checkpoint", default=os.environ.get("EGO2ROBOT_WILOR_CHECKPOINT"),
                    help="WiLoR checkpoint; pass explicitly or set EGO2ROBOT_WILOR_CHECKPOINT")
    ap.add_argument("--wilor_repo", default=None, help="WiLoR source checkout containing demo.py")
    ap.add_argument("--detector", default=None, help="WiLoR detector checkpoint")
    ap.add_argument("--mano_dir", default=None,
                    help="directory containing MANO_RIGHT.pkl; pass explicitly or set EGO2ROBOT_MANO_DIR")
    ap.add_argument("--wilor_command", default=None,
                    help="custom command template; placeholders: {images} {output} {checkpoint} {detector} {mano_dir}")
    ap.add_argument("--dynhamr_command", default=None,
                    help="optional DynHaMR command template; placeholders: {input} {output} {video}")
    ap.add_argument("--dynhamr_repo", default=None,
                    help="official Dyn-HaMR checkout; runs the bundled bridge")
    ap.add_argument("--dynhamr_mano_dir", default=None,
                    help="MANO model directory for official Dyn-HaMR (defaults to --mano_dir)")
    ap.add_argument("--dynhamr_mean_params", default=None,
                    help="MANO mean params NPZ for official Dyn-HaMR")
    ap.add_argument("--dynhamr_gpu", default="0")
    ap.add_argument("--dynhamr_is_static", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--predictions", default=None,
                    help="existing WiLoR/DynHaMR output directory (offline mode)")
    ap.add_argument("--camera_estimator", choices=("geocalib", "assumed"), default="geocalib",
                    help="estimate intrinsics from RGB; assumed is an explicit uncalibrated fallback")
    ap.add_argument("--geocalib_repo", default=os.environ.get("EGO2ROBOT_GEOCALIB_REPO"),
                    help="ViPE checkout or directory containing the geocalib module")
    ap.add_argument("--geocalib_weights", default=os.environ.get("EGO2ROBOT_GEOCALIB_WEIGHTS", "pinhole"))
    ap.add_argument("--camera_samples", type=int, default=8)
    ap.add_argument("--camera_focal_prior_ratio", type=float, default=0.4,
                    help="RGB-only fallback focal/image-width ratio for gross GeoCalib outliers")
    ap.add_argument("--camera_focal_prior_max_deviation", type=float, default=-1.,
                    help="legacy camera-specific focal prior; disabled by default, "
                         "set 0.35 only to reproduce prior experiments")
    ap.add_argument("--camera_self_calibration",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="experimental COLMAP shared-focal proposal with disjoint temporal "
                         "validation; use the same selected K for hands, motion and depth")
    ap.add_argument("--camera_device", default=None, help="camera estimator device, e.g. cuda or cpu")
    ap.add_argument("--camera_motion", choices=("colmap", "static"), default="colmap",
                    help="fixed-K background SfM + per-frame PnP, or an explicit stationary-camera assumption")
    ap.add_argument("--unidepth_repo", default=os.environ.get("EGO2ROBOT_UNIDEPTH_REPO"),
                    help="ViPE checkout, or its vendored UniDepth source directory")
    ap.add_argument("--unidepth_model", default=os.environ.get("EGO2ROBOT_UNIDEPTH_MODEL"),
                    help="local UniDepthV2 checkpoint for background metric scale")
    ap.add_argument("--metric_depth_backend", choices=("unidepth", "da3_metric"),
                    default=os.environ.get("EGO2ROBOT_METRIC_DEPTH_BACKEND", "unidepth"),
                    help="metric depth for COLMAP scale and optional hand depth alignment")
    ap.add_argument("--da3_repo", default=os.environ.get("EGO2ROBOT_DA3_REPO"),
                    help="official depth-anything-3 checkout containing src/")
    ap.add_argument("--da3_metric_model",
                    default=os.environ.get("EGO2ROBOT_DA3_METRIC_MODEL"),
                    help="local DA3METRIC-LARGE checkpoint directory")
    ap.add_argument("--da3_metric_process_res", type=int,
                    default=int(os.environ.get("EGO2ROBOT_DA3_METRIC_PROCESS_RES", "504")),
                    help="DA3Metric processing resolution")
    ap.add_argument("--pose_stride", type=int, default=3)
    ap.add_argument("--pose_intrinsics_policy", choices=("fixed", "video_consistency"),
                    default="fixed",
                    help="compare original/protected K using common background depth and coverage; "
                         "fixed retains the original GeoCalib K for camera motion")
    ap.add_argument("--pose_max_keyframes", type=int, default=300)
    ap.add_argument("--pose_depth_samples", type=int, default=8)
    ap.add_argument("--pose_threads", type=int, default=4)
    ap.add_argument("--pose_mask_margin", type=int, default=25)
    ap.add_argument("--pose_min_registration_fraction", type=float, default=.7,
                    help="minimum registered COLMAP keyframe fraction; lower only for explicitly weak-background videos")
    ap.add_argument("--camera_fit", choices=("pnp", "pnp_ransac", "translation"), default="pnp",
                    help="fit predicted hand pose to its 2D projection under the video camera estimate")
    ap.add_argument("--hand_depth_alignment",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="use video metric depth to correct hand translation before temporal refinement")
    ap.add_argument("--hand_depth_stride", type=int, default=1,
                    help="align every N observed frames; skipped frames keep their original prediction")
    ap.add_argument("--hand_depth_patch_size", type=int, default=7,
                    help="odd patch size for robust depth sampling at wrist and MCP joints")
    ap.add_argument("--hand_depth_max_mad", type=float, default=.03,
                    help="maximum median 3D disagreement between joint translation estimates, in meters")
    ap.add_argument("--hand_depth_max_delta", type=float, default=.15,
                    help="maximum accepted hand translation correction, in meters")
    ap.add_argument("--max_gap", type=int, default=5,
                    help="maximum internal missing-observation gap to interpolate before smoothing")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--smooth_window", type=int, default=11)
    ap.add_argument("--smooth_polyorder", type=int, default=3)
    ap.add_argument("--hand_smoothing_reprojection_gate", type=float, default=3.0,
                    help="maximum additional RGB reprojection error allowed for "
                         "temporal hand smoothing")
    ap.add_argument("--hand_smoothing_mode",
                    choices=("savgol", "reprojection_guard"), default="savgol",
                    help="temporal hand smoother; reprojection_guard is experimental")
    return ap


def run(args):
    process_video(args)


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
