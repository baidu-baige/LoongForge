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
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import zarr

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

# WiLoR/Dyn-HaMR use camera coordinates (x right, y up, z forward).  The
# synthetic head is identity and ``robot_retarget.set_ego_camera`` applies
# RX180, so the two camera conventions cancel and these coordinates can be
# used directly as world coordinates (in particular, z stays in front of the
# MuJoCo camera).
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
                             "pred_keypoints", "mano_joints"))
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
    return [{"keypoints": kp[i], "right": right[i], "frame": int(frame[i]),
             "score": float(score[i])} for i in range(n)]


def collect_predictions(root: Path):
    """Collect prediction files emitted by WiLoR/DynHaMR."""
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".npz", ".json"})
    detections = []
    for path in files:
        detections.extend(_load_prediction_file(path))
    if not detections:
        raise RuntimeError(
            f"no 3D hand predictions found under {root}; expected NPZ/JSON with "
            "pred_keypoints_3d (or keypoints_3d/joints_3d)"
        )
    return detections


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


def _align_3d_tcp_to_2d(kp, keypoints_2d, width, height):
    """Translate refined 3D points so their pinch center matches the 2D track."""
    xyz_raw = np.asarray(kp, dtype=np.float32)
    obs_raw = np.asarray(keypoints_2d, dtype=np.float32)
    if xyz_raw.size != MANO_JOINTS * 3 or obs_raw.size != MANO_JOINTS * 2:
        return xyz_raw.reshape(-1, 3)
    xyz = xyz_raw.reshape(MANO_JOINTS, 3).copy()
    obs = obs_raw.reshape(MANO_JOINTS, 2)
    if not np.isfinite(xyz).all() or not np.isfinite(obs).all():
        return xyz
    focal = float(max(width, height))
    center = np.asarray([width * 0.5, height * 0.5], dtype=np.float32)
    observed_vf = 0.7 * obs[INDEX_TIP] + 0.3 * obs[MIDDLE_TIP]
    observed_tcp = 0.5 * (obs[THUMB_TIP] + observed_vf)
    virtual_finger = 0.7 * xyz[INDEX_TIP] + 0.3 * xyz[MIDDLE_TIP]
    tcp = 0.5 * (xyz[THUMB_TIP] + virtual_finger)
    if tcp[2] <= 1e-4:
        return xyz
    shift = np.zeros(3, dtype=np.float32)
    # Project the actual Eq.1 TCP, not the mean of projected fingertips:
    # perspective projection and averaging do not commute at unequal depths.
    shift[:2] = (observed_tcp - center) * (float(tcp[2]) / focal) - tcp[:2]
    return xyz + shift


def apply_official_dynhamr(predictions, video: str, repo: str, mano_dir: str,
                           work_dir: Path, gpu: str, is_static: bool,
                           mean_params: str | None = None, image_size=None):
    """Run the official Dyn-HaMR bridge and keep the unrefined other hand."""
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
    cmd = [sys.executable, str(Path(__file__).with_name("dynhamr_official.py")),
           "--repo", repo, "--video", video, "--output", str(work_dir / "dynhamr_output"),
           "--predictions", str(source), "--mano_dir", mano_dir, "--gpu", str(gpu)]
    if mean_params:
        cmd.extend(["--mano_mean_params", mean_params])
    if not is_static:
        cmd.append("--no-is_static")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError("official Dyn-HaMR failed (last output):\n" +
                           (proc.stdout + "\n" + proc.stderr)[-10000:])
    refined = collect_predictions(work_dir / "dynhamr_output")
    if image_size is None:
        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    else:
        width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        width, height = 640, 368
    # Dyn-HaMR optimizes in a world frame whose origin/translation can differ
    # from WiLoR's camera-relative MANO coordinates (especially with a static
    # synthetic camera). Align each refined hand by its wrist, then anchor its
    # pinch center to WiLoR's 2D observation so the IK target and demo agree.
    if refined:
        refined_side = bool(float(refined[0]["right"]) > 0.5)
        source_track = [x for x in predictions
                        if bool(float(x["right"]) > 0.5) == refined_side]
        by_frame = {int(x["frame"]): x for x in source_track}
        source_frames = np.asarray(sorted(by_frame), dtype=np.int64)
        for item in refined:
            frame_id = int(item["frame"])
            if frame_id in by_frame:
                source = by_frame[frame_id]
            elif len(source_frames):
                source = by_frame[int(source_frames[np.argmin(np.abs(source_frames - frame_id))])]
            else:
                continue
            source_kp = np.asarray(source["keypoints"], dtype=np.float32).reshape(21, 3)
            refined_kp = np.asarray(item["keypoints"], dtype=np.float32).reshape(21, 3)
            refined_kp = refined_kp + (source_kp[WRIST] - refined_kp[WRIST])
            item["keypoints"] = refined_kp
            # Keep WiLoR's original full-image 2D observations for the demo.
            # Dyn-HaMR refines 3D MANO, but its exported 3D->2D projection can
            # drift because the synthetic camera has different intrinsics.
            source_2d = source.get("_wilor_extra", {}).get("keypoints_2d")
            if source_2d is not None:
                source_2d = np.asarray(source_2d, dtype=np.float32).reshape(21, 2)
                refined_kp = _align_3d_tcp_to_2d(
                    refined_kp, source_2d, width, height)
                item["keypoints_2d"] = source_2d
            item["keypoints"] = refined_kp
    refined_side = bool(refined[0]["right"]) if refined else True
    return refined + [x for x in predictions if bool(float(x["right"]) > 0.5) != refined_side]


def detections_to_2d_tracks(detections, n_frames, width):
    """Collect optional full-image 2D observations in left/right tracks."""
    tracks = {side: np.full((n_frames, MANO_JOINTS, 2), np.nan, np.float32)
              for side in ("left", "right")}
    for det in sorted(detections, key=lambda d: (d["frame"], -d["score"])):
        t = int(det["frame"])
        value = det.get("keypoints_2d")
        if value is None:
            value = det.get("_wilor_extra", {}).get("keypoints_2d")
        if t < 0 or t >= n_frames or value is None:
            continue
        kp = np.asarray(value, dtype=np.float32).reshape(-1, 2)
        if kp.shape != (MANO_JOINTS, 2) or not np.isfinite(kp).all():
            continue
        side = "right" if _side(det["right"], np.pad(kp, ((0, 0), (0, 1))), width) else "left"
        if not np.isfinite(tracks[side][t]).all():
            tracks[side][t] = kp
    for side, arr in tracks.items():
        valid = np.isfinite(arr).all(axis=(1, 2))
        if not valid.any():
            continue
        idx, good = np.arange(n_frames), np.flatnonzero(valid)
        for j in range(MANO_JOINTS):
            for d in range(2):
                arr[:, j, d] = np.interp(idx, good, arr[good, j, d])
    return tracks


def _side(value, kp, width):
    if isinstance(value, str):
        return value.lower().startswith(("r", "1"))
    if np.isfinite(value) and float(value) >= 0:
        return bool(float(value) > 0.5)
    return bool(float(np.mean(kp[:, 0])) >= width / 2.0)


def detections_to_tracks(detections, n_frames, width):
    """Assign detections to stable tracks, accepting single-hand videos.

    Missing sides remain invalid. Downstream retargeting uses the confidence
    arrays to skip IK for a hand that was not observed.
    """
    tracks = {"left": np.full((n_frames, MANO_JOINTS, 3), np.nan, np.float32),
              "right": np.full((n_frames, MANO_JOINTS, 3), np.nan, np.float32)}
    scores = {"left": np.zeros(n_frames, np.float32), "right": np.zeros(n_frames, np.float32)}
    for det in sorted(detections, key=lambda d: (d["frame"], -d["score"])):
        t = det["frame"]
        if t < 0 or t >= n_frames:
            continue
        kp = _as_keypoints(det["keypoints"])
        if kp is None:
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
            # Interpolate short gaps and hold the edge. Long gaps still become a
            # held pose so the downstream dual-arm solver receives a continuous
            # trajectory; confidence records which frames were inferred.
            idx = np.arange(n_frames)
            good = np.flatnonzero(valid)
            for j in range(MANO_JOINTS):
                for d in range(3):
                    arr[:, j, d] = np.interp(idx, good, arr[good, j, d])
            tracks[side] = arr

    if not valid_sides:
        raise RuntimeError("WiLoR produced no hand detections")
    return tracks, scores


def smooth_tracks(tracks, window=11, polyorder=3):
    """Smooth finite tracks, preserving invalid/missing tracks unchanged."""
    if window <= polyorder + 1 or window % 2 == 0:
        raise ValueError("smooth window must be odd and greater than polyorder")
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return tracks
    out = {}
    for side, arr in tracks.items():
        out[side] = arr.copy()
        # Missing hands deliberately contain NaNs. Savitzky-Golay's edge
        # fitting requires finite input; never invent coordinates to fill them.
        if len(arr) >= window and np.isfinite(arr).all():
            out[side] = savgol_filter(arr, window, polyorder, axis=0).astype(np.float32)
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
    # Matrix columns are the Eq. (3) axes. Convert to wxyz without scipy.
    quat = np.zeros((len(kp), 4), np.float32)
    for i, R in enumerate(np.stack([x, y, z], axis=2)):
        tr = np.trace(R)
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            quat[i] = [(0.25 * s), (R[2, 1] - R[1, 2]) / s,
                       (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        else:
            quat[i, 0] = 1.0
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
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


def write_episode(output_dir: Path, episode: str, frames, fps, tracks, scores,
                  tracks_2d=None, metadata=None):
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
    observed_tracks = {side: world_tracks[side].copy() for side in world_tracks}
    for side in observed_tracks:
        observed_tracks[side][scores[side] <= 0.0] = np.nan
    left_kp = observed_tracks["left"].reshape(n, -1).astype(np.float32)
    right_kp = observed_tracks["right"].reshape(n, -1).astype(np.float32)
    left_ee, _ = _tcp_pose(world_tracks["left"], -1.0)
    right_ee, _ = _tcp_pose(world_tracks["right"], +1.0)
    head = np.zeros((n, 7), np.float32)
    head[:, 3:7] = CAMERA_HEAD_QUAT
    attrs = {"total_frames": n, "fps": float(fps), "source": "Ego2Robot Path B",
             "path": "B", "coordinate_frame": "mujoco_world",
             "intrinsics": {"fl_x": float(max(w, h)),
             "fl_y": float(max(w, h)), "cx": w / 2.0, "cy": h / 2.0,
             "w": w, "h": h}}
    if metadata:
        attrs.update(metadata)
    group = zarr.open_group(str(ep_dir), mode="w", zarr_format=3)
    group.attrs.update(attrs)
    arrays = {
        "left.obs_keypoints": left_kp, "right.obs_keypoints": right_kp,
        "left.obs_wrist_pose": np.concatenate([observed_tracks["left"][:, WRIST], _quat_identity(n)], 1),
        "right.obs_wrist_pose": np.concatenate([observed_tracks["right"][:, WRIST], _quat_identity(n)], 1),
        "left.obs_ee_pose": left_ee, "right.obs_ee_pose": right_ee,
        "obs_head_pose": head,
        "obs_rgb_timestamps_ns": (np.arange(n) * (1e9 / fps)).astype(np.int64),
        "path_b_left_confidence": scores["left"], "path_b_right_confidence": scores["right"],
    }
    if tracks_2d is not None:
        arrays["left.obs_keypoints_2d"] = np.asarray(tracks_2d["left"], dtype=np.float32).reshape(n, -1)
        arrays["right.obs_keypoints_2d"] = np.asarray(tracks_2d["right"], dtype=np.float32).reshape(n, -1)
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
    frames, fps = read_video(args.video, args.max_frames)
    with tempfile.TemporaryDirectory(prefix="ego2robot_path_b_") as temp:
        work = Path(temp)
        if args.predictions:
            predictions = collect_predictions(Path(args.predictions))
        else:
            predictions = run_wilor(frames, args.checkpoint, args.wilor_repo,
                                    args.detector, args.mano_dir, args.wilor_command,
                                    args.batch_size, work)
        if args.dynhamr_repo:
            predictions = apply_official_dynhamr(
                predictions, args.video, args.dynhamr_repo,
                args.dynhamr_mano_dir or args.mano_dir, work,
                args.dynhamr_gpu, args.dynhamr_is_static,
                args.dynhamr_mean_params,
                image_size=(frames[0].shape[1], frames[0].shape[0]))
        else:
            predictions = apply_dynhamr(predictions, args.dynhamr_command, work, args.video)
        tracks_2d = detections_to_2d_tracks(predictions, len(frames), frames[0].shape[1])
        tracks, scores = detections_to_tracks(predictions, len(frames), frames[0].shape[1])
        tracks = smooth_tracks(tracks, args.smooth_window, args.smooth_polyorder)
        # Smoothing can move the pinch center by a few pixels. Re-anchor the
        # final 3D tracks after smoothing so IK and the 2D diagnostic agree.
        for side in ("left", "right"):
            for t in range(len(frames)):
                if (scores[side][t] > 0.0 and
                        np.isfinite(tracks[side][t]).all() and
                        np.isfinite(tracks_2d[side][t]).all()):
                    tracks[side][t] = _align_3d_tcp_to_2d(
                        tracks[side][t], tracks_2d[side][t],
                        frames[0].shape[1], frames[0].shape[0])
    output = write_episode(Path(args.output_dir), args.episode, frames, fps, tracks, scores,
                           tracks_2d=tracks_2d,
                           metadata={"wilor_checkpoint": str(args.checkpoint),
                            "temporal_refiner": "DynHaMR" if args.dynhamr_command else "savgol",
                            "detected_hands": [side for side in ("left", "right")
                                               if float(np.max(scores[side])) > 0.0]})
    world_tracks = {
        side: np.einsum("ij,nkj->nki", CAMERA_TO_WORLD,
                        np.asarray(tracks[side], dtype=np.float32))
        for side in ("left", "right")
    }
    observed_tracks = {side: world_tracks[side].copy() for side in world_tracks}
    for side in observed_tracks:
        observed_tracks[side][scores[side] <= 0.0] = np.nan
    pred_out = Path(args.output_dir) / f"{args.episode}_predictions.npz"
    np.savez_compressed(pred_out, left_keypoints=observed_tracks["left"], right_keypoints=observed_tracks["right"],
                        left_confidence=scores["left"], right_confidence=scores["right"],
                        fps=np.asarray(fps))
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
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--smooth_window", type=int, default=11)
    ap.add_argument("--smooth_polyorder", type=int, default=3)
    return ap


def run(args):
    process_video(args)


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
