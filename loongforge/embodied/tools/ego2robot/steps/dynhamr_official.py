#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bridge the official Dyn-HaMR optimizer to Path B's NPZ protocol.

The upstream project is a Hydra video pipeline and does not expose a small
``infer.py`` API. This wrapper runs its official ``run_opt.py`` entry point,
then converts the final MANO parameters into OpenPose-compatible 21-joint
tracks consumed by ``steps.path_b``.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _compat_numpy_chumpy():
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {"bool": bool, "int": int, "float": float,
                        "complex": complex, "object": object,
                        "unicode": str, "str": str}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _find_result(root: Path):
    candidates = sorted(root.rglob("*_smooth_fit_results.npz"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(root.rglob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"Dyn-HaMR produced no NPZ results under {root}")
    return candidates[-1]


def _link_asset(src: Path, dst: Path):
    src = src.expanduser().absolute()
    dst = dst.expanduser().absolute()
    if src == dst:
        try:
            valid = src.is_file()
        except OSError as exc:
            raise FileNotFoundError(f"asset path is a cyclic symlink: {src}") from exc
        if not valid:
            raise FileNotFoundError(src)
        return
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            try:
                if dst.resolve() == src.resolve():
                    return
            except OSError as exc:
                # A self-referential or otherwise cyclic link cannot be
                # resolved. Remove it so the valid source can be installed.
                if exc.errno != 40:
                    raise
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _to_axis_angle(value):
    """Normalize a WiLoR rotation parameter to a flat 3-vector.

    WiLoR normally stores MANO poses as rotation matrices, but singleton
    dimensions can make a root orientation arrive as ``(3, 1)`` or
    ``(1, 3, 3)``. OpenCV 5 is stricter about these shapes than older
    versions, so only an explicit 3x3 matrix is sent to ``Rodrigues``.
    """
    import cv2

    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.shape == (3, 3):
        return cv2.Rodrigues(arr)[0].reshape(3).astype(np.float32)
    flat = arr.reshape(-1)
    if flat.size != 3:
        raise ValueError(f"expected axis-angle or 3x3 rotation, got {np.asarray(value).shape}")
    return flat.astype(np.float32)


def _write_dynhamr_inputs(root: Path, video: Path, seq: str, predictions: Path):
    """Create the official Dyn-HaMR video dataset layout from WiLoR NPZ."""
    import cv2

    with np.load(predictions, allow_pickle=True) as data:
        raw = {k: data[k] for k in data.files}
    n = int(np.max(raw["frame"])) + 1
    images = root / "images" / seq
    tracks_root = root / "dynhamr" / "track_preds" / seq
    shots = root / "dynhamr" / "shot_idcs"
    cameras = root / "dynhamr" / "cameras" / seq / "shot-0"
    for p in (images, tracks_root, shots, cameras):
        p.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 512)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 288)
    idx = 0
    while idx < n:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(images / f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    frame = np.asarray(raw["frame"]).reshape(-1)
    right = np.asarray(raw["right"]).reshape(-1)
    # Select the right-hand track for official single-track optimization. The
    # left hand is handled by a second bridge invocation when needed.
    keep = right > 0.5
    if not keep.any():
        keep = right <= 0.5
    track_id = "001" if bool(np.asarray(right)[keep][0] > 0.5) else "000"
    tracks = tracks_root / track_id
    tracks.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        hits = np.flatnonzero(keep & (frame == i))
        if len(hits):
            j = int(hits[0])
        else:
            valid = np.flatnonzero(keep)
            j = int(valid[np.argmin(np.abs(frame[valid] - i))])
        kp = np.asarray(raw["keypoints_2d"])[j]
        body = np.zeros((21, 3), np.float32)
        body[:, :2] = kp
        body[:, 2] = 1.0
        import json
        (tracks / f"{i:06d}_keypoints.json").write_text(json.dumps(
            {"people": [{"pose_keypoints_2d": body.reshape(-1).tolist()}]}))
        pose = np.asarray(raw["hand_pose"])[j]
        orient = np.asarray(raw["global_orient"])[j]
        pose_aa = np.stack([_to_axis_angle(x) for x in pose[:15]])
        orient_aa = _to_axis_angle(orient)
        (tracks / f"{i:06d}_mano.json").write_text(json.dumps({
            "body_pose": pose_aa.tolist(), "global_orient": orient_aa.tolist(),
            "cam_trans": np.asarray(raw["cam_trans"])[j].tolist(),
            "betas": np.asarray(raw["betas"])[j].tolist(), "is_right": int(right[j] > 0.5)}))
    (shots / f"{seq}.json").write_text(json.dumps({f"{i:06d}.jpg": 0 for i in range(n)}))
    # Static camera file prevents the upstream preprocessor from invoking
    # DROID-SLAM. Dyn-HaMR uses its own fallback intrinsics for static cameras.
    focal = 0.5 * (width + height)
    np.savez(cameras / "cameras.npz", height=height, width=width, focal=focal,
             intrins=np.tile([focal, focal, width / 2.0, height / 2.0], (n, 1)),
             w2c=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)))
    return track_id


def _mano_joints(result, mano_dir: Path):
    """Convert Dyn-HaMR MANO parameters to (N,21,3), preserving track order."""
    _compat_numpy_chumpy()
    import torch
    import smplx
    from smplx.vertex_ids import vertex_ids

    pose = np.asarray(result["pose_body"])
    root = np.asarray(result["root_orient"])
    trans = np.asarray(result.get("trans", np.zeros_like(root)))
    betas = np.asarray(result.get("betas", np.zeros((*root.shape[:-1], 10))))
    # Official output is [tracks, frames, ...]. Collapse a single track; for
    # multiple tracks, concatenate in track order and emit per-track handedness.
    if pose.ndim == 3 and pose.shape[-1] == 3:
        pose = pose[None]
    if root.ndim == 2:
        root = root[None]
    if trans.ndim == 2:
        trans = trans[None]
    if betas.ndim == 2:
        betas = np.repeat(betas[:, None, :], pose.shape[1], axis=1)
    b, t = pose.shape[:2]
    model = smplx.create(str(mano_dir / "MANO_RIGHT.pkl"), model_type="mano", is_rhand=True,
                         use_pca=False, num_pca_comps=45, batch_size=b * t,
                         flat_hand_mean=False)
    def tensor(x):
        return torch.as_tensor(x.reshape(b * t, -1), dtype=torch.float32)
    out = model(global_orient=tensor(root), hand_pose=tensor(pose),
                betas=tensor(betas), transl=tensor(trans))
    base = out.joints
    extras = out.vertices[:, [vertex_ids["mano"][k] for k in vertex_ids["mano"]]]
    # smplx's MANO order is rearranged to OpenPose order by this fixed map.
    joints = torch.cat([base, extras], dim=1)
    order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12,
             19, 7, 8, 9, 20]
    joints = joints[:, order].reshape(b, t, 21, 3).detach().cpu().numpy()
    return joints


def run(args):
    repo = Path(args.repo).resolve()
    run_opt = repo / "dyn-hamr" / "run_opt.py"
    if not run_opt.is_file():
        raise FileNotFoundError(f"official Dyn-HaMR entry point not found: {run_opt}")
    video = Path(args.video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = repo / "_DATA" / "data"
    _link_asset(Path(args.mano_dir).expanduser().resolve() / "MANO_RIGHT.pkl",
                data_root / "mano" / "MANO_RIGHT.pkl")
    if args.mano_mean_params:
        # Do not call resolve() here: a stale destination may be a cyclic
        # symlink, and _link_asset handles replacing it safely.
        _link_asset(Path(args.mano_mean_params).expanduser(),
                    data_root / "mano_mean_params.npz")
    if not (repo / "_DATA" / "BMC").is_dir():
        raise FileNotFoundError(
            f"Dyn-HaMR BMC assets missing: {repo / '_DATA' / 'BMC'}. "
            "Run the official scripts/prepare.sh and BMC preparation first."
        )
    seq = args.seq
    with tempfile.TemporaryDirectory(prefix="dynhamr_bridge_") as td:
        root = Path(td)
        (root / "videos").mkdir()
        target = root / "videos" / f"{seq}.mp4"
        try:
            target.symlink_to(video)
        except OSError:
            shutil.copy2(video, target)
        track_id = _write_dynhamr_inputs(root, video, seq, Path(args.predictions))
        cmd = [sys.executable, str(run_opt), "data=video_driod", "run_opt=True",
               f"data.root={root}", "data.video_dir=videos", f"data.seq={seq}",
               "data.ext=mp4", f"is_static={'True' if args.is_static else 'False'}",
               f"data.track_ids={track_id}", "data.shot_idx=0",
               "run_vis=False", "run_prior=False", f"gpu={args.gpu}",
               f"log_root={output / 'logs'}"]
        # Load the local compatibility shim before Dyn-HaMR's legacy imports.
        shim_dir = Path(__file__).resolve().parent
        pythonpath = os.pathsep.join((str(shim_dir), str(repo / "dyn-hamr")))
        env = {**os.environ, "PYTHONPATH": pythonpath}
        proc = subprocess.run(cmd, cwd=str(repo / "dyn-hamr"), env=env,
                              text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError("Dyn-HaMR failed (last output):\n" +
                               (proc.stdout + "\n" + proc.stderr)[-10000:])
        result_path = _find_result(output / "logs")
        with np.load(result_path, allow_pickle=True) as data:
            result = {k: data[k] for k in data.files}
        joints = _mano_joints(result, Path(args.mano_dir).resolve())
        right = np.asarray(result.get("is_right", np.ones(joints.shape[:2])), dtype=np.float32)
        if right.ndim == 1:
            right = right[:, None]
        frame = np.arange(joints.shape[1], dtype=np.int64)[None].repeat(joints.shape[0], 0)
        np.savez_compressed(output / "dynhamr_predictions.npz",
                            pred_keypoints_3d=joints.reshape(-1, 21, 3),
                            right=right.reshape(-1), frame=frame.reshape(-1),
                            score=np.ones(joints.shape[:2], np.float32).reshape(-1))
    print(f"Dyn-HaMR predictions: {output / 'dynhamr_predictions.npz'}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Run official Dyn-HaMR and export Path B NPZ")
    p.add_argument("--repo", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--mano_dir", required=True)
    p.add_argument("--mano_mean_params", default="/tmp/WiLoR-source/mano_data/mano_mean_params.npz")
    p.add_argument("--seq", default="path_b")
    p.add_argument("--gpu", default="0")
    p.add_argument("--is_static", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
