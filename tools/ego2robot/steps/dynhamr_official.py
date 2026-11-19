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
import json
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# smplx/MANO's native joint order differs from the OpenPose hand order used by
# Dyn-HaMR tracks and Path B. Values are native MANO indices at OpenPose slots.
MANO_TO_OPENPOSE = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
OPENPOSE_TO_MANO = np.argsort(MANO_TO_OPENPOSE)


def _compat_numpy_chumpy():
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {"bool": bool, "int": int, "float": float,
                        "complex": complex, "object": object,
                        "unicode": str, "str": str}.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _find_result(root: Path):
    # Official releases save iteration-numbered files inside smooth_fit/.
    smooth = [p for p in root.rglob("*_world_results.npz")
              if p.parent.name == "smooth_fit"]
    if smooth:
        return max(smooth, key=lambda p: (int(p.name.split("_")[-3]),
                                          p.stat().st_mtime))
    candidates = sorted(root.rglob("*_smooth_fit_results.npz"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"Dyn-HaMR produced no completed smooth results under {root}")
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


def _write_dynhamr_inputs(root: Path, video: Path, seq: str, predictions: Path,
                         max_frames: int = 0, camera=None, camera_trajectory=None):
    """Create the official Dyn-HaMR video dataset layout from WiLoR NPZ."""
    import cv2

    with np.load(predictions, allow_pickle=True) as data:
        raw = {k: data[k] for k in data.files}
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raise ValueError(f"cannot determine video length: {video}")
    if max_frames > 0:
        n = min(n, max_frames)
    images = root / "images" / seq
    tracks_root = root / "dynhamr" / "track_preds" / seq
    shots = root / "dynhamr" / "shot_idcs"
    cameras = root / "dynhamr" / "cameras" / seq / "shot-0"
    for p in (images, tracks_root, shots, cameras):
        p.mkdir(parents=True, exist_ok=True)
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
    if idx != n:
        raise ValueError(f"decoded {idx}/{n} frames from {video}")
    frame = np.asarray(raw["frame"]).reshape(-1)
    right = np.asarray(raw["right"]).reshape(-1)
    track_ids = []
    for is_right in (False, True):
        keep = (right > 0.5) == is_right
        keep &= (frame >= 0) & (frame < n)
        keep &= np.isfinite(raw["score"]) & (np.asarray(raw["score"]) > 0)
        valid = np.flatnonzero(keep)
        if not len(valid):
            continue
        track_id = f"{int(is_right):03d}"
        track_ids.append(track_id)
        tracks = tracks_root / track_id
        tracks.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            hits = np.flatnonzero(keep & (frame == i))
            if len(hits):
                j = int(hits[np.argmax(raw["score"][hits])])
                confidence = float(raw["score"][j])
            else:
                j = int(valid[np.argmin(np.abs(frame[valid] - i))])
                confidence = 0.0
            kp = np.asarray(raw["keypoints_2d"])[j]
            body = np.zeros((21, 3), np.float32)
            body[:, :2] = kp
            body[:, 2] = confidence
            (tracks / f"{i:06d}_keypoints.json").write_text(json.dumps(
                {"people": [{"pose_keypoints_2d": body.reshape(-1).tolist()}]}))
            pose = np.asarray(raw["hand_pose"])[j]
            orient = np.asarray(raw["global_orient"])[j]
            pose_aa = np.stack([_to_axis_angle(x) for x in pose[:15]])
            orient_aa = _to_axis_angle(orient)
            # The upstream MANO wrapper mirrors x AFTER adding translation.
            # WiLoR's full-image left translation is already unmirrored.
            trans = np.asarray(raw["cam_trans"])[j].copy()
            if not is_right:
                trans[0] *= -1
            (tracks / f"{i:06d}_mano.json").write_text(json.dumps({
                "body_pose": pose_aa.tolist(), "global_orient": orient_aa.tolist(),
                "cam_trans": trans.tolist(),
                "betas": np.asarray(raw["betas"])[j].tolist(),
                "is_right": int(is_right)}))
    if not track_ids:
        raise ValueError("no hand observations inside requested video interval")
    (shots / f"{seq}.json").write_text(json.dumps({f"{i:06d}.jpg": 0 for i in range(n)}))
    # Use the same RGB-estimated intrinsics as initialization and export.
    if camera is None:
        values = [float(max(width, height)), float(max(width, height)), width / 2, height / 2]
    else:
        values = [float(camera[k]) for k in ("fl_x", "fl_y", "cx", "cy")]
        if camera["w"] != width or camera["h"] != height:
            raise ValueError("video camera estimate resolution does not match input")
    focal = float(np.mean(values[:2]))
    w2c = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    if camera_trajectory is not None:
        try:
            from .camera_geometry import validate_trajectory
        except ImportError:
            from camera_geometry import validate_trajectory
        w2c = np.linalg.inv(validate_trajectory(camera_trajectory, n)).astype(np.float32)
    np.savez(cameras / "cameras.npz", height=height, width=width, focal=focal,
             intrins=np.tile(values, (n, 1)),
             w2c=w2c)
    return "-".join(track_ids)


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
                         flat_hand_mean=True)
    def tensor(x):
        return torch.as_tensor(x.reshape(b * t, -1), dtype=torch.float32)
    out = model(global_orient=tensor(root), hand_pose=tensor(pose),
                betas=tensor(betas), transl=tensor(trans))
    base = out.joints
    extras = out.vertices[:, [vertex_ids["mano"][k] for k in vertex_ids["mano"]]]
    # smplx's MANO order is rearranged to OpenPose order by this fixed map.
    joints = torch.cat([base, extras], dim=1)
    joints = joints[:, MANO_TO_OPENPOSE].reshape(b, t, 21, 3)
    # The bridge protocol is OpenPose order, matching WiLoR and Path B.
    return joints.detach().cpu().numpy()


def _restore_camera_handedness(joints, is_right):
    """Undo Dyn-HaMR's right-hand MANO canonicalization for left tracks."""
    joints = np.asarray(joints, dtype=np.float32).copy()
    right = np.asarray(is_right, dtype=np.float32)
    while right.ndim < joints.ndim - 1:
        right = right[..., None]
    if right.shape[-1:] == (1,):
        right = right[..., 0]
    sign = np.where(right > 0.5, 1.0, -1.0).astype(np.float32)
    joints[..., 0] *= sign[..., None]
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
    # BMC statistics are only required when the optional bio loss is enabled.
    # The official default weights are zero; the optimizer checks when used.
    seq = args.seq
    from contextlib import nullcontext
    work_root = Path(args.work_dir).resolve() if args.work_dir else output / "inputs"
    work_root.mkdir(parents=True, exist_ok=True)
    with nullcontext(work_root) as root:
        (root / "videos").mkdir(exist_ok=True)
        target = root / "videos" / f"{seq}.mp4"
        if not target.exists():
            try:
                target.symlink_to(video)
            except OSError:
                shutil.copy2(video, target)
        camera = json.loads(Path(args.video_camera).read_text()) if args.video_camera else None
        if camera is not None and not (
                str(camera.get("source", "")).startswith("video:GeoCalib")
                or camera.get("source") == "assumed:max(width,height)"):
            raise ValueError("bridge camera must be generated by the video-only camera estimator")
        trajectory = None
        if args.video_camera_trajectory:
            if camera is None:
                raise ValueError("camera trajectory requires video-estimated intrinsics")
            with np.load(args.video_camera_trajectory, allow_pickle=False) as data:
                if str(data["source"]) != "video:COLMAP" or str(data["units"]) != "meters":
                    raise ValueError("unsupported camera trajectory provenance")
                trajectory = data["c2w"]
        track_id = _write_dynhamr_inputs(root, video, seq, Path(args.predictions),
                                        args.max_frames, camera, trajectory)
        n_frames = len(list((root / "images" / seq).glob("*.jpg")))
        cmd = [sys.executable, str(run_opt), "data=video_driod", "run_opt=True",
               f"data.root={root}", "data.video_dir=videos", f"data.seq={seq}",
               "data.ext=mp4", f"is_static={'True' if args.is_static else 'False'}",
               f"data.track_ids={track_id}", "data.shot_idx=0",
               f"data.end_idx={n_frames}",
               "run_vis=False", "run_prior=False", f"gpu={args.gpu}",
               f"optim.root.num_iters={args.root_iters}",
               f"optim.smooth.num_iters={args.smooth_iters}",
               f"log_root={output / 'logs'}"]
        if trajectory is not None:
            # Scale has already been inferred from RGB/MANO. Keep the same
            # camera trajectory for world export and temporal optimization.
            cmd.append("model.opt_scale=False")
        # Load the local compatibility shim before Dyn-HaMR's legacy imports.
        shim_dir = Path(__file__).resolve().parent
        pythonpath = os.pathsep.join((str(shim_dir), str(repo / "dyn-hamr")))
        env = {**os.environ, "PYTHONPATH": pythonpath}
        if str(args.gpu) == "-1":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        log_path = output / "optimizer.log"
        with log_path.open("w") as log:
            log.write(" ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(repo / "dyn-hamr"), env=env,
                                  text=True, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError("Dyn-HaMR failed (last output):\n" +
                               log_path.read_text()[-10000:])
        result_path = _find_result(output / "logs")
        with np.load(result_path, allow_pickle=True) as data:
            result = {k: data[k] for k in data.files}
        joints = _mano_joints(result, Path(args.mano_dir).resolve())
        right = np.asarray(result.get("is_right", np.ones(joints.shape[:2])), dtype=np.float32)
        if right.ndim == 1:
            right = right[:, None]
        joints = _restore_camera_handedness(joints, right)
        if joints.shape[1] != n_frames:
            raise ValueError(f"optimizer returned {joints.shape[1]}/{n_frames} frames")
        if "cam_R" in result and "cam_t" in result:
            joints = (np.einsum("btij,btkj->btki", result["cam_R"], joints)
                      + np.asarray(result["cam_t"])[:, :, None, :])
        frame = np.arange(joints.shape[1], dtype=np.int64)[None].repeat(joints.shape[0], 0)
        with np.load(args.predictions, allow_pickle=True) as source:
            scores = np.zeros(joints.shape[:2], np.float32)
            for b in range(joints.shape[0]):
                side = bool(right[b, 0] > 0.5)
                for j in np.flatnonzero((source["right"] > 0.5) == side):
                    t = int(source["frame"][j])
                    if 0 <= t < n_frames:
                        scores[b, t] = max(scores[b, t], float(source["score"][j]))
        np.savez_compressed(output / "dynhamr_predictions.npz",
                            pred_keypoints_3d=joints.reshape(-1, 21, 3),
                            right=right.reshape(-1), frame=frame.reshape(-1),
                            score=scores.reshape(-1))
        iterations = {"root_fit": 0, "smooth_fit": 0}
        stage = None
        for line in log_path.read_text().splitlines():
            for name in iterations:
                if line.startswith(f"OPTIMIZING {name} "):
                    stage = name
            if stage is not None and re.match(r"^ITER:\s+\d+", line):
                iterations[stage] += 1
        (output / "run_metadata.json").write_text(json.dumps({
            "result_file": str(result_path), "frames": n_frames,
            "track_ids": track_id, "root_iters": args.root_iters,
            "smooth_iters": args.smooth_iters, "run_prior": False,
            "executed_iterations": iterations,
            "optimizer_device": "cpu" if str(args.gpu) == "-1" else f"cuda:{args.gpu}",
            "camera": camera, "coordinate_frame": "camera",
            "optimizer_log": str(log_path)}, indent=2))
    print(f"Dyn-HaMR predictions: {output / 'dynhamr_predictions.npz'}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Run official Dyn-HaMR and export Path B NPZ")
    p.add_argument("--repo", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--mano_dir", required=True)
    p.add_argument("--mano_mean_params", default=None)
    p.add_argument("--seq", default="path_b")
    p.add_argument("--gpu", default="0")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--root_iters", type=int, default=50)
    p.add_argument("--smooth_iters", type=int, default=300)
    p.add_argument("--work_dir", default=None)
    p.add_argument("--video_camera", default=None,
                   help="internal camera estimate generated from RGB by Path B")
    p.add_argument("--video_camera_trajectory", default=None,
                   help="internal RGB-only COLMAP c2w trajectory with estimated metric scale")
    p.add_argument("--is_static", action=argparse.BooleanOptionalAction, default=True)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
