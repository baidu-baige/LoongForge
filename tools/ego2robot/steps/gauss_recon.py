#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Step 5g - Scene reconstruction via DVLT (Déjà View) for depth compositing.

Reconstruct the EgoVerse scene from an ego-centric video with NVIDIA's DVLT
(feed-forward multi-view reconstruction: camera poses, per-pixel depth, world
points, confidence), estimate scale from hand keypoints (trajectory-ratio
fallback), and unproject depth through SLAM poses into the metric world frame.
Pose similarity registration is diagnostic only. Project the metric points
for every aligned frame with the retarget-compatible pinhole model. The
result is a ``scene_depth.npz`` interface-compatible with the DA3 output of
:mod:`steps.depth_estimate` (``backend="gauss"`` marks it as already metric, so
the retarget step skips its DA3 scale/shift alignment).

Input:
  --zarr_dir  <EgoVerse zarr root>      (raw frames + obs_head_pose + attrs)
  --state_dir <02_align>                 (frame count alignment; {ep}.npz)
  --bg_dir    <04_inpaint>               (only to detect the bg video size)
Output:
  {output_dir}/{ep}/scene_depth.npz      (float16 (N,H,W) meters, DA3 interface)
  {output_dir}/{ep}/{ep}_dvlt.pt         (raw DVLT + alignment for offline analysis)
  {output_dir}/{ep}/recon_vis.mp4        (original | depth, sampled)
  {output_dir}/{ep}/depth_vis.mp4        (turbo colormap)
  {output_dir}/{ep}/meta.json            (config + alignment stats)

The DVLT checkpoint is a local directory (offline) or an HF hub id, e.g.:

    python cli.py gauss-recon --zarr_dir data/egoverse --state_dir data/02_align \
        --bg_dir data/04_inpaint --output_dir data/05_gauss \
        --checkpoint /path/to/nvidia/dvlt
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from steps import config as config_mod  # noqa: F401  (env resolution side effects)
from steps.gauss import align, export, scene_io
from steps.gauss.dvlt_runner import (DEFAULT_CHECKPOINT, DEFAULT_IMG_SIZE,
                                DEFAULT_PATCH_SIZE, load_model, run_dvlt)

log = logging.getLogger("gauss_recon")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(3, 3) rotation -> wxyz quaternion."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / (np.linalg.norm(q) + 1e-12)


def _c2w_to_pose7(c2w: np.ndarray) -> np.ndarray:
    """(S, 3, 4) camera-to-world -> (S, 7) [t xyz, q wxyz]."""
    S = c2w.shape[0]
    out = np.zeros((S, 7), dtype=np.float64)
    out[:, :3] = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    for i in range(S):
        out[i, 3:7] = _mat_to_quat(R[i])
    return out


def _sample_points(wp: np.ndarray, conf: np.ndarray, conf_threshold: float,
                   points_per_frame: int, seed: int = 0) -> np.ndarray:
    """Flatten, threshold on confidence, and downsample per frame."""
    S, h, w, _ = wp.shape
    flat = wp.reshape(S, -1, 3)
    cflat = conf.reshape(S, -1)
    rng = np.random.default_rng(seed)
    out = []
    for s in range(S):
        m = cflat[s] >= conf_threshold
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            idx = np.flatnonzero(np.isfinite(cflat[s]))  # keep geometry anyway
        if len(idx) > points_per_frame:
            idx = idx[rng.choice(len(idx), size=points_per_frame, replace=False)]
        if len(idx):
            out.append(flat[s][idx])
    if not out:
        raise RuntimeError("no points survived confidence filtering")
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Step 5g: DVLT scene reconstruction -> metric scene depth")
    p.add_argument("--zarr_dir", default=None,
                   help="Raw EgoVerse zarr root (subdirs are episodes)")
    p.add_argument("--state_dir", default=None,
                   help="Required except --check: Step 2 output ({ep}.npz); missing files fall back to Zarr frame count")
    p.add_argument("--bg_dir", default=None,
                   help="Required except --check: Step 4 output; missing bg video falls back to source image size")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Local directory with nvidia/dvlt weights "
                        "(config.json + model.safetensors) or HF hub id")
    p.add_argument("--dinov2_weights", default=None,
                   help="Optional local DINOv2 ViT-B/14 reg4 state dict, used "
                        "only if the checkpoint lacks patch_embed weights")
    p.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--sample_fps", type=float, default=2.0,
                   help="Frames per second fed to DVLT (None = every frame)")
    p.add_argument("--fps", type=float, default=None,
                   help="Source video fps (default: zarr attrs or 30)")
    p.add_argument("--fy", type=float, default=None,
                   help="Vertical focal length in pixels (default: DVLT prediction "
                        "scaled to output height; keep retarget consistent)")
    p.add_argument("--conf_threshold", type=float, default=0.2,
                   help="DVLT world-point confidence cutoff")
    p.add_argument("--points_per_frame", type=int, default=8192,
                   help="Max world points kept per DVLT view (downsampling)")
    p.add_argument("--hole_fill_iters", type=int, default=6,
                   help="Nearest-neighbor hole fill iterations")
    p.add_argument("--inference_steps", type=int, default=None,
                   help="DVLT loop steps (default: model choice)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episodes", nargs="*", default=None,
                   help="Specific episode IDs; default = all in zarr_dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true",
                   help="Environment self-check only (no reconstruction)")
    return p


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def process_episode(ep, args, model=None) -> dict:
    """Reconstruct one episode and write its scene-depth artifacts."""
    import zarr

    t0 = time.time()
    out_dir = Path(args.output_dir) / ep
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_path = Path(args.bg_dir) / ep / "bg.mp4"

    n = scene_io.resolve_n_frames(args.state_dir, args.zarr_dir, ep)
    z = zarr.open_group(str(Path(args.zarr_dir) / ep), mode="r")
    attrs = dict(z.attrs)
    fps = float(args.fps or attrs.get("fps") or 30.0)

    # 1. Sample frames + poses (raw zarr indices; aligned with retarget).
    frames, idx = scene_io.read_frames_sampled(
        args.zarr_dir, ep, n, sample_fps=args.sample_fps, fps=fps)
    print(f"  [{ep}] frames: n={n} sampled={len(frames)} @ {args.sample_fps}fps "
          f"({time.time() - t0:.1f}s)", flush=True)

    # 2. Feed-forward reconstruction.
    dvlt = run_dvlt(model, frames, device=args.device,
                    img_size=args.img_size, patch_size=args.patch_size,
                    inference_steps=args.inference_steps)
    print(f"  [{ep}] dvlt: {dvlt.c2w.shape} views, depth {dvlt.depth.shape}, "
          f"conf median={np.nanmedian(dvlt.conf):.3f} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # 3. Diagnostic pose registration (informational only): monocular DVLT
    # poses carry an unobservable gauge, so this does not gate the pipeline.
    slam = scene_io.read_head_poses(args.zarr_dir, ep, n)[idx]
    dvlt_pose7 = _c2w_to_pose7(dvlt.c2w)
    sim, stats = align.align_pose_sequences(dvlt_pose7, slam, idx)
    # Scale: prefer a direct metric fit from world-frame hand keypoints
    # (exact, same trick as retarget's DA3 alignment), falling back to the
    # gauge-invariant trajectory length ratio.
    kp_world = None
    for side in ("left", "right"):
        if f"{side}.obs_keypoints" in z:
            k = np.asarray(z[f"{side}.obs_keypoints"][idx], dtype=np.float64)
            if k.size and np.isfinite(k).any():
                kp_world = k.reshape(len(idx), -1, 3)
                break
    if kp_world is not None:
        fit = align.estimate_scale_from_hand_keypoints(
            np.asarray(dvlt.depth, np.float64), dvlt.K, slam, kp_world)
        if fit["mode"] == "keypoints":
            scale = float(fit["scale"])
        else:
            scale = align.trajectory_scale(dvlt_pose7, slam)
        print(f"  [{ep}] scale: {scale:.4f} (keypoints n={fit['n_pairs']} "
              f"resid={fit['resid_ratio']:.1%} traj="
              f"{align.trajectory_scale(dvlt_pose7, slam):.4f} "
              f"umeyama={sim.scale:.4f})", flush=True)
    else:
        scale = align.trajectory_scale(dvlt_pose7, slam)
        print(f"  [{ep}] scale: {scale:.4f} (trajectory only)", flush=True)

    # 4. Metric world points: unproject DVLT's per-view depth through the SLAM
    # poses of the same frames (pose-independent depth; see align docstring).
    wp = align.unproject_slam_world(dvlt.depth, dvlt.K, slam, scale=scale)
    points_w = _sample_points(wp, dvlt.conf, args.conf_threshold,
                              args.points_per_frame, seed=args.seed).astype(np.float64)
    print(f"  [{ep}] points: {len(points_w)} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # 5. Effective scene size (mirrors DA3 cropping behavior).  When no bg
    # video is available (standalone gauss-recon before inpainting), use the
    # source zarr frame size; the projection target only needs to match what
    # the retarget step later composes against.
    if bg_path.exists():
        H, W, crop_top, crop_bottom = scene_io.bg_effective_size(str(bg_path))
    elif len(frames):
        fh, fw = np.asarray(frames[0]).shape[:2]
        H, W, crop_top, crop_bottom = fh, fw, 0, 0
    else:
        H, W, crop_top, crop_bottom = dvlt.height, dvlt.width, 0, 0
    # Projection intrinsics: the DVLT-consistent field of view (its predicted
    # focal length scaled to the target height).  An explicit --fy wins so the
    # user can pin the retarget/synthesis cameras to the same value.
    fy_proj = scene_io.resolve_fy(args.zarr_dir, ep, n, args.fy) if args.fy else (
        float(np.median(dvlt.K[:, 1, 1])) * H / dvlt.height)
    K = scene_io.make_k(fy_proj, W, H)

    # 6. Project for every aligned frame.
    head_all = scene_io.read_head_poses(args.zarr_dir, ep, n)
    scene_depth = np.full((n, H, W), np.nan, np.float32)
    for t in range(n):
        d, _ = align.project_points_to_depth(points_w, head_all[t], K, H, W)
        scene_depth[t] = align.fill_depth_holes(d, iterations=args.hole_fill_iters)
    coverage = float(np.isfinite(scene_depth).mean())
    if coverage < 0.5:
        print(f"  [WARN] {ep}: depth coverage {coverage:.1%} is low", flush=True)
    print(f"  [{ep}] projected {n} frames, median depth="
          f"{np.nanmedian(scene_depth):.3f}m coverage={coverage:.1%} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # 7. Write outputs.
    export.write_scene_depth_npz(
        out_dir / "scene_depth.npz", scene_depth.astype(np.float32),
        meta={"model": "dvlt", "height": H, "width": W,
              "crop_top": crop_top, "crop_bottom": crop_bottom, "K": K})
    export.write_dvlt_cache(out_dir / f"{ep}_dvlt.pt", {
        "episode": ep, "n_frames": n, "sample_indices": idx,
        "c2w": dvlt.c2w, "K": dvlt.K, "depth": dvlt.depth,
        "world_points": dvlt.world_points, "conf": dvlt.conf,
        "img_size": dvlt.img_size,
        "similarity": {"scale": scale, "R": sim.R, "t": sim.t},
        "align_stats": stats.as_dict(),
        "fps": fps, "fy": float(K[0, 0]),
    })
    frames_rgb = np.stack([np.asarray(f) for f in frames])
    export.write_recon_vis(out_dir / "recon_vis.mp4", frames_rgb,
                           dvlt.depth, step=5)
    export.write_meta(out_dir / "meta.json", {
        "episode": ep, "backend": "gauss", "n_frames": n,
        "sample_fps": args.sample_fps, "fps": fps,
        "conf_threshold": args.conf_threshold,
        "points_per_frame": args.points_per_frame,
        "coverage": coverage,
        "align": stats.as_dict(),
        "scale": float(scale), "R": sim.R.tolist(), "t": sim.t.tolist(),
        "K": K.tolist(), "scene_shape": [int(n), int(H), int(W)],
        "crop": [int(crop_top), int(crop_bottom)],
        "checkpoint": str(args.checkpoint),
        "fy_used": float(K[1, 1]),
        "fy_recommended_retarget": float(np.median(dvlt.K[:, 1, 1])) * H / dvlt.height,
        "elapsed_s": round(time.time() - t0, 1),
    })
    print(f"  ✓ {out_dir} ({time.time() - t0:.1f}s)", flush=True)
    return {"coverage": coverage, "scale": float(scale),
            "rot_err_deg": float(stats.rot_err_deg)}


def check_env(args) -> int:
    """Environment self-check (imports, checkpoint, GPU) without reconstruction."""
    ok = True
    try:
        import torch  # noqa: F401
        print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
              f"devices={torch.cuda.device_count()}")
    except Exception as e:
        print(f"torch import failed: {e}")
        ok = False
    try:
        from dvlt.util.preprocess import load_sequence  # noqa: F401
        from dvlt.model.dvlt.model import DVLT  # noqa: F401
        print("dvlt package: OK")
    except Exception as e:
        print(f"dvlt package import failed: {e}")
        ok = False
    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        files = sorted(p.name for p in ckpt.iterdir())
        print(f"checkpoint {ckpt}: {files}")
        if "config.json" not in files or not any(
                n.endswith(".safetensors") for n in files):
            print("  [WARN] expected config.json + model.safetensors")
    else:
        print(f"checkpoint {args.checkpoint}: not a local directory "
              f"(assuming HF hub id; requires network)")
    return 0 if ok else 1


def run(args):
    if getattr(args, "check", False):
        return check_env(args)
    for name in ("zarr_dir", "state_dir", "bg_dir", "output_dir"):
        if not getattr(args, name):
            raise SystemExit(f"gauss-recon: --{name.replace('_', '-')} is required")
    eps = args.episodes or sorted(
        p.name for p in Path(args.zarr_dir).iterdir()
        if (p / "zarr.json").exists())
    print(f"episodes: {len(eps)}  device={args.device}  "
          f"checkpoint={args.checkpoint}")
    model = load_model(args.checkpoint, img_size=args.img_size,
                       patch_size=args.patch_size, device=args.device,
                       dinov2_weights=args.dinov2_weights)
    n_ok = 0
    for i, ep in enumerate(eps, 1):
        print(f"[{i}/{len(eps)}] {ep}", flush=True)
        try:
            process_episode(ep, args, model=model)
            n_ok += 1
        except Exception as e:
            print(f"  [FAIL] {ep}: {type(e).__name__}: {e}", flush=True)
            raise
    print(f"\ndone: {n_ok}/{len(eps)} episodes")
    return n_ok


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    if getattr(args, "check", False):
        raise SystemExit(check_env(args))
    run(args)


if __name__ == "__main__":
    main()
