#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Step 5 - Depth Anything V3 Metric depth estimation on inpainted bg videos.

Estimate dense metric depth in meters for each inpainted background frame and output
scene_depth.npz and depth_vis.mp4. scene_depth.npz is used by morphology retargeting
for robot/background occlusion decisions during depth-aware compositing.

Input:  {input_dir}/{ep}/bg.mp4      (Step 4 inpainting output directory)
Output: {output_dir}/{ep}/scene_depth.npz  (float16, shape (N,360,640), unit=meters)
        {output_dir}/{ep}/depth_vis.mp4    (colormap visualization)

Two backends:
  - da3 (default): Official Depth Anything 3 weights (config.json + model.safetensors)
    through the official depth-anything-3 library (ByteDance-Seed/depth-anything-3,
    which is not on the official PyPI and requires git clone + `pip install -e .`).
    The weights directory can be a local path for offline use:
        python cli.py depth --input_dir data_output/04_inpaint \
            --output_dir data_output/05_depth \
            --model_id ~/DA3-BASE
    or an HF Hub model ID (for example, depth-anything/da3-base).
  - transformers: Standard Depth Anything V2/V3 HF weights
    (AutoModelForDepthEstimation; the directory must contain preprocessor_config.json):
        python cli.py depth --input_dir data_output/04_inpaint \
            --output_dir data_output/05_depth \
            --backend transformers \
            --model_id depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf

Backend detection rules (--backend auto): a local directory containing
preprocessor_config.json selects transformers; otherwise, including a directory in
the official format, it selects da3. An HF Hub repository ending in `-hf` selects
transformers; all others select da3.

Usage:
    python cli.py depth --input_dir data_output/04_inpaint --output_dir data_output/05_depth
"""
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import cv2
import torch

# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V3-Base-hf"  # transformers format
# bg.mp4 is 368x640 with H.264 padding; the actual scene is 360x640 with 4 px on each side.
BG_H, BG_W = 368, 640
CROP_TOP   = 4
SCENE_H, SCENE_W = 360, 640


def resolve_model_id(cli_arg):
    """Resolve --model_id from the environment or default when omitted."""
    from steps.config import resolve_da3_model_dir
    resolved = resolve_da3_model_dir(cli_arg)
    return resolved or DEFAULT_MODEL_ID


def build_arg_parser():
    """Build the argument parser for the depth subcommand."""
    p = argparse.ArgumentParser(description="Step 5: Depth Anything V3 depth estimation")
    p.add_argument("--input_dir", required=True, help="Step 4 inpainting output directory containing {ep}/bg.mp4")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default=None,
                   help="HF model ID or local directory: official DA3 weights (config.json + "
                        "model.safetensors, using the official depth-anything-3 library) or "
                        "transformers format (with preprocessor_config.json, using AutoModel). "
                        "When omitted, fall back to EGO2ROBOT_DA3_MODEL_DIR, then %(default)s")
    p.add_argument("--backend", default="auto", choices=["auto", "transformers", "da3"],
                   help="auto: select transformers if the directory contains preprocessor_config.json; otherwise da3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=4,
                   help="Batch size for depth inference (VRAM limited)")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--process_res", type=int, default=504,
                   help="DA3 inference processing resolution (official default: 504)")
    p.add_argument("--episodes", nargs="*", default=None,
                   help="Specific episode IDs; default=all in input_dir")
    return p


def detect_backend(model_id):
    """Auto-detect transformers format versus the official DA3 format."""
    if os.path.isdir(model_id):
        if os.path.exists(os.path.join(model_id, "preprocessor_config.json")):
            return "transformers"
        return "da3"
    if isinstance(model_id, str) and model_id.endswith("-hf"):
        return "transformers"
    return "da3"  # HF IDs without the -hf suffix use the official library.


def load_model(device, model_id=None, dtype="fp16", backend="auto"):
    """Load the depth model.

    - backend=transformers: Standard HF weights such as Depth Anything V2
      (AutoModelForDepthEstimation)
    - backend=da3: Official Depth Anything V3 weights
      (depth_anything_3.DepthAnything3); for offline use, a local directory with
      config.json and model.safetensors is sufficient.
    Returns (backend, proc_or_none, model).
    """
    model_id = resolve_model_id(model_id)
    if backend == "auto":
        backend = detect_backend(model_id)

    if backend == "transformers":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                       "fp32": torch.float32}[dtype]
        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        model = model.to(device).eval()
        return backend, proc, model

    # Official DA3 package (the entry point is depth_anything_3.api in 0.1.x).
    try:
        from depth_anything_3.api import DepthAnything3  # noqa: F401
    except ImportError:
        from depth_anything_3 import DepthAnything3  # noqa: F401
    try:
        model = DepthAnything3.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(
            f"DepthAnything3.from_pretrained failed: {model_id!r}\n"
            f"  Original error: {e}\n"
            f"Please confirm: 1) the official depth-anything-3 library is installed "
            f"(it is not on PyPI; run\n"
            f"  git clone https://github.com/ByteDance-Seed/depth-anything-3 && pip install -e .）\n"
            f"  2) the local directory contains config.json + model.safetensors "
            f"(for example, an extracted DA3-BASE directory), or\n"
            f"  3) use --backend transformers with standard HF weights") from None
    model.to(device)
    return backend, None, model


def read_video_frames(path):
    """Read mp4 → (N, H, W, 3) uint8 via ffmpeg pipe."""
    import subprocess as sp
    import re as _re
    # Detect video dimensions dynamically: legacy output is 368 px high with padding,
    # while the new H.264 pipeline emits 360 px directly. The container has only a
    # static ffmpeg build (no ffprobe), so parse the Video line from stderr.
    probe = sp.run(["ffmpeg", "-hide_banner", "-i", str(path), "-f", "null", "-"],
                   capture_output=True, text=True)
    m = _re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", probe.stderr or "")
    if m:
        v_w, v_h = int(m.group(1)), int(m.group(2))
    else:
        v_w, v_h = BG_W, BG_H
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    proc = sp.Popen(cmd, stdout=sp.PIPE)
    raw = proc.stdout.read()
    proc.wait()
    n_bytes = v_h * v_w * 3
    N = len(raw) // n_bytes
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(N, v_h, v_w, 3)
    # Only the legacy EgoDex output carries 4 px H.264 padding at the top and
    # bottom (368 -> 360). Every other source (e.g. the 640x480 Path B
    # background) is processed at its native resolution so the depth map matches
    # the background the retarget step reads back.
    if v_h == BG_H and CROP_TOP > 0:
        frames = frames[:, CROP_TOP:CROP_TOP + SCENE_H, :SCENE_W]
    return frames


def write_vis_video(path, depth_maps, fps=30):
    """Write depth colormap video (N, H, W) float → turbo-ish mp4."""
    import subprocess as sp
    H, W = depth_maps.shape[1], depth_maps.shape[2]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(path)
    ]
    proc = sp.Popen(cmd, stdin=sp.PIPE)
    # normalize per-video for visualization
    vmin = np.percentile(depth_maps, 1)
    vmax = np.percentile(depth_maps, 99)
    for d in depth_maps:
        norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0, 1)
        rgb = colormap_turbo(norm)
        proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    proc.wait()


def colormap_turbo(x):
    """Fast turbo-ish colormap: x in [0,1] → (H,W,3) uint8."""
    r = np.clip(1.5 - np.abs(x - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(x - 0.5) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(x - 0.25) * 4, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def run_depth_batch(proc, model, frames, device, batch_size=4):
    """transformers backend: (N,H,W,3) → (N,H,W) float meters."""
    from PIL import Image
    N = len(frames)
    H, W = frames.shape[1], frames.shape[2]
    depths = np.zeros((N, H, W), dtype=np.float32)

    for i in range(0, N, batch_size):
        batch_frames = frames[i:i + batch_size]
        images = [Image.fromarray(f) for f in batch_frames]
        inputs = proc(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        preds = out.predicted_depth if hasattr(out, "predicted_depth") else out[0]
        # bilinear interpolate to scene resolution
        preds_up = torch.nn.functional.interpolate(
            preds.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # (B, H, W)
        depths[i:i + len(batch_frames)] = preds_up.cpu().float().numpy()
    return depths


def run_depth_da3(model, frames, device, process_res=504):
    """Official DA3 backend: (N,H,W,3) -> (N,H,W) float meters per frame.

    The official InputProcessor accepts uint8 0-255 PIL/NumPy images; do not
    pre-normalize them.
    """
    N = len(frames)
    H, W = frames.shape[1], frames.shape[2]
    depths = np.zeros((N, H, W), dtype=np.float32)
    for i, img in enumerate(frames):
        pred = _da3_inference(model, img, process_res)
        d = np.asarray(pred.depth)  # (1, H, W) meters
        if d.ndim == 3:
            d = d[0]
        if d.shape[:2] != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        depths[i] = d.astype(np.float32)
    return depths


def _da3_inference(model, img, process_res):
    """Run model.inference on one frame, including versions without process_res."""
    try:
        return model.inference(
            [img],
            export_dir=None,
            process_res=process_res,
            process_res_method="upper_bound_resize",
        )
    except TypeError:
        # Older official releases do not accept process_res; retry without it.
        return model.inference([img], export_dir=None)


def process_episode(ep, input_dir, output_dir, backend, proc, model, device,
                    batch_size, model_tag=None, process_res=None):
    """Estimate depth for one episode's bg.mp4 and save NPZ and visualization output."""
    bg_path = Path(input_dir) / ep / "bg.mp4"
    if not bg_path.exists():
        print(f"  [SKIP] no bg.mp4: {bg_path}")
        return False

    out_dir = Path(output_dir) / ep
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames = read_video_frames(bg_path)
    print(f"  frames: {frames.shape}", flush=True)

    if backend == "da3":
        depths = run_depth_da3(model, frames, device, process_res or 504)
    else:
        depths = run_depth_batch(proc, model, frames, device, batch_size)
    dt = time.time() - t0

    med = float(np.median(depths))
    p01, p99 = float(np.percentile(depths, 1)), float(np.percentile(depths, 99))
    print(f"  depth: median={med:.3f}m  p1={p01:.3f}m  p99={p99:.3f}m  ({dt:.1f}s)",
          flush=True)

    np.savez_compressed(out_dir / "scene_depth.npz",
                        depth=depths.astype(np.float16),
                        model=str(model_tag or "unknown"),
                        unit="meters")
    write_vis_video(out_dir / "depth_vis.mp4", depths)

    sz = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"  ✓ {out_dir}  ({sz:.1f}MB)", flush=True)
    return True


def run(args):
    """Estimate depth for every episode under --input_dir."""
    eps = args.episodes or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.input_dir, "*"))
        if os.path.isdir(p))
    print(f"episodes: {len(eps)}  device={args.device}  batch={args.batch}")

    model_id = resolve_model_id(getattr(args, "model_id", None))
    backend, proc, model = load_model(args.device, model_id, args.dtype,
                                     args.backend)
    model_tag = str(Path(model_id).name)
    print(f"model loaded: {model_id}  backend={backend}  ({args.dtype})",
          flush=True)

    n_ok = 0
    for i, ep in enumerate(eps, 1):
        print(f"[{i}/{len(eps)}] {ep}", flush=True)
        if process_episode(ep, args.input_dir, args.output_dir, backend,
                           proc, model, args.device, args.batch,
                           model_tag=model_tag,
                           process_res=getattr(args, "process_res", None)):
            n_ok += 1
    print(f"\ndone: {n_ok}/{len(eps)} episodes")


def main():
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
