# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Step 4: Video Inpainting - remove the person, hands, and forearms segmented by SAM3
from the ego video and reconstruct a clean background.

Input:
  --zarr_dir   EgoVerse Zarr episode directory (original RGB frames, images.front_1 JPEG)
  --mask_dir   Step 3 SAM3 output directory (one masks.npz per episode, key='masks', (T,H,W) uint8 0/1)

Output (one subdirectory per episode):
  bg/*.png          inpainted background frames (optional, --save_frames)
  bg.mp4            inpainted background video
  debug_3panel.mp4  original | green mask overlay | inpainted background
  meta.json         frame count, mask-area statistics, and elapsed time

Implementation:
  Reuse the official ProPainter inference_propainter.py CLI. Decode Zarr frames and
  masks into PNG sequences, feed them to the CLI, and collect its output. This avoids
  reimplementing optical flow and propagation logic or drifting from the official code.
  config.py centrally manages the ProPainter repository and staging paths, which can
  be overridden with environment variables.

Usage:
    python cli.py inpaint \
        --zarr_dir data_input/egoverse/narrow_tabletop \
        --mask_dir data_output/03_mask \
        --output_dir data_output/04_inpaint \
        --mask_dilation 4 --fp16 --ref_stride 10 --neighbor_length 10 \
        --subvideo_length 80 --raft_iter 20 --fps 30
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config

try:
    import zarr
    import simplejpeg
except ImportError as e:
    print(f"missing dep: {e}", file=sys.stderr)
    raise


def unwrap(v):
    """Unwrap JPEG bytes nested in one or more 0-D arrays in a Zarr object array."""
    while isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, np.ndarray) and v.dtype == object:
        v = unwrap(v.flat[0])
    return v


def load_zarr_frames(ep_dir: Path):
    """Read the front_1 image sequence from an episode's Zarr directory."""
    store = zarr.open_group(str(ep_dir), mode="r")
    imgs = store["images.front_1"]
    n = int(store.attrs["total_frames"])
    frames = [simplejpeg.decode_jpeg(unwrap(imgs[t]), colorspace="RGB") for t in range(n)]
    return frames


def stage_inputs(frames, masks, work_dir: Path):
    """Write frames and masks as PNG sequences for the ProPainter CLI."""
    fdir, mdir = work_dir / "frames", work_dir / "masks"
    fdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    for t, (rgb, m) in enumerate(zip(frames, masks)):
        cv2.imwrite(str(fdir / f"{t:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mdir / f"{t:05d}.png"), (m > 0).astype(np.uint8) * 255)
    return fdir, mdir


def run_propainter(
    frames_dir,
    masks_dir,
    out_root,
    width,
    height,
    mask_dilation,
    fps,
    fp16=True,
    ref_stride=10,
    neighbor_length=10,
    subvideo_length=80,
    raft_iter=20,
):
    """Run the ProPainter CLI with defaults matching the original Ego2Robot setup."""
    cmd = [
        sys.executable, "inference_propainter.py",
        "-i", str(frames_dir),
        "-m", str(masks_dir),
        "-o", str(out_root),
        "--mask_dilation", str(mask_dilation),
        "--ref_stride", str(ref_stride),
        "--neighbor_length", str(neighbor_length),
        "--subvideo_length", str(subvideo_length),
        "--raft_iter", str(raft_iter),
        "--save_fps", str(fps),
        "--save_frames",
    ]
    if fp16:
        cmd.append("--fp16")
    if width > 0 and height > 0:
        cmd += ["--width", str(width), "--height", str(height)]
    proc = subprocess.run(cmd, cwd=config.PROPAINTER_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ProPainter failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return proc.stdout


def build_debug_video(frames, masks, bg_frames, out_path: Path, fps: int):
    """Create a three-panel original/mask/background debug video."""
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 3, H))
    for rgb, m, bg in zip(frames, masks, bg_frames):
        orig_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = orig_bgr.copy()
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        mask3 = (m > 0)[:, :, None]
        overlay = np.where(mask3, (0.4 * overlay + 0.6 * green).astype(np.uint8), overlay)
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR) if bg.shape[:2] == (H, W) else cv2.resize(
            cv2.cvtColor(bg, cv2.COLOR_RGB2BGR), (W, H))
        panel = np.hstack([orig_bgr, overlay, bg_bgr])
        writer.write(panel)
    writer.release()
    try:
        from .config import reencode_h264
    except ImportError:
        from config import reencode_h264
    reencode_h264(str(out_path))


def episode_inpaint(ep_hash: str, zarr_dir: Path, mask_dir: Path, output_dir: Path,
                     mask_dilation: int, fps: int, save_frames: bool, max_frames: int = 0,
                     chunk_frames: int = 120, fp16: bool = True,
                     ref_stride: int = 10, neighbor_length: int = 10,
                     subvideo_length: int = 80, raft_iter: int = 20):
    """Run ProPainter on one episode and output bg.mp4, debug video, and meta.json.

    ProPainter loads the entire video into GPU memory at once, so long episodes with
    thousands of frames can run out of memory. Process chunk_frames at a time and
    concatenate the resulting frames into one complete video.
    """
    ep_out = output_dir / ep_hash
    ep_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames = load_zarr_frames(zarr_dir / ep_hash)
    npz = np.load(mask_dir / ep_hash / "masks.npz")
    masks = npz["masks"]
    n = min(len(frames), len(masks))
    if max_frames > 0:
        n = min(n, max_frames)
    frames, masks = frames[:n], masks[:n]
    H, W = frames[0].shape[:2]

    work_dir = Path(config.PROPAINTER_STAGE_ROOT) / ep_hash
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    bg_frames: list = []
    chunk_times = []
    for start in range(0, n, chunk_frames):
        end = min(start + chunk_frames, n)
        chunk_dir = work_dir / f"chunk_{start:06d}"
        chunk_rgb = frames[start:end]
        chunk_masks = masks[start:end]
        # ProPainter's RAFT path computes frame-to-frame flow and crashes when
        # a final chunk contains only one frame. Duplicate that frame for the
        # inference call, then keep only the original frame below.
        if len(chunk_rgb) == 1:
            chunk_rgb = chunk_rgb + chunk_rgb
            chunk_masks = np.concatenate((chunk_masks, chunk_masks), axis=0)
        fdir, mdir = stage_inputs(chunk_rgb, chunk_masks, chunk_dir)
        pp_out = chunk_dir / "pp_out"
        run_propainter(
            fdir,
            mdir,
            pp_out,
            W,
            H,
            mask_dilation,
            fps,
            fp16=fp16,
            ref_stride=ref_stride,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
            raft_iter=raft_iter,
        )

        result_dir = pp_out / "frames"
        frames_out_dir = result_dir / "frames"
        if not frames_out_dir.exists():
            raise RuntimeError(f"ProPainter chunk [{start}:{end}] output missing at {frames_out_dir}")
        for t in range(start, end):
            # ProPainter restarts zero-padded frame names for each chunk.
            p = frames_out_dir / f"{t - start:04d}.png"
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise RuntimeError(f"ProPainter frame output missing or unreadable: {p}")
            bg_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        elapsed = time.time() - t0
        chunk_times.append((start, end, round(elapsed, 1)))
        print(f"    chunk [{start}:{end}] done in {elapsed:.1f}s")
        # The merged RGB frames are already held in ``bg_frames``. Remove all
        # per-chunk PNG/MP4 intermediates before starting the next chunk to
        # keep peak disk usage bounded for multi-GPU runs.
        shutil.rmtree(chunk_dir, ignore_errors=True)

    # Chunking produces no single ProPainter MP4, so encode bg.mp4 from the merged frames.
    writer = cv2.VideoWriter(str(ep_out / "bg.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in bg_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    try:
        from .config import reencode_h264
    except ImportError:
        from config import reencode_h264
    reencode_h264(str(ep_out / "bg.mp4"))

    if save_frames:
        bg_dir = ep_out / "bg_frames"
        bg_dir.mkdir(exist_ok=True)
        for t, f in enumerate(bg_frames):
            cv2.imwrite(str(bg_dir / f"{t:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    build_debug_video(frames, masks, bg_frames, ep_out / "debug_3panel.mp4", fps)

    meta = {
        "episode": ep_hash,
        "n_frames": n,
        "width": W,
        "height": H,
        "mean_mask_area": float((masks > 0).mean()),
        "mask_dilation": mask_dilation,
        "chunks": [(s, e, f"{sec:.1f}s") for s, e, sec in chunk_times],
        "fp16": fp16,
        "ref_stride": ref_stride,
        "neighbor_length": neighbor_length,
        "subvideo_length": subvideo_length,
        "raft_iter": raft_iter,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (ep_out / "meta.json").write_text(json.dumps(meta, indent=2))

    shutil.rmtree(work_dir, ignore_errors=True)
    return meta


def build_arg_parser():
    """Build the argument parser for the inpaint subcommand."""
    ap = argparse.ArgumentParser(description="Step 4: Video Inpainting (ProPainter)")
    ap.add_argument("--zarr_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--episodes", nargs="*", default=None, help="Limit processing to these episode directory names")
    ap.add_argument(
        "--mask_dilation",
        type=int,
        default=4,
        help="ProPainter mask dilation (Ego2Robot: 4)",
    )
    ap.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fp16 during ProPainter inference (Ego2Robot: enabled)",
    )
    ap.add_argument("--ref_stride", type=int, default=10,
                    help="Stride of global reference frames")
    ap.add_argument("--neighbor_length", type=int, default=10,
                    help="Length of local neighboring frames")
    ap.add_argument("--subvideo_length", type=int, default=80,
                    help="Length of sub-video for long-video inference")
    ap.add_argument("--raft_iter", type=int, default=20,
                    help="Iterations for RAFT inference")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--save_frames", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0, help="Debugging: process only the first N frames; 0 means all")
    return ap


def run(args):
    """Run ProPainter video inpainting for every episode under --mask_dir."""
    zarr_dir = Path(args.zarr_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eps = args.episodes or sorted(
        d.name for d in mask_dir.iterdir() if d.is_dir() and (d / "masks.npz").exists()
    )
    print(f"{len(eps)} episodes -> {output_dir}")
    for i, ep in enumerate(eps):
        print(f"[{i+1}/{len(eps)}] {ep}")
        try:
            meta = episode_inpaint(
                ep, zarr_dir, mask_dir, output_dir,
                args.mask_dilation, args.fps, args.save_frames, args.max_frames,
                fp16=args.fp16,
                ref_stride=args.ref_stride,
                neighbor_length=args.neighbor_length,
                subvideo_length=args.subvideo_length,
                raft_iter=args.raft_iter,
            )
            print(f"  done: {meta['n_frames']}f, mask_area={meta['mean_mask_area']:.3f}, "
                  f"{meta['elapsed_sec']}s")
        except Exception as e:
            # A failed ProPainter invocation otherwise leaves hundreds of PNGs
            # in the staging tree, which can make the next worker fail with
            # ENOSPC even after the original process has exited.
            shutil.rmtree(Path(config.PROPAINTER_STAGE_ROOT) / ep, ignore_errors=True)
            print(f"  FAILED: {e}")


def main():
    """Entry point for the inpaint command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
