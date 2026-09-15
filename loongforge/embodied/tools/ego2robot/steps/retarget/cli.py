# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry points for dual-arm retargeting."""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

try:
    from ..robot_registry import ROBOT_SPECS, get_robot_spec
except ImportError:
    from robot_registry import ROBOT_SPECS, get_robot_spec

from .episode import process_episode
from .ik import MINK_CONTINUITY_COST, MINK_CONTINUITY_MAX_STEP


def build_arg_parser():
    """Build the morphology-aware retarget argument parser."""
    ap = argparse.ArgumentParser(description="Step 6: dual-arm morphology retargeting + rendering")
    ap.add_argument("--robot_type", choices=tuple(sorted(ROBOT_SPECS)), default="panda",
                    help="target dual-arm morphology (default: panda)")
    ap.add_argument("--zarr_dir", required=True)
    ap.add_argument("--bg_dir", required=True,
                    help="Step 4 inpaint output directory containing {ep}/bg.mp4")
    ap.add_argument("--state_dir", required=True,
                    help="Use the valid-frame prefix length from state; values come from world-frame zarr")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--feather", type=int, default=5)
    ap.add_argument("--scene_depth_dir", default=None,
                    help="Depth step output directory containing {ep}/scene_depth.npz")
    ap.add_argument("--depth_epsilon", type=float, default=0.02)
    ap.add_argument("--depth_mode", choices=("depth-aware", "alpha"), default="depth-aware")
    ap.add_argument(
        "--ik_solver", choices=("dls", "mink"), default="mink",
        help=("Trajectory and base-search IK backend: mink uses differential "
              "QP IK; dls uses the original MuJoCo Jacobian solver."),
    )
    ap.add_argument(
        "--base_search_mode",
        choices=("original", "balanced", "slow", "fast"),
        default="balanced",
        help=("Base Pose Search strategy: balanced is the default multi-orientation "
              "strategy; fast uses a single-orientation screen; slow uses a more exhaustive strategy; "
              "original restores the hand-anchored fixed-orientation search."),
    )
    ap.add_argument(
        "--enable_base_orientation_search", action="store_true",
        help=("Enable balanced base pitch/yaw/roll search; by default "
              "balanced searches translation only and keeps the nominal "
              "camera-facing base orientation."),
    )
    ap.add_argument("--base_pullback", type=float, default=0.0,
                    help="Move both arm mounts opposite the head viewing direction in meters (disabled by default)")
    ap.add_argument("--fy", type=float, default=None,
                    help="Camera vertical focal length in pixels; inferred from dataset intrinsics by default, "
                         "falling back to 490.1961")
    ap.add_argument(
        "--max_jump_rad",
        type=float,
        default=0.0,
        help="Maximum per-frame IK joint displacement in radians; 0 disables clamping",
    )
    ap.add_argument("--mink_continuity_max_step", type=float,
                    default=MINK_CONTINUITY_MAX_STEP,
                    help=("Mink maximum joint displacement from the previous "
                          "frame during one solve, in radians (0 disables the "
                          "hard continuity bound)"))
    ap.add_argument("--mink_continuity_cost", type=float,
                    default=MINK_CONTINUITY_COST,
                    help="Mink soft posture cost toward the previous frame (0 disables it)")
    ap.add_argument("--target_smooth_window", type=int, default=11,
                    help="Savitzky-Golay window for Eq.1 positions/widths (0 disables smoothing)")
    ap.add_argument("--target_smooth_polyorder", type=int, default=3,
                    help="Savitzky-Golay polynomial order for retarget targets")
    ap.add_argument("--target_orientation_sigma", type=float, default=6.0,
                    help="Gaussian quaternion smoothing sigma in frames (0 disables it)")
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--episodes", nargs="*")
    return ap


def run(args):
    """Process all episodes for one selected morphology."""
    spec = get_robot_spec(getattr(args, "robot_type", "panda"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_files = sorted(glob.glob(os.path.join(args.state_dir, "*.npz")))
    if args.episodes:
        state_files = [f for f in state_files if Path(f).stem in args.episodes]
    print(f"Robot morphology: {spec.name} ({spec.state_dim} state dims, reach={spec.reach:.3f}m)")
    print(f"Processing {len(state_files)} episodes\n")

    summary = []
    for i, sf in enumerate(state_files):
        ep = Path(sf).stem
        bg = Path(args.bg_dir) / ep / "bg.mp4"
        if not bg.exists():
            print(f"[{i+1}/{len(state_files)}] {ep} — SKIP (no bg.mp4)")
            continue
        print(f"[{i+1}/{len(state_files)}] {ep}")
        st = np.load(sf, allow_pickle=True)
        state = st["state"]
        try:
            depth_path = (Path(args.scene_depth_dir) / ep / "scene_depth.npz"
                          if args.scene_depth_dir else None)
            res = process_episode(
                ep, args.zarr_dir, str(bg), state, out_dir,
                args.fps, args.feather, args.fy, args.height, args.debug,
                scene_depth_path=str(depth_path) if depth_path else None,
                depth_epsilon=args.depth_epsilon, depth_mode=args.depth_mode,
                base_search_mode=getattr(args, "base_search_mode", "balanced"),
                enable_base_orientation_search=getattr(
                    args, "enable_base_orientation_search", False),
                ik_solver=getattr(args, "ik_solver", "mink"),
                base_pullback=getattr(args, "base_pullback", 0.0),
                max_jump_rad=getattr(args, "max_jump_rad", 0.0),
                mink_continuity_max_step=getattr(
                    args, "mink_continuity_max_step", MINK_CONTINUITY_MAX_STEP),
                mink_continuity_cost=getattr(
                    args, "mink_continuity_cost", MINK_CONTINUITY_COST),
                target_smooth_window=getattr(args, "target_smooth_window", 11),
                target_smooth_polyorder=getattr(args, "target_smooth_polyorder", 3),
                target_orientation_sigma=getattr(args, "target_orientation_sigma", 6.0),
                spec=spec)
            summary.append({"episode": ep, "robot_type": spec.name, "output": res})
        except Exception as e:
            import traceback
            traceback.print_exc()
            if args.depth_mode == "depth-aware":
                raise
            summary.append({"episode": ep, "output": None, "error": str(e)})

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. {len([s for s in summary if s['output']])} episodes produced.")


def main():
    """Entry point for the retarget command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
