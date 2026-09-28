# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
run-all orchestration: call each step's run(args) in order, automatically feed
one step's output directory into the next, and store intermediate artifacts in
fixed subdirectories under --output_dir.

Pipeline (--with_depth inserts depth after inpaint for z-buffer compositing):
    load -> align -> mask -> inpaint [-> depth] -> retarget -> lerobot -> demo

Fixed subdirectory names:
    01_load/   02_align/   03_mask/   04_inpaint/
    05_depth/ (optional)   06_retarget/   07_lerobot/   08_demo/

Individual commands (load/align/mask/inpaint/depth/retarget/lerobot/validate/demo
in cli.py) still require all input and output paths explicitly. This supports
isolated debugging and nonstandard directory layouts. run-all is simply a fixed
orchestration of those individual run() functions.

Usage:
    python cli.py run-all --input_dir <zarr-dir> --output_dir <output-root> [--episodes ...]
    python cli.py run-all --input_dir <zarr-dir> --output_dir <output-root> --with_depth
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from steps import (
    loader,
    action_alignment,
    hand_mask,
    inpaint,
    depth_estimate,
    robot_retarget,
    lerobot_writer,
    demo_grid,
)
from steps.robot_registry import ROBOT_SPECS


def build_arg_parser():
    """Build the argument parser for the run-all subcommand."""
    ap = argparse.ArgumentParser(
        description="run-all: convert EgoVerse zarr to a LeRobot v3.0 dataset and 3x3 demo in one command"
    )
    ap.add_argument("--input_dir", required=True, help="Raw EgoVerse zarr directory (for example, bimanual_sample/)")
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Output root containing intermediate artifacts in fixed subdirectories",
    )
    ap.add_argument("--episodes", nargs="*", default=None, help="Episode directory names to process (default: all)")
    ap.add_argument("--robot_type", choices=tuple(sorted(ROBOT_SPECS)), default="panda",
                    help="Target dual-arm morphology (default: panda)")
    ap.add_argument("--ik_solver", choices=("dls", "mink"), default="mink",
                    help="IK solver; mink uses differential QP (default), while dls uses the MuJoCo Jacobian")
    ap.add_argument("--base_search_mode",
                    choices=("original", "balanced", "slow", "fast"),
                    # default="original",
                    default="balanced",
                    help="Base-pose search mode (default: balanced)")
    ap.add_argument(
        "--enable_base_orientation_search", action="store_true",
        help="Use the exhaustive paper pitch/yaw/roll grid.",
    )
    ap.add_argument(
        "--trajectory_orientation_base_search",
        action=argparse.BooleanOptionalAction, default=False,
        help=("Search six bounded base orientations using whole-trajectory "
              "pose reachability (default: disabled)."),
    )
    ap.add_argument("--base_pullback", type=float, default=0.0,
                    help="Move both arm mounts backward from the head gaze direction, in meters (default: disabled)")
    ap.add_argument("--max_jump_rad", type=float, default=0.0,
                    help="Maximum per-frame joint displacement during IK postprocessing, in radians (0 disables it)")
    ap.add_argument(
        "--mink_continuity_max_step",
        type=float,
        default=0.35,
        help=(
            "Mink maximum joint displacement from the previous frame during one solve, "
            "in radians (0 disables the hard bound)"
        ),
    )
    ap.add_argument("--mink_continuity_cost", type=float, default=0.2,
                    help="Mink soft posture cost toward the previous frame (0 disables it)")
    ap.add_argument("--target_smooth_window", type=int, default=11,
                    help="Savitzky-Golay window for retarget targets (0 disables smoothing)")
    ap.add_argument("--target_smooth_polyorder", type=int, default=3,
                    help="Savitzky-Golay polynomial order for retarget targets")
    ap.add_argument("--target_orientation_sigma", type=float, default=6.0,
                    help="Gaussian quaternion smoothing sigma in frames (0 disables it)")
    ap.add_argument("--no_trajectory_refine", action="store_true",
                    help="Disable whole-episode refinement after frame-wise Mink IK")
    ap.add_argument("--trajectory_refine_position_weight", type=float, default=1.0)
    ap.add_argument("--trajectory_refine_orientation_weight", type=float, default=0.05)
    ap.add_argument("--trajectory_refine_velocity_weight", type=float, default=0.2)
    ap.add_argument("--trajectory_refine_acceleration_weight", type=float, default=1.0)
    ap.add_argument("--trajectory_refine_home_weight", type=float, default=0.002)
    ap.add_argument("--trajectory_refine_joint_margin_weight", type=float, default=0.05)
    ap.add_argument("--trajectory_refine_joint_margin_fraction", type=float, default=0.10)
    ap.add_argument("--trajectory_refine_collision_weight", type=float, default=4.0)
    ap.add_argument("--trajectory_refine_collision_safe_distance", type=float, default=0.015)
    ap.add_argument("--trajectory_refine_position_tolerance", type=float, default=0.005,
                    help="Hard per-frame TCP position tolerance in meters")
    ap.add_argument("--trajectory_refine_max_nfev", type=int, default=20)
    ap.add_argument(
        "--primary_trajectory_opt", action="store_true",
        help=("Gray-scale switch (Scope C): make whole-trajectory optimization "
              "the primary producer of the retarget trajectory (analytic "
              "Jacobian; frame-wise Mink pass becomes a branch initializer). "
              "Disabled by default."))
    ap.add_argument(
        "--trajectory_refine_primary_max_nfev", type=int, default=100,
        help="least_squares max_nfev used when --primary_trajectory_opt is set")

    # Expose the few required downstream options that cannot use convenient
    # defaults at the run-all level. Other tunable options retain each step's
    # defaults, matching the individual commands.
    ap.add_argument("--target_fps", type=float, default=30.0, help="Step 1 target FPS")
    ap.add_argument("--action_skip", type=int, default=5, help="Step 2 action lookahead steps")
    ap.add_argument("--sam3_checkpoint", default=None,
                    help="Step 3 SAM3 checkpoint (or set EGO2ROBOT_SAM3_CKPT)")
    ap.add_argument("--device", default="cuda:0", help="Step 3 inference device")
    ap.add_argument("--body_prompt", default="person", help="SAM3 text prompt for visible human body")
    ap.add_argument("--mask_mode", choices=("both", "person", "arms"), default="both",
                    help="SAM3 tracks: both, person text only, or arm boxes only")
    ap.add_argument("--no_body", action="store_true", help="Deprecated alias for --mask_mode arms")
    ap.add_argument(
        "--with_depth",
        action="store_true",
        help=(
            "Insert DA3 depth estimation after inpaint and write 05_depth/; "
            "requires the official depth-anything-3 package"
        ),
    )
    ap.add_argument("--depth_mode", choices=("depth-aware", "alpha"), default="depth-aware",
                    help="depth-aware uses scene/robot depth for occlusion; alpha uses the original mask composite")
    ap.add_argument(
        "--depth_epsilon",
        type=float,
        default=0.02,
        help="Depth occlusion margin in meters; the robot must be closer than the background by this amount",
    )
    ap.add_argument("--depth_temporal_window", type=int, default=5,
                    help="Positive odd temporal median window for depth compositing")
    ap.add_argument("--depth_transition_width", type=float, default=0.04,
                    help="Smooth depth-occlusion transition half-width in meters")
    ap.add_argument(
        "--depth_align", action="store_true",
        help="Experimental hand-keypoint alignment for DA3 depth (off by default)")
    ap.add_argument("--skip_depth", action="store_true",
                    help="Skip depth estimation and force alpha compositing (fallback for --with_depth)")
    ap.add_argument(
        "--no_scene_support", action="store_true",
        help=("Keep depth-aware compositing but stop base search from snapping "
              "the mount to the estimated scene support surface. Use this when "
              "the metric depth and the hand-target reconstruction are on "
              "inconsistent scales (e.g. the monocular Path B pipeline)."))
    return ap


def run(args):
    """Run load->align->mask->inpaint[->depth]->retarget->lerobot->demo."""
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_fps = float(getattr(args, "target_fps", 30.0))
    if raw_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {raw_fps}")
    # ProPainter's --save_fps only accepts an integer. Use one normalized rate
    # for state resampling, intermediate videos, LeRobot timestamps, and demo.
    pipeline_fps = max(1, int(round(raw_fps)))

    requested_with_depth = bool(getattr(args, "with_depth", False))
    requested_depth_mode = getattr(args, "depth_mode", "depth-aware")
    skip_depth = bool(getattr(args, "skip_depth", False))
    # Depth is both estimated and consumed only in depth-aware mode.  The
    # explicit alpha/skip switches keep the previous mask-only pipeline intact.
    with_depth = requested_with_depth and requested_depth_mode == "depth-aware" and not skip_depth
    effective_depth_mode = "depth-aware" if with_depth else "alpha"
    dir_load = root / "01_load"
    dir_align = root / "02_align"
    dir_mask = root / "03_mask"
    dir_inpaint = root / "04_inpaint"
    dir_depth = root / "05_depth"
    dir_retarget = root / "06_retarget"
    dir_lerobot = root / "07_lerobot"
    dir_demo = root / "08_demo"
    n_steps = 8

    print("=" * 60)
    print(f"[1/{n_steps}] Load: EgoVerse Loader + Cleaning")
    print("=" * 60)
    loader.run(SimpleNamespace(
        input_dir=args.input_dir,
        output_dir=str(dir_load),
        target_fps=pipeline_fps,
    ))

    print("\n" + "=" * 60)
    print(f"[2/{n_steps}] Align: Simplified Action Alignment")
    print("=" * 60)
    action_alignment.run(SimpleNamespace(
        input_dir=str(dir_load),
        output_dir=str(dir_align),
        skip=args.action_skip,
        no_smooth=False,
        smooth_window=11,
    ))

    print("\n" + "=" * 60)
    print(f"[3/{n_steps}] Mask: SAM3 Human/Hand/Arm Mask")
    print("=" * 60)
    hand_mask.run(SimpleNamespace(
        zarr_dir=args.input_dir,
        output_dir=str(dir_mask),
        sam3_checkpoint=args.sam3_checkpoint,
        device=args.device,
        fps=pipeline_fps,
        dilation_px=0,
        body_prompt=args.body_prompt,
        mask_mode=args.mask_mode,
        no_body=args.no_body,
        episodes=args.episodes,
        max_frames=None,
    ))

    print("\n" + "=" * 60)
    print(f"[4/{n_steps}] Inpaint: Video Inpainting (ProPainter)")
    print("=" * 60)
    inpaint.run(SimpleNamespace(
        zarr_dir=args.input_dir,
        mask_dir=str(dir_mask),
        output_dir=str(dir_inpaint),
        episodes=args.episodes,
        mask_dilation=4,
        fps=pipeline_fps,
        save_frames=False,
        max_frames=0,
        fp16=True,
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=80,
        raft_iter=20,
    ))

    if with_depth:
        print("\n" + "=" * 60)
        print(f"[5/{n_steps}] Depth: DA3 depth estimation on inpainted bg")
        print(f"            output -> {dir_depth} (the default model follows "
              f"EGO2ROBOT_DA3_MODEL_DIR / official DA3-BASE weights; "
              f"see cli.py depth --help)")
        print("=" * 60)
        depth_estimate.run(SimpleNamespace(
            input_dir=str(dir_inpaint),
            output_dir=str(dir_depth),
            model_id=None,
            backend="auto",
            device=args.device,
            batch=4,
            dtype="fp16",
            process_res=504,
            episodes=args.episodes,
        ))

    print("\n" + "=" * 60)
    print(f"[6/{n_steps}] Retarget: {args.robot_type} dual-arm retargeting + rendering + composite")
    print("=" * 60)
    robot_retarget.run(SimpleNamespace(
        zarr_dir=args.input_dir,
        bg_dir=str(dir_inpaint),
        mask_dir=str(dir_mask),
        state_dir=str(dir_align),
        output_dir=str(dir_retarget),
        scene_depth_dir=str(dir_depth) if with_depth else None,
        depth_mode=effective_depth_mode,
        depth_epsilon=float(getattr(args, "depth_epsilon", 0.02)),
        depth_temporal_window=int(getattr(args, "depth_temporal_window", 5)),
        depth_transition_width=float(getattr(args, "depth_transition_width", 0.04)),
        human_hand_mask_dilation=8,
        depth_align=getattr(args, "depth_align", False),
        robot_type=args.robot_type,
        base_search_mode=args.base_search_mode,
        enable_base_orientation_search=getattr(
            args, "enable_base_orientation_search", False),
        trajectory_orientation_base_search=getattr(
            args, "trajectory_orientation_base_search", False),
        ik_solver=getattr(args, "ik_solver", "mink"),
        base_pullback=getattr(args, "base_pullback", 0.0),
        max_jump_rad=getattr(args, "max_jump_rad", 0.0),
        mink_continuity_max_step=getattr(args, "mink_continuity_max_step", 0.35),
        mink_continuity_cost=getattr(args, "mink_continuity_cost", 0.2),
        target_smooth_window=getattr(args, "target_smooth_window", 11),
        target_smooth_polyorder=getattr(args, "target_smooth_polyorder", 3),
        target_orientation_sigma=getattr(args, "target_orientation_sigma", 6.0),
        no_trajectory_refine=getattr(args, "no_trajectory_refine", False),
        trajectory_refine_position_weight=getattr(
            args, "trajectory_refine_position_weight", 1.0),
        trajectory_refine_orientation_weight=getattr(
            args, "trajectory_refine_orientation_weight", 0.05),
        trajectory_refine_velocity_weight=getattr(
            args, "trajectory_refine_velocity_weight", 0.2),
        trajectory_refine_acceleration_weight=getattr(
            args, "trajectory_refine_acceleration_weight", 1.0),
        trajectory_refine_home_weight=getattr(
            args, "trajectory_refine_home_weight", 0.002),
        trajectory_refine_joint_margin_weight=getattr(
            args, "trajectory_refine_joint_margin_weight", 0.05),
        trajectory_refine_joint_margin_fraction=getattr(
            args, "trajectory_refine_joint_margin_fraction", 0.10),
        trajectory_refine_collision_weight=getattr(
            args, "trajectory_refine_collision_weight", 4.0),
        trajectory_refine_collision_safe_distance=getattr(
            args, "trajectory_refine_collision_safe_distance", 0.015),
        trajectory_refine_position_tolerance=getattr(
            args, "trajectory_refine_position_tolerance", 0.005),
        trajectory_refine_max_nfev=getattr(
            args, "trajectory_refine_max_nfev", 20),
        primary_trajectory_opt=getattr(
            args, "primary_trajectory_opt", False),
        trajectory_refine_primary_max_nfev=getattr(
            args, "trajectory_refine_primary_max_nfev", 100),
        no_scene_support=getattr(args, "no_scene_support", False),
        fps=pipeline_fps,
        feather=5,
        fy=None,  # Derive from each episode's intrinsics; fall back to 490.1961.
        height=360,
        debug=False,
        episodes=args.episodes,
    ))

    print("\n" + "=" * 60)
    print(f"[7/{n_steps}] LeRobot: LeRobot v3.0 Dataset Writer")
    print("=" * 60)
    lerobot_writer.run(SimpleNamespace(
        ik_dir=str(dir_retarget),
        state_dir=str(dir_align),
        bg_video_dir=str(dir_retarget),
        output_dir=str(dir_lerobot),
        tmp_dir=str(dir_lerobot / "_tmp_av1"),
        episodes=args.episodes,
        fps=pipeline_fps,
    ))

    print("\n" + "=" * 60)
    print(f"[8/{n_steps}] Demo: 3x3 split-screen video")
    print("=" * 60)
    demo_grid.run(SimpleNamespace(
        zarr_dir=args.input_dir,
        mask_dir=str(dir_mask),
        inpaint_dir=str(dir_inpaint),
        ik_dir=str(dir_retarget),
        depth_dir=str(dir_depth) if with_depth else None,
        output_dir=str(dir_demo),
        episodes=args.episodes,
        fps=pipeline_fps,
    ))

    print("\n" + "=" * 60)
    print(f"run-all complete. LeRobot dataset -> {dir_lerobot}")
    print(f"                  Split-screen demo video -> {dir_demo}")
    print("=" * 60)


def main():
    """Entry point for the run-all command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
