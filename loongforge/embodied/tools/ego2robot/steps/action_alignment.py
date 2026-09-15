# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Step 2 Path A: Simplified Action Alignment

Uses EgoVerse's existing obs_ee_pose (palm-center pose) and computes
gripper_width from keypoints. This applies only to datasets that provide EEF
poses, such as EgoVerse.

Input: .npz files produced by Step 1 (head-relative coordinates).
Output: state/action .npz for each episode.

state[t] = concat(left_ee_pose(7), right_ee_pose(7), left_gripper(1), right_gripper(1))  # (16,)
action[t] = state[t + skip]  # absolute single-step

Usage:
  python cli.py align --input_dir ./data_output/01_load --output_dir ./data_output/02_align --skip 5

References:
  - MANO keypoint order: thumb_tip=kp[4], index_tip=kp[8], middle_tip=kp[12]
  - Qwen-RobotManip virtual finger: k_vf = 0.7*index_tip + 0.3*middle_tip
  - gripper_width = ||thumb_tip - k_vf||
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter


# ============================================================
# 1. Gripper Width Computation
# ============================================================

# Fingertip indices in MANO canonical order.
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8
MIDDLE_TIP_IDX = 12


def compute_gripper_width(keypoints_seq: np.ndarray) -> np.ndarray:
    """
    Compute gripper opening width from 21 MANO keypoints.

    Qwen-RobotManip method:
      k_vf = 0.7 * index_tip + 0.3 * middle_tip  (virtual finger)
      gripper_width = ||thumb_tip - k_vf||

    Args:
        keypoints_seq: (N, 63) — 21 keypoints x 3, head-relative coordinates

    Returns:
        (N,) gripper width in meters
    """
    N = keypoints_seq.shape[0]
    kp = keypoints_seq.reshape(N, 21, 3)

    thumb_tip = kp[:, THUMB_TIP_IDX, :]    # (N, 3)
    index_tip = kp[:, INDEX_TIP_IDX, :]    # (N, 3)
    middle_tip = kp[:, MIDDLE_TIP_IDX, :]  # (N, 3)

    # Virtual finger.
    k_vf = 0.7 * index_tip + 0.3 * middle_tip  # (N, 3)

    # Gripper width is the Euclidean distance from thumb tip to virtual finger.
    gripper_width = np.linalg.norm(thumb_tip - k_vf, axis=1)  # (N,)

    return gripper_width


# ============================================================
# 2. State Construction
# ============================================================

def construct_state(
    left_ee_pose: np.ndarray,
    right_ee_pose: np.ndarray,
    left_keypoints: np.ndarray,
    right_keypoints: np.ndarray,
) -> np.ndarray:
    """
    Construct the state vector.

    state[t] = [left_ee_pose(7), right_ee_pose(7), left_gripper(1), right_gripper(1)]
             = (16,)

    Args:
        left_ee_pose: (N, 7) — [tx, ty, tz, qw, qx, qy, qz]
        right_ee_pose: (N, 7)
        left_keypoints: (N, 63)
        right_keypoints: (N, 63)

    Returns:
        state: (N, 16)
    """
    left_gripper = compute_gripper_width(left_keypoints)[:, None]   # (N, 1)
    right_gripper = compute_gripper_width(right_keypoints)[:, None]  # (N, 1)

    state = np.concatenate([
        left_ee_pose,     # (N, 7)
        right_ee_pose,    # (N, 7)
        left_gripper,     # (N, 1)
        right_gripper,    # (N, 1)
    ], axis=1)  # (N, 16)

    return state


# ============================================================
# 3. Action Construction
# ============================================================

def construct_action(state: np.ndarray, skip: int = 5) -> np.ndarray:
    """
    action[t] = state[t + skip] (absolute single-step)
    For the final ``skip`` frames, use ``action = state[-1]`` (hold position).

    Args:
        state: (N, 16)
        skip: Lookahead step count.

    Returns:
        action: (N, 16)
    """
    N = state.shape[0]
    action = np.zeros_like(state)

    # Frames where t + skip < N.
    valid_len = N - skip
    if valid_len > 0:
        action[:valid_len] = state[skip:]
        # Fill the final skip frames with the last frame.
        action[valid_len:] = state[-1]
    else:
        # Episode is shorter than skip; use the last frame throughout.
        action[:] = state[-1]

    return action


# ============================================================
# 4. Optional Smoothing (Savitzky-Golay)
# ============================================================

def smooth_state(state: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to the translational state components.
    Quaternions are left untouched (SLERP would be required; skipped for this
    MVP).

    Args:
        state: (N, 16)
        window: Savitzky-Golay window size (must be odd and >= polyorder + 2).
        polyorder: Polynomial order.

    Returns:
        smoothed state: (N, 16)
    """
    N = state.shape[0]
    if N < window:
        return state  # Not enough frames for smoothing.

    smoothed = state.copy()

    # Smooth translation components (left hand 0:3, right hand 7:10).
    for col_range in [(0, 3), (7, 10)]:
        start, end = col_range
        for d in range(start, end):
            smoothed[:, d] = savgol_filter(state[:, d], window, polyorder)

    # Gripper widths can also be smoothed to reduce jitter.
    for col in [14, 15]:  # left_gripper, right_gripper
        smoothed[:, col] = savgol_filter(state[:, col], window, polyorder)
        # Ensure gripper widths remain non-negative.
        smoothed[:, col] = np.maximum(smoothed[:, col], 0.0)

    return smoothed


# ============================================================
# 5. Main Pipeline
# ============================================================

def process_episode(
    npz_path: str,
    skip: int = 5,
    smooth: bool = True,
    smooth_window: int = 11,
) -> Optional[dict]:
    """
    Process one episode through the Step 2 Path A pipeline.

    Returns:
        dict with state, action, metadata; or None if episode too short
    """
    data = np.load(npz_path, allow_pickle=True)

    left_ee = data["left_obs_ee_pose"]       # (N, 7)
    right_ee = data["right_obs_ee_pose"]     # (N, 7)
    left_kp = data["left_obs_keypoints"]     # (N, 63)
    right_kp = data["right_obs_keypoints"]   # (N, 63)
    total_frames = int(data["total_frames"])
    timestamps_ns = data["timestamps_ns"]

    print(f"    Frames: {total_frames}")

    # Construct state.
    state = construct_state(left_ee, right_ee, left_kp, right_kp)
    print(f"    State shape: {state.shape}")  # (N, 16)

    # Optional smoothing.
    if smooth and total_frames >= smooth_window:
        state = smooth_state(state, window=smooth_window)
        print(f"    Smoothing applied (savgol, window={smooth_window})")

    # Construct action.
    action = construct_action(state, skip=skip)
    print(f"    Action shape: {action.shape}, skip={skip}")

    # Sanity checks
    left_gripper = state[:, 14]
    right_gripper = state[:, 15]
    print(f"    Left gripper: min={left_gripper.min():.4f} max={left_gripper.max():.4f} "
          f"mean={left_gripper.mean():.4f}")
    print(f"    Right gripper: min={right_gripper.min():.4f} max={right_gripper.max():.4f} "
          f"mean={right_gripper.mean():.4f}")

    # Check ee_pose quaternion normalization (inherited from Step 1 and expected
    # to contain unit quaternions).
    left_quat_norms = np.linalg.norm(state[:, 3:7], axis=1)
    right_quat_norms = np.linalg.norm(state[:, 10:14], axis=1)
    print(f"    Quat norms — left: [{left_quat_norms.min():.6f}, {left_quat_norms.max():.6f}], "
          f"right: [{right_quat_norms.min():.6f}, {right_quat_norms.max():.6f}]")

    result = {
        "state": state,                     # (N, 16)
        "action": action,                   # (N, 16)
        "timestamps_ns": timestamps_ns,     # (N,)
        "total_frames": total_frames,
        "skip": skip,
        "smoothed": smooth and total_frames >= smooth_window,
        # Preserve metadata for Step 7 (LeRobot dataset writing).
        "episode_path": str(data.get("episode_path", npz_path)),
        "annotations": str(data.get("annotations", "[]")),
    }

    return result


def build_arg_parser():
    """Build the argument parser for the align subcommand."""
    parser = argparse.ArgumentParser(description="Step 2 Path A: Simplified Action Alignment")
    parser.add_argument("--input_dir", required=True, help="Directory with Step 1 .npz outputs")
    parser.add_argument("--output_dir", required=True, help="Output directory for state/action .npz")
    parser.add_argument("--skip", type=int, default=5, help="Action lookahead skip (default: 5)")
    parser.add_argument("--no-smooth", action="store_true", help="Disable Savitzky-Golay smoothing")
    parser.add_argument("--smooth-window", type=int, default=11, help="Savgol window size (default: 11)")
    return parser


def run(args):
    """Process each episode .npz under --input_dir and save state/action data."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all Step 1 outputs.
    input_path = Path(args.input_dir)
    npz_files = sorted(input_path.glob("*.npz"))
    print(f"Found {len(npz_files)} episode files in {args.input_dir}")

    if not npz_files:
        print("ERROR: No .npz files found. Run Step 1 first.")
        return

    results = []
    for i, npz_file in enumerate(npz_files):
        ep_name = npz_file.stem
        print(f"\n[{i+1}/{len(npz_files)}] Episode: {ep_name}")

        result = process_episode(
            str(npz_file),
            skip=args.skip,
            smooth=not args.no_smooth,
            smooth_window=args.smooth_window,
        )
        if result is None:
            continue

        # Save output.
        output_file = os.path.join(args.output_dir, f"{ep_name}.npz")
        np.savez_compressed(
            output_file,
            state=result["state"],
            action=result["action"],
            timestamps_ns=result["timestamps_ns"],
            total_frames=np.array(result["total_frames"]),
            skip=np.array(result["skip"]),
            smoothed=np.array(result["smoothed"]),
            episode_path=result["episode_path"],
            annotations=result["annotations"],
        )
        results.append({
            "name": ep_name,
            "frames": result["total_frames"],
            "file": output_file,
        })
        print(f"    Saved → {output_file}")

    # Summary.
    print(f"\n{'='*60}")
    print(f"Step 2 Path A Complete: {len(results)}/{len(npz_files)} episodes processed")
    print("State/Action shape: (N, 16)")
    print(f"Action construction: state[t + {args.skip}]")
    print(f"Smoothing: {'OFF' if args.no_smooth else f'ON (savgol window={args.smooth_window})'}")
    for r in results:
        print(f"  {r['name']}: {r['frames']} frames")
    print(f"Output: {args.output_dir}")


def main():
    """Entry point for the align command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
