# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Step 1: EgoVerse Zarr Episode Loader + Cleaning

Pipeline step 1: load episodes from EgoVerse zarr v3 format and clean the data.

Features:
  - Load zarr v3 episodes (obs_ee_pose / obs_keypoints / obs_head_pose / obs_wrist_pose / images / annotations)
  - Truncate padding (keep only total_frames frames)
  - Filter 1e9 sentinel frames (mark invalid)
  - Head-relative coordinate transform (use obs_head_pose as pivot for cross-episode comparability)
  - Normalize FPS to 30fps (resample 28/29fps samples)
  - Decode JPEG frames

Usage:
  python cli.py load --input_dir /path/to/egoverse/bimanual_sample --output_dir ./load_output

Output:
  Each episode is saved as an .npz file containing all cleaned fields.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import simplejpeg
import zarr


# ============================================================
# 1. Zarr v3 episode loading
# ============================================================

def load_episode(episode_path: str) -> dict:
    """Load all fields from one EgoVerse zarr v3 episode."""
    store = zarr.open_group(episode_path, mode="r")
    attrs = store.attrs.asdict() if hasattr(store.attrs, 'asdict') else dict(store.attrs)

    total_frames = attrs.get("total_frames", None)
    if total_frames is None:
        # Fallback: use the obs_head_pose shape.
        total_frames = store["obs_head_pose"].shape[0]

    data = {
        "left_obs_ee_pose": np.array(store["left.obs_ee_pose"][:total_frames]),
        "right_obs_ee_pose": np.array(store["right.obs_ee_pose"][:total_frames]),
        "left_obs_wrist_pose": np.array(store["left.obs_wrist_pose"][:total_frames]),
        "right_obs_wrist_pose": np.array(store["right.obs_wrist_pose"][:total_frames]),
        "left_obs_keypoints": np.array(store["left.obs_keypoints"][:total_frames]),
        "right_obs_keypoints": np.array(store["right.obs_keypoints"][:total_frames]),
        "obs_head_pose": np.array(store["obs_head_pose"][:total_frames]),
    }

    # Some hash-named episodes lack obs_rgb_timestamps_ns; synthesize timestamps from FPS.
    if "obs_rgb_timestamps_ns" in store:
        data["timestamps_ns"] = np.array(store["obs_rgb_timestamps_ns"][:total_frames])
    else:
        fps = attrs.get("fps", 30)
        interval_ns = int(1e9 / fps)
        data["timestamps_ns"] = (np.arange(total_frames) * interval_ns).astype(np.int64)
        data["_synthesized_timestamps"] = True

    # Load annotations (JSON strings).
    annotations_raw = store["annotations"][:]
    annotations = []
    for ann in annotations_raw:
        try:
            ann_str = bytes(ann).decode("utf-8") if not isinstance(ann, str) else ann
            annotations.append(json.loads(ann_str))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    data["annotations"] = annotations

    # Load attributes.
    data["attributes"] = attrs
    data["total_frames"] = total_frames

    # Do not decode all images here (too memory-intensive); keep only the raw blob path.
    data["_episode_path"] = episode_path
    data["_has_images"] = "images.front_1" in store

    return data


def decode_frame_jpeg(episode_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Decode one JPEG frame on demand."""
    store = zarr.open_group(episode_path, mode="r")
    try:
        blob = store["images.front_1"][frame_idx]
        return simplejpeg.decode_jpeg(bytes(blob), colorspace="RGB")
    except Exception:
        return None


# ============================================================
# 2. Filter 1e9 sentinel frames
# ============================================================

def find_invalid_frames(data: dict, sentinel: float = 1e9) -> np.ndarray:
    """
    Detect frames containing 1e9 sentinel values. A value greater than or equal
    to the sentinel in any pose/keypoint field marks the frame invalid.

    Returns:
        valid_mask: bool array, shape (total_frames,), True = valid
    """
    n = data["total_frames"]
    valid = np.ones(n, dtype=bool)

    pose_keys = [
        "left_obs_ee_pose", "right_obs_ee_pose",
        "left_obs_wrist_pose", "right_obs_wrist_pose",
        "obs_head_pose",
        "left_obs_keypoints", "right_obs_keypoints",
    ]

    for key in pose_keys:
        arr = data[key]
        # A value above the sentinel in any dimension invalidates the frame.
        frame_max = np.abs(arr).max(axis=1) if arr.ndim == 2 else np.abs(arr)
        valid &= (frame_max < sentinel)

    return valid


# ============================================================
# 3. Head-relative coordinate transform
# ============================================================

def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """
    EgoVerse pose (7,) = [tx, ty, tz, qw, qx, qy, qz] -> a 4x4 homogeneous transform.
    EgoVerse stores the first three values as translation and the last four as a w-first quaternion.
    """
    t = pose7[:3]
    q_wxyz = pose7[3:]  # (qw, qx, qy, qz)
    R = quat_wxyz_to_matrix(q_wxyz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous matrix to pose7 [tx, ty, tz, qw, qx, qy, qz]."""
    t = T[:3, 3]
    R = T[:3, :3]
    # Rotation matrix → quaternion (w, x, y, z)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)  # Normalize.
    return np.concatenate([t, q])


def transform_pose_to_head_relative(pose7_seq: np.ndarray, head_pose7_seq: np.ndarray) -> np.ndarray:
    """
    Transform a pose sequence from SLAM world coordinates to head-relative coordinates.

    T_head_relative = T_head^{-1} @ T_world

    Args:
        pose7_seq: (N, 7) poses in world coordinates
        head_pose7_seq: (N, 7) corresponding head poses

    Returns:
        (N, 7) head-relative pose
    """
    N = pose7_seq.shape[0]
    result = np.zeros_like(pose7_seq)
    for i in range(N):
        T_head = pose7_to_matrix(head_pose7_seq[i])
        T_world = pose7_to_matrix(pose7_seq[i])
        T_head_inv = np.linalg.inv(T_head)
        T_rel = T_head_inv @ T_world
        result[i] = matrix_to_pose7(T_rel)
    return result


def transform_keypoints_to_head_relative(kp_seq: np.ndarray, head_pose7_seq: np.ndarray) -> np.ndarray:
    """
    Transform keypoints (N, 63) from world coordinates to head-relative coordinates.
    63 = 21 keypoints × 3 (x, y, z)。
    """
    N = kp_seq.shape[0]
    result = np.zeros_like(kp_seq)
    for i in range(N):
        T_head = pose7_to_matrix(head_pose7_seq[i])
        T_head_inv = np.linalg.inv(T_head)
        R_inv = T_head_inv[:3, :3]
        t_inv = T_head_inv[:3, 3]
        kp = kp_seq[i].reshape(21, 3)  # (21, 3)
        kp_rel = (R_inv @ kp.T).T + t_inv  # (21, 3)
        result[i] = kp_rel.flatten()
    return result


def apply_head_relative_transform(data: dict) -> dict:
    """Apply a head-relative transform to all poses and keypoints.
    obs_head_pose should become identity after the transform (the field is retained for downstream consistency)."""
    head = data["obs_head_pose"].copy()  # Back up before in-place updates.

    data["left_obs_ee_pose"] = transform_pose_to_head_relative(data["left_obs_ee_pose"], head)
    data["right_obs_ee_pose"] = transform_pose_to_head_relative(data["right_obs_ee_pose"], head)
    data["left_obs_wrist_pose"] = transform_pose_to_head_relative(data["left_obs_wrist_pose"], head)
    data["right_obs_wrist_pose"] = transform_pose_to_head_relative(data["right_obs_wrist_pose"], head)
    data["left_obs_keypoints"] = transform_keypoints_to_head_relative(data["left_obs_keypoints"], head)
    data["right_obs_keypoints"] = transform_keypoints_to_head_relative(data["right_obs_keypoints"], head)
    # A head pose expressed in its own frame is identity (for downstream consistency).
    data["obs_head_pose"] = transform_pose_to_head_relative(head, head)

    return data


# ============================================================
# 4. FPS resampling (normalize to 30fps)
# ============================================================

def estimate_fps(timestamps_ns: np.ndarray) -> float:
    """Estimate the actual FPS from nanosecond timestamps."""
    if len(timestamps_ns) < 2:
        return 30.0
    diffs = np.diff(timestamps_ns)
    median_interval_ns = np.median(diffs)
    if median_interval_ns <= 0:
        return 30.0
    return 1e9 / median_interval_ns


def resample_to_target_fps(data: dict, target_fps: float = 30.0) -> dict:
    """
    If the actual FPS differs from the target by more than 1 FPS, resample linearly.
    Pose (7D) translation uses linear interpolation; rotation uses a simplified normalized linear interpolation (NLERP).
    Keypoints (63D) use linear interpolation.
    """
    actual_fps = estimate_fps(data["timestamps_ns"])

    # No processing within tolerance.
    if abs(actual_fps - target_fps) < 1.0:
        return data

    n_original = data["total_frames"]
    duration_s = (n_original - 1) / actual_fps
    n_target = int(round(duration_s * target_fps)) + 1

    # Original and target time axes.
    t_orig = np.linspace(0, 1, n_original)
    t_new = np.linspace(0, 1, n_target)

    # Interpolate each numeric field.
    numeric_keys = [
        "left_obs_ee_pose", "right_obs_ee_pose",
        "left_obs_wrist_pose", "right_obs_wrist_pose",
        "obs_head_pose",
        "left_obs_keypoints", "right_obs_keypoints",
    ]

    for key in numeric_keys:
        arr = data[key]  # (n_original, D)
        D = arr.shape[1]
        resampled = np.zeros((n_target, D))
        for d in range(D):
            resampled[:, d] = np.interp(t_new, t_orig, arr[:, d])

        # Normalize quaternion parts of pose7 to preserve unit quaternions.
        if arr.shape[1] == 7:
            quat = resampled[:, 3:]
            norms = np.linalg.norm(quat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            resampled[:, 3:] = quat / norms

        data[key] = resampled

    # Rebuild timestamps.
    data["timestamps_ns"] = np.linspace(
        data["timestamps_ns"][0],
        data["timestamps_ns"][-1],
        n_target
    ).astype(np.int64)

    data["total_frames"] = n_target
    data["_resampled_from_fps"] = actual_fps

    return data


# ============================================================
# 5. Main pipeline
# ============================================================

def process_episode(episode_path: str, target_fps: float = 30.0) -> Optional[dict]:
    """
    Process one episode through the complete Step 1 pipeline.

    Returns:
        Processed data dict, or None if all episode frames are invalid.
    """
    print(f"  Loading: {episode_path}")
    data = load_episode(episode_path)
    n_orig = data["total_frames"]

    # Filter 1e9 sentinel frames.
    valid_mask = find_invalid_frames(data)
    n_valid = valid_mask.sum()
    n_invalid = n_orig - n_valid
    print(f"    Frames: {n_orig} total, {n_invalid} invalid (1e9 sentinel)")

    if n_valid == 0:
        print("    SKIP: all frames invalid")
        return None

    # Drop invalid frames when the ratio is below 50%; otherwise drop the entire episode.
    invalid_ratio = n_invalid / n_orig
    if invalid_ratio >= 0.5:
        print(f"    SKIP: {invalid_ratio:.0%} frames invalid, dropping episode")
        return None

    if n_invalid > 0:
        # Keep only valid frames.
        numeric_keys = [
            "left_obs_ee_pose", "right_obs_ee_pose",
            "left_obs_wrist_pose", "right_obs_wrist_pose",
            "obs_head_pose",
            "left_obs_keypoints", "right_obs_keypoints",
            "timestamps_ns",
        ]
        for key in numeric_keys:
            data[key] = data[key][valid_mask]
        data["total_frames"] = n_valid
        data["_dropped_frames"] = int(n_invalid)
        print(f"    Dropped {n_invalid} invalid frames, {n_valid} remaining")

    # FPS resampling.
    actual_fps = estimate_fps(data["timestamps_ns"])
    data = resample_to_target_fps(data, target_fps)
    if "_resampled_from_fps" in data:
        print(f"    Resampled: {actual_fps:.1f}fps → {target_fps:.1f}fps ({data['total_frames']} frames)")
    else:
        print(f"    FPS OK: {actual_fps:.1f}fps (~{target_fps:.0f}), no resample needed")

    # Head-relative transform.
    data = apply_head_relative_transform(data)
    print("    Head-relative transform applied")

    return data


def discover_episodes(input_dir: str) -> list:
    """Find all zarr v3 episodes (subdirectories containing zarr.json)."""
    episodes = []
    input_path = Path(input_dir)
    for child in sorted(input_path.iterdir()):
        if child.is_dir() and (child / "zarr.json").exists():
            episodes.append(str(child))
    return episodes


def build_arg_parser():
    """Build the argument parser for the load subcommand."""
    parser = argparse.ArgumentParser(description="Step 1: EgoVerse Episode Loader + Cleaning")
    parser.add_argument("--input_dir", required=True, help="Path to bimanual_sample/ directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for cleaned .npz files")
    parser.add_argument("--target_fps", type=float, default=30.0, help="Target FPS (default: 30)")
    return parser


def run(args):
    """Process and save cleaned data for every episode under --input_dir."""
    os.makedirs(args.output_dir, exist_ok=True)

    episodes = discover_episodes(args.input_dir)
    print(f"Discovered {len(episodes)} episodes in {args.input_dir}")

    results = []
    for i, ep_path in enumerate(episodes):
        ep_name = Path(ep_path).name
        print(f"\n[{i+1}/{len(episodes)}] Episode: {ep_name}")

        data = process_episode(ep_path, args.target_fps)
        if data is None:
            continue

        # Save cleaned data (images are decoded on demand).
        output_file = os.path.join(args.output_dir, f"{ep_name}.npz")
        np.savez_compressed(
            output_file,
            left_obs_ee_pose=data["left_obs_ee_pose"],
            right_obs_ee_pose=data["right_obs_ee_pose"],
            left_obs_wrist_pose=data["left_obs_wrist_pose"],
            right_obs_wrist_pose=data["right_obs_wrist_pose"],
            left_obs_keypoints=data["left_obs_keypoints"],
            right_obs_keypoints=data["right_obs_keypoints"],
            obs_head_pose=data["obs_head_pose"],
            timestamps_ns=data["timestamps_ns"],
            total_frames=np.array(data["total_frames"]),
            annotations=json.dumps(data["annotations"]),
            episode_path=str(data["_episode_path"]),
            attributes=json.dumps(data["attributes"]),
        )
        results.append({
            "name": ep_name,
            "frames": data["total_frames"],
            "file": output_file,
        })
        print(f"    Saved → {output_file}")

    # Summary.
    print(f"\n{'='*60}")
    print(f"Step 1 Complete: {len(results)}/{len(episodes)} episodes processed")
    for r in results:
        print(f"  {r['name']}: {r['frames']} frames")
    print(f"Output: {args.output_dir}")


def main():
    """Entry point for the load command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
