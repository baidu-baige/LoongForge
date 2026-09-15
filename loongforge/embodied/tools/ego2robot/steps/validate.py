# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate the generated LeRobot v3.0 dataset: structure, schema, cross-ref, video-frame-count.

The release version performs general structural validation without asserting the exact
episode count, frame count, or task text from a particular historical run. It applies
to any dataset produced by this pipeline:
  - Complete directory structure
  - Consistent key info.json fields (codebase_version/fps/robot_type/feature dimensions)
  - Correct data parquet rows and columns, monotonic global index, and episode_index coverage of 0..total_episodes-1
  - episodes parquet has total_episodes rows; every episode has length > 0,
    consistent from/to indices, and covers the full data range
  - tasks parquet has total_tasks rows and every task has non-empty text
  - Each stats.json feature dimension matches the shape declared in info.json
  - Video frame count and dimensions match info.json

Usage:
    python cli.py validate --dataset_dir data_output/06_lerobot
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def check(cond, msg, warn_only=False):
    """Check cond and print the result; exit on failure unless warn_only is set."""
    if not cond:
        tag = "⚠️ " if warn_only else "❌"
        print(f"{tag} {msg}")
        if not warn_only:
            sys.exit(1)
    else:
        print(f"  ✓ {msg}")


def run(args):
    """Run all nine structural checks for a LeRobot v3.0 dataset."""
    root = args.dataset_dir

    # --- 1. Directory structure ---
    print("═══ 1. Directory structure ═══")
    check(os.path.isdir(root), "root exists")
    check(os.path.isfile(f"{root}/meta/info.json"), "meta/info.json")
    check(os.path.isfile(f"{root}/meta/stats.json"), "meta/stats.json")
    check(os.path.isfile(f"{root}/meta/tasks.parquet"), "meta/tasks.parquet")
    check(os.path.isfile(f"{root}/meta/episodes/chunk-000/file-000.parquet"), "episodes parquet")
    check(os.path.isfile(f"{root}/data/chunk-000/file-000.parquet"), "data parquet")
    video_path = f"{root}/videos/observation.images.ego/chunk-000/file-000.mp4"
    check(os.path.isfile(video_path), "video mp4")

    # --- 2. info.json ---
    print("\n═══ 2. info.json ═══")
    info = json.load(open(f"{root}/meta/info.json"))
    check(info["codebase_version"] == "v3.0", "codebase_version v3.0")
    check(info["fps"] > 0, f"fps={info['fps']} > 0")
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]
    total_tasks = info["total_tasks"]
    check(total_episodes > 0, f"total_episodes={total_episodes} > 0")
    check(total_frames > 0, f"total_frames={total_frames} > 0")
    check(total_tasks > 0, f"total_tasks={total_tasks} > 0")
    action_dim = info["features"]["action"]["shape"][0]
    state_dim = info["features"]["observation.state"]["shape"][0]
    check(action_dim == state_dim, f"action/state dims match ({action_dim})")
    check(info["features"]["observation.images.ego"]["dtype"] == "video", "ego video dtype")
    img_info = info["features"]["observation.images.ego"]["info"]
    video_h = img_info["video.height"]
    video_w = img_info["video.width"]
    robot_type = str(info.get("robot_type", ""))
    check(bool(robot_type), f"robot_type={robot_type}")

    # --- 3. data parquet ---
    print("\n═══ 3. data parquet ═══")
    td = pq.read_table(f"{root}/data/chunk-000/file-000.parquet")
    check(td.num_rows == total_frames,
          f"data rows={td.num_rows} matches info.total_frames={total_frames}")
    cols = td.column_names
    check("action" in cols, "action column")
    check("observation.state" in cols, "observation.state column")
    check("timestamp" in cols and "frame_index" in cols and "episode_index" in cols, "index columns")
    r0 = td.slice(0, 1).to_pylist()[0]
    check(len(r0["action"]) == action_dim, f"action dim={action_dim}")
    check(len(r0["observation.state"]) == state_dim, f"state dim={state_dim}")
    check(r0["frame_index"] == 0, "first frame_index=0")
    check(r0["episode_index"] == 0, "first episode_index=0")
    check(r0["index"] == 0, "first index=0")
    indices = td.column("index").to_pylist()
    check(indices == list(range(total_frames)), "global index monotonic 0..N-1")
    ep_arr = td.column("episode_index").to_pylist()
    eps = sorted(set(ep_arr))
    check(eps == list(range(total_episodes)),
          f"episode_index covers 0..{total_episodes - 1}")

    # --- 4. episodes parquet ---
    print("\n═══ 4. episodes parquet ═══")
    te = pq.read_table(f"{root}/meta/episodes/chunk-000/file-000.parquet")
    check(te.num_rows == total_episodes,
          f"episode rows={te.num_rows} matches total_episodes={total_episodes}")
    ep_rows = te.to_pylist()
    ep_rows.sort(key=lambda r: r["episode_index"])
    check([r["episode_index"] for r in ep_rows] == list(range(total_episodes)),
          "episode_index values are 0..N-1")
    cursor = 0
    for r in ep_rows:
        check(r["length"] > 0, f"ep{r['episode_index']} length={r['length']} > 0")
        check(r["dataset_from_index"] == cursor,
              f"ep{r['episode_index']} from_index={r['dataset_from_index']} matches cursor={cursor}")
        check(r["dataset_to_index"] == cursor + r["length"],
              f"ep{r['episode_index']} to_index consistent with length")
        cursor += r["length"]
    check(cursor == total_frames, f"episode lengths sum to total_frames={total_frames}")
    check("stats/action/mean" in ep_rows[0], "per-ep stats present")

    # --- 5. tasks parquet ---
    print("\n═══ 5. tasks parquet ═══")
    tt = pq.read_table(f"{root}/meta/tasks.parquet")
    check(tt.num_rows == total_tasks,
          f"task rows={tt.num_rows} matches total_tasks={total_tasks}")
    task_texts = tt.to_pandas().index.tolist()
    check(all(isinstance(t, str) and len(t) > 0 for t in task_texts), "all task texts non-empty")

    # --- 6. stats.json ---
    print("\n═══ 6. stats.json ═══")
    st = json.load(open(f"{root}/meta/stats.json"))
    check("action" in st, "action in global stats")
    check("observation.state" in st, "state in global stats")
    check("observation.images.ego" in st, "image in global stats")
    check(len(st["action"]["mean"]) == action_dim, f"action stats dim={action_dim}")
    check(st["action"]["count"] == [total_frames], f"action count={total_frames}")

    # --- 7. Video frame count ---
    print("\n═══ 7. Video validation ═══")
    import av
    c = av.open(video_path)
    s = c.streams.video[0]
    check(s.frames == total_frames, f"video frame_count={s.frames} matches total_frames={total_frames}")
    check(s.codec_context.width == video_w, f"video width={video_w}")
    check(s.codec_context.height == video_h, f"video height={video_h}")
    c.close()

    # --- 8. Optional exact checks for comparison with a specific historical run. ---
    if args.expect_episodes is not None:
        check(total_episodes == args.expect_episodes,
              f"total_episodes={total_episodes} == expect {args.expect_episodes}")
    if args.expect_frames is not None:
        check(total_frames == args.expect_frames,
              f"total_frames={total_frames} == expect {args.expect_frames}")

    # --- 9. Real-loader smoke test ---
    if not args.skip_loader_check:
        print("\n═══ 9. LeRobotDataset loader smoke test ═══")
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(repo_id=None, root=root)
            check(len(ds) == total_frames, f"loader len(ds)={len(ds)} matches total_frames={total_frames}")
            _ = ds[0]
            check(True, "loader can index ds[0]")
        except ImportError:
            check(True, "lerobot package not installed, skipped loader smoke test", warn_only=True)

    if getattr(args, "depth_dir", None):
        check_depth_consistency(args)
    print("\n══════════════════════════════")
    print("✅ All validation checks passed!")


def build_arg_parser():
    """Build the command-line argument parser."""
    ap = argparse.ArgumentParser(description="Validate a LeRobot v3.0 dataset (structural checks)")
    ap.add_argument("--dataset_dir", required=True, help="LeRobot v3.0 dataset root directory")
    ap.add_argument("--expect_episodes", type=int, default=None,
                     help="Optionally check total_episodes exactly against a specific historical run")
    ap.add_argument("--expect_frames", type=int, default=None,
                     help="Optionally check total_frames exactly against a specific historical run")
    ap.add_argument("--skip_loader_check", action="store_true",
                     help="Skip the real LeRobotDataset loader smoke test (useful when lerobot is not installed)")
    ap.add_argument(
        "--depth_dir",
        default=None,
        help="Depth-step output containing {ep}/scene_depth.npz; enables numeric depth consistency checks",
    )
    ap.add_argument(
        "--depth_ik_dir",
        default=None,
        help="Retarget IK directory containing {ep}_ik.npz, used for robot and DA3 depth comparison",
    )
    ap.add_argument("--depth_samples", type=int, default=12,
                    help="Number of frames sampled for depth consistency checks (default: 12)")
    return ap


def check_depth_consistency(args):
    """Optional section 8: compare sampled DA3 metric depth with rendered robot view depth."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import mujoco
    from steps.robot_retarget import build_dual_model
    from steps.robot_registry import get_robot_spec
    import cv2
    top = args.depth_dir
    eps = sorted(d for d in os.listdir(top) if os.path.isdir(os.path.join(top, d)))
    print(f"{'episode':<30}{'samples':>8}{'z_med':>9}{'d_med':>9}{'ratio':>9}{'CV':>8}")
    for ep in eps:
        depth_npz = os.path.join(top, ep, "scene_depth.npz")
        ik_path = os.path.join(args.depth_ik_dir or "", f"{ep}_ik.npz")
        if not (os.path.exists(depth_npz) and os.path.exists(ik_path)):
            print(f"  [SKIP] {ep}: missing scene_depth.npz or {ep}_ik.npz")
            continue
        d3 = np.load(depth_npz)["depth"].astype(np.float32)
        ik = np.load(ik_path)
        N = ik["qpos"].shape[0]
        robot_type = str(ik["robot_type"].item()) if "robot_type" in ik else "panda"
        spec = get_robot_spec(robot_type)
        left_quat = np.asarray(ik["left_base_quat"] if "left_base_quat" in ik else ik["base_quat"])
        right_quat = np.asarray(ik["right_base_quat"] if "right_base_quat" in ik else ik["base_quat"])
        model = build_dual_model(np.asarray(ik["left_base_pos"]), left_quat,
                                 np.asarray(ik["right_base_pos"]), right_quat, 60.0, spec)
        data = mujoco.MjData(model)
        rend = mujoco.Renderer(model, height=368, width=640)
        rend.enable_depth_rendering()
        idx = np.linspace(0, N - 1, min(args.depth_samples, N)).astype(int)
        ratios, zmeds, dmeds = [], [], []
        for t in idx:
            data.qpos[:] = np.asarray(ik["qpos"][t])
            mujoco.mj_forward(model, data)
            rend.update_scene(data, camera="ego")
            z = rend.render().astype(np.float32)
            mask = (z > 1e-3) & np.isfinite(z)
            if not mask.any():
                continue
            dm = d3[min(t, len(d3) - 1)]
            dm = cv2.resize(dm, (640, 368), interpolation=cv2.INTER_LINEAR)
            zm = float(np.median(z[mask]))
            dm2 = float(np.median(dm[mask]))
            if dm2 > 1e-4 and zm > 1e-4:
                ratios.append(zm / dm2)
                zmeds.append(zm)
                dmeds.append(dm2)
        rend.close()
        if not ratios:
            print(f"{ep:<30}{0:>8}  (no valid depth samples)")
            continue
        a = np.array(ratios)
        cv = float(a.std() / a.mean()) if a.mean() > 1e-9 else float("nan")
        print(f"{ep:<30}{len(ratios):>8}{np.median(zmeds):>9.3f}{np.median(dmeds):>9.3f}{a.mean():>9.3f}{cv:>8.2f}")
    print("  (z=robot view depth, d=DA3 metric depth; ratio near 1 means agreement, low CV means stability)")



def main():
    """Entry point for the validate command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
