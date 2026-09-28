#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Step 7: LeRobot v3.0 Dataset Writer

Reads morphology-aware dual-arm IK results, action-alignment annotations, ego/mask/wrist
videos, and produces a LeRobot v3.0 dataset (data parquet + video + meta).

State: morphology-specific dual-arm joint positions stored in each IK result.
Action: joint-space delta (state[t+1] - state[t]), last frame repeats penultimate action.

Usage:
    python cli.py lerobot --ik_dir data_output/05_retarget --state_dir data_output/02_align \
        --output_dir data_output/06_lerobot
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ─── Config ──────────────────────────────────────────────────────────────────

FPS = float(os.environ.get("EGO2ROBOT_FPS", "30"))
CHUNKS_SIZE = 1000  # episodes per chunk
VIDEO_CODEC = "h264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_CRF = 30
VIDEO_G = 2
VIDEO_PRESET = 12  # SVT-AV1 preset (matching reference)
IMAGE_H, IMAGE_W = 368, 640  # robot_on_bg.mp4 is 640x368 (MuJoCo renderer rounds H up to multiple of 16)

# Legacy Panda layout; new retarget outputs carry their own morphology metadata.
STATE_DIM = 18
STATE_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "left_finger_joint1", "left_finger_joint2",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
    "right_finger_joint1", "right_finger_joint2",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_stats(arr: np.ndarray) -> dict:
    """Compute per-feature statistics matching LeRobot v3.0 schema."""
    # arr shape: (N, D) or scalar-like
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0, ddof=0).tolist(),
        "count": [int(n)],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def compute_image_stats_from_video(video_path: str, sample_n: int = 100) -> dict:
    """Sample frames from a video and compute per-channel image stats.
    Returns stats with shape (3,1,1) as nested lists to match LeRobot format.

    Uses pyav (the lerobot video backend) rather than cv2 — this system's
    OpenCV build lacks a software AV1 decoder and silently fails to read.
    """
    import av as pyav
    container = pyav.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    if total <= 0:
        # frames count sometimes unavailable; decode fully once to count
        total = sum(1 for _ in container.decode(video=0))
        container.close()
        container = pyav.open(video_path)
        stream = container.streams.video[0]
    n = min(sample_n, total) if total > 0 else 0
    idxs = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist()) if n > 0 else set()
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in idxs:
            arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            frames.append(arr)
        if len(frames) >= n:
            break
    container.close()
    if not frames:
        # Fallback zero stats
        zero = [[[0.0]], [[0.0]], [[0.0]]]
        return {
            "min": zero, "max": zero, "mean": zero, "std": zero,
            "count": [0], "q01": zero, "q10": zero, "q50": zero, "q90": zero, "q99": zero,
        }
    stack = np.stack(frames, axis=0)  # (N, H, W, 3)
    # Compute per-channel scalar stats -> nested (3,1,1)
    def per_ch(fn):
        return [[[float(fn(stack[..., c]))]] for c in range(3)]
    def per_ch_q(q):
        return [[[float(np.quantile(stack[..., c], q))]] for c in range(3)]
    return {
        "min": per_ch(np.min),
        "max": per_ch(np.max),
        "mean": per_ch(np.mean),
        "std": per_ch(np.std),
        "count": [int(len(frames))],
        "q01": per_ch_q(0.01),
        "q10": per_ch_q(0.10),
        "q50": per_ch_q(0.50),
        "q90": per_ch_q(0.90),
        "q99": per_ch_q(0.99),
    }


def _ffmpeg_has_svtav1() -> bool:
    """Return whether the local ffmpeg exposes the SVT-AV1 encoder."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
        )
        return "svt_av1" in (out.stdout or "") or "libsvtav1" in (out.stdout or "")
    except Exception:
        return False


def ffmpeg_reencode_av1(src: str, dst: str):
    """Re-encode a video to H.264 yuv420p (default, broadly previewable encoding).

    Raises CalledProcessError via subprocess.run(check=True) on ffmpeg failure.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src,
        "-c:v", "libx264",
        "-pix_fmt", VIDEO_PIX_FMT,
        "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-r", str(FPS),
        "-an",
        dst,
    ]
    subprocess.run(cmd, check=True)


def concat_videos_av1(inputs: list, dst: str):
    """Concatenate multiple AV1 mp4 files into one via ffmpeg concat demuxer."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        listfile = fh.name
        for p in inputs:
            fh.write(f"file '{os.path.abspath(p)}'\n")
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", listfile,
            "-c", "copy", dst,
        ]
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(listfile)


def build_state_action(ik_path: str, n_frames: int) -> tuple:
    """Build (state, action) arrays from a morphology-aware IK result.

    state[t] = qpos[t]
    action[t] = qpos[t+1] - qpos[t] (joint-space delta); action[N-1] = action[N-2]
    """
    d = np.load(ik_path)
    qpos = d["state"] if "state" in d else d["qpos"]
    qpos = qpos.astype(np.float32)
    assert qpos.shape[0] == n_frames, (
        f"qpos {qpos.shape} vs n_frames {n_frames}"
    )
    state = qpos.copy()
    action = np.zeros_like(state)
    if n_frames >= 2:
        action[:-1] = state[1:] - state[:-1]
        action[-1] = action[-2]
    return state, action


def build_episode_rows(ep_index: int, task_index: int, n: int, state: np.ndarray,
                        action: np.ndarray, dataset_offset: int):
    """Build a per-frame dict list for one episode's parquet rows."""
    rows = []
    for i in range(n):
        rows.append({
            "action": action[i].tolist(),
            "observation.state": state[i].tolist(),
            "timestamp": float(i) / FPS,
            "frame_index": int(i),
            "episode_index": int(ep_index),
            "index": int(dataset_offset + i),
            "task_index": int(task_index),
        })
    return rows


def write_data_parquet(all_rows: list, out_path: str, state_dim: int):
    """Write concatenated frames to a single parquet file with fixed_size_list.

    Args:
        all_rows: list of per-frame row dicts (see build_episode_rows).
        out_path: destination parquet file path.
    """
    action_type = pa.list_(pa.float32(), state_dim)
    state_type = pa.list_(pa.float32(), state_dim)
    schema = pa.schema([
        pa.field("action", action_type),
        pa.field("observation.state", state_type),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ])
    arrays = {
        "action": pa.array([r["action"] for r in all_rows], type=action_type),
        "observation.state": pa.array([r["observation.state"] for r in all_rows], type=state_type),
        "timestamp": pa.array([r["timestamp"] for r in all_rows], type=pa.float32()),
        "frame_index": pa.array([r["frame_index"] for r in all_rows], type=pa.int64()),
        "episode_index": pa.array([r["episode_index"] for r in all_rows], type=pa.int64()),
        "index": pa.array([r["index"] for r in all_rows], type=pa.int64()),
        "task_index": pa.array([r["task_index"] for r in all_rows], type=pa.int64()),
    }
    table = pa.table(arrays, schema=schema)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path)


def write_episodes_parquet(ep_metas: list, out_path: str):
    """Write meta/episodes/chunk-000/file-000.parquet.
    Each row has episode-level info + per-episode stats + video pointers.
    """
    rows = []
    for m in ep_metas:
        rows.append(m)
    # Build schema matching reference dataset
    table = pa.table({k: [r[k] for r in rows] for k in rows[0].keys()})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path)


def write_tasks_parquet(task_list: list, out_path: str):
    """Write meta/tasks.parquet.

    LeRobot indexes this frame by the *task text* (pandas index_columns=["task"]),
    with task_index as the only column. Writing task as a plain column instead
    makes meta.tasks.index numeric and `item["task"]` resolve to an int.
    """
    import pandas as pd
    df = pd.DataFrame(
        {"task": [t[1] for t in task_list], "task_index": [t[0] for t in task_list]}
    ).set_index("task")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)


def write_info_json(total_episodes: int, total_frames: int, total_tasks: int,
                    splits: dict, out_path: str, robot_type: str,
                    state_names: list[str], state_dim: int,
                    image_h: int = IMAGE_H, image_w: int = IMAGE_W,
                    video_codec: str = VIDEO_CODEC, has_mask: bool = False,
                    has_wrist: dict[str, bool] | None = None):
    """Write meta/info.json describing dataset schema, sizes, and splits."""
    def video_feature():
        return {
            "dtype": "video",
            "shape": [image_h, image_w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "is_depth_map": False,
                "video.height": image_h,
                "video.width": image_w,
                "video.codec": video_codec,
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
                "video.g": VIDEO_G,
                "video.crf": VIDEO_CRF,
                "video.preset": VIDEO_PRESET,
                "video.fast_decode": 0,
                "video.video_backend": "pyav",
                "video.extra_options": {},
            },
        }

    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "features": {
            "action": {
                "dtype": "float32",
                "names": [f"{n}.pos" for n in state_names],
                "shape": [state_dim],
            },
            "observation.state": {
                "dtype": "float32",
                "names": [f"{n}.pos" for n in state_names],
                "shape": [state_dim],
            },
            "observation.images.ego": video_feature(),
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "robot_type": robot_type,
        "splits": splits,
    }
    has_wrist = has_wrist or {"wrist_l": False, "wrist_r": False}
    if has_mask:
        info["features"]["observation.masks.ego"] = video_feature()
    for wrist_name in ("wrist_l", "wrist_r"):
        if has_wrist.get(wrist_name, False):
            info["features"][f"observation.images.{wrist_name}"] = video_feature()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def aggregate_global_stats(per_ep_arrays: dict) -> dict:
    """Concatenate all episodes' per-feature arrays and compute global stats.

    per_ep_arrays: {feature_name: [arr_ep0, arr_ep1, ...]} for vector features.
    Returns {feature_name: stats_dict}.
    """
    out = {}
    for feat, arr_list in per_ep_arrays.items():
        full = np.concatenate(arr_list, axis=0)
        out[feat] = compute_stats(full)
    return out


def discover_episodes(ik_dir: str, state_dir: str) -> list:
    """Find episodes present in both ik_dir ({ep}_ik.npz) and state_dir ({ep}.npz).

    Sorting keeps episode_index assignments stable.
    """
    ik_eps = {p.name[:-len("_ik.npz")] for p in Path(ik_dir).glob("*_ik.npz")}
    state_eps = {p.stem for p in Path(state_dir).glob("*.npz")}
    eps = sorted(ik_eps & state_eps)
    missing_ik = state_eps - ik_eps
    missing_state = ik_eps - state_eps
    if missing_ik:
        print(
            f"  [WARN] {len(missing_ik)} episode(s) in state_dir have no matching IK result; "
            f"skipped: {sorted(missing_ik)[:3]}..."
        )
    if missing_state:
        print(
            f"  [WARN] {len(missing_state)} episode(s) in ik_dir have no matching state; "
            f"skipped: {sorted(missing_state)[:3]}..."
        )
    return eps


def run(args):
    """Write a LeRobot v3.0 dataset from retargeted IK results + action-alignment state."""
    episodes = args.episodes or discover_episodes(args.ik_dir, args.state_dir)
    if not episodes:
        print("ERROR: no episodes found (need matching {ep}_ik.npz in --ik_dir and {ep}.npz in --state_dir)")
        sys.exit(1)
    print(f"episodes ({len(episodes)}): {episodes}")

    first_ik = np.load(f"{args.ik_dir}/{episodes[0]}_ik.npz", allow_pickle=True)
    state_dim = int(first_ik["state"].shape[1]) if "state" in first_ik else int(first_ik["qpos"].shape[1])
    robot_type = str(first_ik["robot_type"].item()) if "robot_type" in first_ik else "panda_dual"
    if "state_names" in first_ik:
        state_names = [str(x) for x in first_ik["state_names"].tolist()]
    else:
        state_names = list(STATE_NAMES)
    if len(state_names) != state_dim:
        raise ValueError(f"state_names length {len(state_names)} != state_dim {state_dim}")
    for ep in episodes[1:]:
        d = np.load(f"{args.ik_dir}/{ep}_ik.npz", allow_pickle=True)
        ep_dim = int(d["state"].shape[1]) if "state" in d else int(d["qpos"].shape[1])
        ep_robot = str(d["robot_type"].item()) if "robot_type" in d else "panda_dual"
        if ep_dim != state_dim or ep_robot != robot_type:
            raise ValueError(
                f"mixed morphology outputs in one dataset: {ep} has "
                f"robot_type={ep_robot}, state_dim={ep_dim}; expected "
                f"{robot_type}, {state_dim}"
            )
    print(f"robot_type={robot_type}, state_dim={state_dim}")

    auxiliary_streams = {
        "observation.masks.ego": ("arm_mask", "_arm_mask.mp4"),
        "observation.images.wrist_l": ("wrist_l", "_wrist_l.mp4"),
        "observation.images.wrist_r": ("wrist_r", "_wrist_r.mp4"),
    }
    auxiliary_enabled = {}
    for feature, (_, suffix) in auxiliary_streams.items():
        existing = [
            (Path(args.ik_dir) / f"{episode}{suffix}").is_file()
            for episode in episodes
        ]
        if any(existing) and not all(existing):
            missing = [episode for episode, present in zip(episodes, existing)
                       if not present]
            raise FileNotFoundError(
                f"incomplete auxiliary video stream {feature}: missing "
                f"{len(missing)}/{len(episodes)} episode(s), e.g. {missing[:3]}"
            )
        auxiliary_enabled[feature] = all(existing)

    out = Path(args.output_dir)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "videos" / "observation.images.ego" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for feature, enabled in auxiliary_enabled.items():
        if enabled:
            (out / "videos" / feature / "chunk-000").mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    # ── Task table (dedupe annotation text) ──
    task_text_to_index = {}
    task_list = []  # (index, text)
    ep_task_text = {}

    for ep in episodes:
        d = np.load(f"{args.state_dir}/{ep}.npz", allow_pickle=True)
        annots = json.loads(str(d["annotations"]))
        # Path B annotations use ``task``/``frame`` metadata rather than the
        # EgoVerse ``text`` field. Accept both schemas for task-table labels.
        text = "manipulation"
        if annots and isinstance(annots[0], dict):
            value = annots[0].get("text", annots[0].get("task", text))
            if value is not None and str(value).strip():
                text = str(value)
        ep_task_text[ep] = text
        if text not in task_text_to_index:
            idx = len(task_list)
            task_text_to_index[text] = idx
            task_list.append((idx, text))

    # ── Per-episode processing ──
    all_rows = []
    ep_metas = []
    per_ep_state = []
    per_ep_action = []
    reencoded_paths = []
    reencoded_auxiliary_paths = {
        feature: [] for feature, enabled in auxiliary_enabled.items() if enabled
    }
    ep_lengths = []
    ep_image_stats = []

    dataset_offset = 0
    cumulative_time = 0.0

    for ep_index, ep in enumerate(episodes):
        ik_path = f"{args.ik_dir}/{ep}_ik.npz"
        d2a = np.load(f"{args.state_dir}/{ep}.npz", allow_pickle=True)
        n = int(d2a["state"].shape[0])

        state, action = build_state_action(ik_path, n)
        with np.load(ik_path, allow_pickle=False) as ik_data:
            quality_keys = ("ik_ok", "ik_err_pos", "ik_err_rot")
            missing_quality = [key for key in quality_keys if key not in ik_data]
            if missing_quality:
                raise KeyError(f"{ik_path} missing IK quality fields: {missing_quality}")
            ik_ok = np.asarray(ik_data["ik_ok"])
            ik_err_pos = np.asarray(ik_data["ik_err_pos"])
            ik_err_rot = np.asarray(ik_data["ik_err_rot"])
        for key, values in (("ik_ok", ik_ok),
                            ("ik_err_pos", ik_err_pos),
                            ("ik_err_rot", ik_err_rot)):
            if values.ndim == 0 or values.shape[0] != n:
                raise ValueError(
                    f"{ik_path} {key} shape {values.shape} does not match {n} frames")
        per_ep_state.append(state)
        per_ep_action.append(action)
        ep_lengths.append(n)

        rows = build_episode_rows(
            ep_index, task_text_to_index[ep_task_text[ep]], n,
            state, action, dataset_offset,
        )
        all_rows.extend(rows)

        # Re-encode robot_on_bg.mp4 -> H.264
        src_mp4 = f"{args.bg_video_dir}/{ep}_robot_on_bg.mp4"
        av1_mp4 = str(tmp / f"{ep}_av1.mp4")
        ffmpeg_reencode_av1(src_mp4, av1_mp4)
        reencoded_paths.append(av1_mp4)

        for feature, paths in reencoded_auxiliary_paths.items():
            label, suffix = auxiliary_streams[feature]
            src = str(Path(args.ik_dir) / f"{ep}{suffix}")
            encoded = str(tmp / f"{ep}_{label}_av1.mp4")
            ffmpeg_reencode_av1(src, encoded)
            paths.append(encoded)

        # Per-episode image stats (sample from the re-encoded video)
        img_stats = compute_image_stats_from_video(av1_mp4, sample_n=100)
        ep_image_stats.append(img_stats)

        # Per-episode state/action stats
        st_stats = compute_stats(state)
        ac_stats = compute_stats(action)

        to_ts = float(n) / FPS

        ep_meta = {
            "episode_index": ep_index,
            "tasks": [ep_task_text[ep]],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": dataset_offset,
            "dataset_to_index": dataset_offset + n,
            "videos/observation.images.ego/chunk_index": 0,
            "videos/observation.images.ego/file_index": 0,
            "videos/observation.images.ego/from_timestamp": cumulative_time,
            "videos/observation.images.ego/to_timestamp": cumulative_time + to_ts,
            "ik/ok_ratio": float(ik_ok.mean()),
            "ik/pos_err_mean_mm": float(ik_err_pos.mean() * 1000.0),
            "ik/rot_err_mean_deg": float(np.degrees(ik_err_rot.mean())),
        }
        for feature in reencoded_auxiliary_paths:
            video_prefix = f"videos/{feature}"
            ep_meta[f"{video_prefix}/chunk_index"] = 0
            ep_meta[f"{video_prefix}/file_index"] = 0
            ep_meta[f"{video_prefix}/from_timestamp"] = cumulative_time
            ep_meta[f"{video_prefix}/to_timestamp"] = cumulative_time + to_ts
        # Flatten per-episode stats into stats/<feat>/<stat> columns
        def add_stats(prefix, sd):
            for sk, sv in sd.items():
                ep_meta[f"stats/{prefix}/{sk}"] = sv
        add_stats("action", ac_stats)
        add_stats("observation.state", st_stats)
        add_stats("observation.images.ego", img_stats)
        # scalar features
        ts_arr = np.array([float(i) / FPS for i in range(n)], dtype=np.float64)
        fi_arr = np.arange(n, dtype=np.float64)
        idx_arr = np.arange(dataset_offset, dataset_offset + n, dtype=np.float64)
        ei_arr = np.full(n, ep_index, dtype=np.float64)
        ti_arr = np.full(n, task_text_to_index[ep_task_text[ep]], dtype=np.float64)
        add_stats("timestamp", compute_stats(ts_arr))
        add_stats("frame_index", compute_stats(fi_arr))
        add_stats("episode_index", compute_stats(ei_arr))
        add_stats("index", compute_stats(idx_arr))
        add_stats("task_index", compute_stats(ti_arr))
        ep_meta["meta/episodes/chunk_index"] = 0
        ep_meta["meta/episodes/file_index"] = 0

        ep_metas.append(ep_meta)

        dataset_offset += n
        cumulative_time += to_ts
        print(f"[{ep_index}] {ep}: N={n} task='{ep_task_text[ep]}' H.264 ok")

    total_frames = dataset_offset

    # ── Write data parquet ──
    write_data_parquet(all_rows, str(out / "data" / "chunk-000" / "file-000.parquet"), state_dim)
    print(f"data parquet: {total_frames} frames")

    # ── Concat H.264 videos ──
    concat_videos_av1(reencoded_paths, str(out / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"))
    for feature, paths in reencoded_auxiliary_paths.items():
        concat_videos_av1(
            paths, str(out / "videos" / feature / "chunk-000" / "file-000.mp4"))
    print("concatenated H.264 video written")

    # ── Write episodes parquet ──
    write_episodes_parquet(ep_metas, str(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

    # ── Write tasks parquet ──
    write_tasks_parquet(task_list, str(out / "meta" / "tasks.parquet"))

    # ── Global stats.json ──
    global_stats = aggregate_global_stats({
        "action": per_ep_action,
        "observation.state": per_ep_state,
    })
    # scalar global stats
    ts_all, fi_all, idx_all, ei_all, ti_all = [], [], [], [], []
    off = 0
    for ep_index, n in enumerate(ep_lengths):
        ts_all.append(np.array([float(i) / FPS for i in range(n)]))
        fi_all.append(np.arange(n, dtype=np.float64))
        idx_all.append(np.arange(off, off + n, dtype=np.float64))
        ei_all.append(np.full(n, ep_index, dtype=np.float64))
        ti_all.append(np.full(n, task_text_to_index[ep_task_text[episodes[ep_index]]], dtype=np.float64))
        off += n
    global_stats["timestamp"] = compute_stats(np.concatenate(ts_all))
    global_stats["frame_index"] = compute_stats(np.concatenate(fi_all))
    global_stats["index"] = compute_stats(np.concatenate(idx_all))
    global_stats["episode_index"] = compute_stats(np.concatenate(ei_all))
    global_stats["task_index"] = compute_stats(np.concatenate(ti_all))
    # global image stats: average per-ep image stats weighted by count
    def agg_img_stats(stats_list):
        # mean/std/min/max/quantiles: aggregate via simple weighted mean for mean; min/max exact
        counts = np.array([s["count"][0] for s in stats_list], dtype=np.float64)
        tot = counts.sum()
        def wmean(key):
            acc = np.zeros((3, 1, 1))
            for s, c in zip(stats_list, counts):
                acc += np.array(s[key]) * c
            return (acc / tot).tolist()
        def ext(key, fn):
            vals = np.stack([np.array(s[key]) for s in stats_list], axis=0)
            return fn(vals, axis=0).tolist()
        return {
            "min": ext("min", np.min),
            "max": ext("max", np.max),
            "mean": wmean("mean"),
            "std": wmean("std"),
            "count": [int(tot)],
            "q01": wmean("q01"),
            "q10": wmean("q10"),
            "q50": wmean("q50"),
            "q90": wmean("q90"),
            "q99": wmean("q99"),
        }
    global_stats["observation.images.ego"] = agg_img_stats(ep_image_stats)

    with open(out / "meta" / "stats.json", "w") as f:
        json.dump(global_stats, f, indent=4, ensure_ascii=False)

    # Probe the produced stream instead of assuming MuJoCo's nominal size.
    try:
        import av as _av
        video_path = out / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"
        with _av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            image_h = int(stream.codec_context.height)
            image_w = int(stream.codec_context.width)
            codec_name = stream.codec_context.name or ""
        video_codec = "av1" if "av1" in codec_name else "h264"
    except Exception:
        image_h, image_w, video_codec = IMAGE_H, IMAGE_W, VIDEO_CODEC

    # ── info.json ──
    write_info_json(
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=len(task_list),
        splits={"train": f"0:{len(episodes)}"},
        out_path=str(out / "meta" / "info.json"),
        robot_type=robot_type,
        state_names=state_names,
        state_dim=state_dim,
        image_h=image_h,
        image_w=image_w,
        video_codec=video_codec,
        has_mask=auxiliary_enabled["observation.masks.ego"],
        has_wrist={
            "wrist_l": auxiliary_enabled["observation.images.wrist_l"],
            "wrist_r": auxiliary_enabled["observation.images.wrist_r"],
        },
    )

    print(f"\n✅ LeRobot v3.0 dataset written to {out}")
    print(f"   episodes={len(episodes)} frames={total_frames} tasks={len(task_list)}")


def build_arg_parser():
    """Build the argument parser for the lerobot subcommand."""
    ap = argparse.ArgumentParser(description="Step 7: LeRobot v3.0 Dataset Writer")
    ap.add_argument("--ik_dir", required=True,
                     help="Retarget output directory (contains {ep}_ik.npz and {ep}_robot_on_bg.mp4)")
    ap.add_argument("--state_dir", required=True,
                     help="Step 2 output directory (contains {ep}.npz for annotations and frame-count checks)")
    ap.add_argument("--bg_video_dir", default=None,
                     help="Directory containing {ep}_robot_on_bg.mp4; defaults to --ik_dir")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tmp_dir", default=None,
                     help="Temporary directory for AV1 re-encoding; defaults to <output_dir>/_tmp_av1")
    ap.add_argument("--episodes", nargs="*", default=None,
                     help="Explicit episode list; defaults to the sorted intersection of ik_dir and state_dir")
    return ap


def main():
    """Entry point for the lerobot command."""
    args = build_arg_parser().parse_args()
    if args.bg_video_dir is None:
        args.bg_video_dir = args.ik_dir
    if args.tmp_dir is None:
        args.tmp_dir = str(Path(args.output_dir) / "_tmp_av1")
    run(args)


if __name__ == "__main__":
    main()
