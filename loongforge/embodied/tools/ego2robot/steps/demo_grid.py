# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end split-screen demo: generate one synchronized video per episode.

Layout: a 3x3 grid (960x552) with 320x184 cells.
  ┌─────────────┬────────────────────┬──────────────────────┐
  │ 1. Original │ 2. MANO Keypoints  │ 3. Gripper Width     │
  ├─────────────┼────────────────────┼──────────────────────┤
  │ 4. SAM Mask │ 5. Inpaint BG      │ 6. Robot Only (sim)  │
  ├─────────────┼────────────────────┼──────────────────────┤
  │ 7. Robot+BG │ 8. Episode Info    │ 9. (legend/blank)    │
  └─────────────┴────────────────────┴──────────────────────┘

Each cell has a small label in its top-left corner. All views are synchronized
frame by frame, and the output duration matches the original video.

This combines tmp_demo_video.py (the narrow_tabletop dataset) and
tmp_demo_video_bimanual.py (the bimanual_sample dataset). Their logic was identical
apart from directory constants and EPISODES lists, which are now CLI arguments.
When --episodes is omitted, discover every subdirectory under --zarr_dir that
contains zarr.json, matching loader.discover_episodes.

Usage:
    python cli.py demo --zarr_dir ... --mask_dir ... --inpaint_dir ... \\
        --ik_dir ... --output_dir ... [--episodes EP1 EP2 ...]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse

# Match retargeting by pinning the rendering GPU; switching GPUs can cause depth jitter and gaps.
from pathlib import Path

import cv2
import numpy as np

# Layout constants, not paths; keep these fixed.
CELL_W, CELL_H = 320, 184
GRID_COLS, GRID_ROWS = 3, 3
OUT_W = CELL_W * GRID_COLS   # 960
OUT_H = CELL_H * GRID_ROWS   # 552
FPS = float(os.environ.get("EGO2ROBOT_FPS", "30"))


# Common utilities

def resize_cell(frame):
    """Ensure the frame is CELL_W x CELL_H in BGR format."""
    fh, fw = frame.shape[:2]
    if fh == CELL_H and fw == CELL_W:
        return frame
    return cv2.resize(frame, (CELL_W, CELL_H), interpolation=cv2.INTER_LINEAR)


def add_label(frame, text):
    """Overlay a translucent label in the frame's top-left corner."""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.4, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    # Translucent black label background.
    overlay = out[:th + 8, :tw + 12].copy()
    cv2.rectangle(out, (0, 0), (tw + 12, th + 8), (0, 0, 0), -1)
    out[:th + 8, :tw + 12] = cv2.addWeighted(overlay, 0.3, out[:th + 8, :tw + 12], 0.7, 0)
    cv2.putText(out, text, (4, th + 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def read_video_cv2(path, max_frames=None):
    """Read all MP4 frames in BGR format, optionally capped by max_frames."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def make_info_cell(ep, t, n_total, robot_type="panda"):
    """Generate an episode information cell with white text on black."""
    img = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "Episode:",
        f"  {ep[:10]}",
        f"  ...{ep[10:]}",
        f"Frame: {t+1}/{n_total}",
        f"FPS: {FPS}",
        f"Pipeline: EgoVerse->{robot_type}",
    ]
    y = 30
    for line in lines:
        cv2.putText(img, line, (10, y), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 22
    return img


def discover_episodes(zarr_dir: str) -> list:
    """Discover Zarr v3 episodes whose subdirectories contain zarr.json, matching loader."""
    episodes = []
    for child in sorted(Path(zarr_dir).iterdir()):
        if child.is_dir() and (child / "zarr.json").exists():
            episodes.append(child.name)
    return episodes


# Cell generators

def load_original_frames(zarr_dir, ep, n_frames):
    """Read original JPEG frames from Zarr."""
    import zarr
    import simplejpeg
    store = zarr.open(f"{zarr_dir}/{ep}", mode="r")
    imgs = store["images.front_1"]
    frames = []
    for t in range(min(n_frames, int(imgs.shape[0]))):
        v = imgs[t]
        while isinstance(v, np.ndarray) and v.ndim == 0:
            v = v.item()
        if isinstance(v, np.ndarray) and v.dtype == object:
            v = v.flat[0]
            while isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
        rgb = simplejpeg.decode_jpeg(bytes(v), colorspace="RGB")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(resize_cell(bgr))
    return frames


def load_keypoint_frames(zarr_dir, ep, n_frames):
    """Generate frames with keypoint skeleton overlays."""
    import zarr
    import simplejpeg
    from scipy.spatial.transform import Rotation

    BONES = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    COLORS = ([(0, 255, 255)] * 4 + [(0, 255, 0)] * 4 + [(255, 255, 0)] * 4
              + [(255, 0, 255)] * 4 + [(255, 0, 0)] * 4)

    store = zarr.open(f"{zarr_dir}/{ep}", mode="r")
    total = int(store.attrs["total_frames"])
    imgs = store["images.front_1"]
    # Match hand_mask intrinsic parsing, supporting legacy 3x4 and new flat dictionary formats.
    from steps.config import dataset_intrinsics_k, fallback_episode_attrs
    _ep_attrs, _borrow = fallback_episode_attrs(f"{zarr_dir}/{ep}")
    v0 = imgs[0]
    while isinstance(v0, np.ndarray) and v0.ndim == 0:
        v0 = v0.item()
    if isinstance(v0, np.ndarray) and v0.dtype == object:
        v0 = v0.flat[0]
    ih0, iw0 = simplejpeg.decode_jpeg(bytes(v0), colorspace="RGB").shape[:2]
    K = (dataset_intrinsics_k(_ep_attrs, camera="front_1", img_shape=(ih0, iw0))
         if _ep_attrs is not None else None)
    if K is None:
        print("  ⚠️ No intrinsics for front_1, skipping keypoint frames")
        return []
    left_kp = np.array(store["left.obs_keypoints"][:total]).reshape(total, 21, 3)
    right_kp = np.array(store["right.obs_keypoints"][:total]).reshape(total, 21, 3)
    left_kp2d = (np.array(store["left.obs_keypoints_2d"][:total]).reshape(total, 21, 2)
                 if "left.obs_keypoints_2d" in store else None)
    right_kp2d = (np.array(store["right.obs_keypoints_2d"][:total]).reshape(total, 21, 2)
                  if "right.obs_keypoints_2d" in store else None)
    left_conf = (np.array(store["path_b_left_confidence"][:total])
                 if "path_b_left_confidence" in store else np.ones(total))
    right_conf = (np.array(store["path_b_right_confidence"][:total])
                  if "path_b_right_confidence" in store else np.ones(total))
    head_pose = np.array(store["obs_head_pose"][:total])

    frames = []
    for t in range(min(n_frames, total)):
        v = imgs[t]
        while isinstance(v, np.ndarray) and v.ndim == 0:
            v = v.item()
        if isinstance(v, np.ndarray) and v.dtype == object:
            v = v.flat[0]
            while isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
        rgb = simplejpeg.decode_jpeg(bytes(v), colorspace="RGB")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        hp = head_pose[t]
        if np.any(np.abs(hp[:3]) > 1e8):
            frames.append(resize_cell(bgr))
            continue
        q_wxyz = hp[3:7]
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        R_inv = Rotation.from_quat(q_xyzw).as_matrix().T
        t_inv = -R_inv @ hp[:3]

        for kp, kp2d, confidence, color_base in [
                (left_kp[t], left_kp2d[t] if left_kp2d is not None else None,
                 left_conf[t], (0, 255, 0)),
                (right_kp[t], right_kp2d[t] if right_kp2d is not None else None,
                 right_conf[t], (0, 200, 255))]:
            # Zero confidence marks an unobserved hand and must not appear in
            # the demo overlay.
            if confidence <= 0.0 or not np.isfinite(kp).all() or np.any(np.abs(kp) > 1e8):
                continue
            if kp2d is not None and np.isfinite(kp2d).all():
                px = kp2d.astype(np.float32, copy=False)
                valid = np.isfinite(px).all(axis=1)
            else:
                pts_cam = (R_inv @ kp.T).T + t_inv
                z = pts_cam[:, 2]
                valid = z > 0.01
                proj = K @ pts_cam.T
                px = np.zeros((21, 2))
                px[valid, 0] = proj[0, valid] / proj[2, valid]
                px[valid, 1] = proj[1, valid] / proj[2, valid]
            for bi, (a, b) in enumerate(BONES):
                if valid[a] and valid[b]:
                    pa = (int(px[a, 0]), int(px[a, 1]))
                    pb = (int(px[b, 0]), int(px[b, 1]))
                    cv2.line(bgr, pa, pb, COLORS[bi], 2, cv2.LINE_AA)
            for i in range(21):
                if valid[i]:
                    p = (int(px[i, 0]), int(px[i, 1]))
                    cv2.circle(bgr, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
        frames.append(resize_cell(bgr))
    return frames


def load_gripper_frames(zarr_dir, ep, n_frames):
    """Generate frames annotated with gripper width."""
    import zarr
    import simplejpeg
    from scipy.spatial.transform import Rotation

    THUMB, INDEX, MIDDLE = 4, 8, 12
    store = zarr.open(f"{zarr_dir}/{ep}", mode="r")
    total = int(store.attrs["total_frames"])
    imgs = store["images.front_1"]
    # Match hand_mask intrinsic parsing, supporting legacy 3x4 and new flat dictionary formats.
    from steps.config import dataset_intrinsics_k, fallback_episode_attrs
    _ep_attrs, _borrow = fallback_episode_attrs(f"{zarr_dir}/{ep}")
    v0 = imgs[0]
    while isinstance(v0, np.ndarray) and v0.ndim == 0:
        v0 = v0.item()
    if isinstance(v0, np.ndarray) and v0.dtype == object:
        v0 = v0.flat[0]
    ih0, iw0 = simplejpeg.decode_jpeg(bytes(v0), colorspace="RGB").shape[:2]
    K = (dataset_intrinsics_k(_ep_attrs, camera="front_1", img_shape=(ih0, iw0))
         if _ep_attrs is not None else None)
    if K is None:
        print("  ⚠️ No intrinsics for front_1, skipping keypoint frames")
        return []
    left_kp = np.array(store["left.obs_keypoints"][:total]).reshape(total, 21, 3)
    right_kp = np.array(store["right.obs_keypoints"][:total]).reshape(total, 21, 3)
    left_kp2d = (np.array(store["left.obs_keypoints_2d"][:total]).reshape(total, 21, 2)
                 if "left.obs_keypoints_2d" in store else None)
    right_kp2d = (np.array(store["right.obs_keypoints_2d"][:total]).reshape(total, 21, 2)
                  if "right.obs_keypoints_2d" in store else None)
    left_conf = (np.array(store["path_b_left_confidence"][:total])
                 if "path_b_left_confidence" in store else np.ones(total))
    right_conf = (np.array(store["path_b_right_confidence"][:total])
                  if "path_b_right_confidence" in store else np.ones(total))
    head_pose = np.array(store["obs_head_pose"][:total])

    frames = []
    for t in range(min(n_frames, total)):
        v = imgs[t]
        while isinstance(v, np.ndarray) and v.ndim == 0:
            v = v.item()
        if isinstance(v, np.ndarray) and v.dtype == object:
            v = v.flat[0]
            while isinstance(v, np.ndarray) and v.ndim == 0:
                v = v.item()
        rgb = simplejpeg.decode_jpeg(bytes(v), colorspace="RGB")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        hp = head_pose[t]
        if np.any(np.abs(hp[:3]) > 1e8):
            frames.append(resize_cell(bgr))
            continue
        q = hp[3:7]
        R_inv = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T
        t_inv = -R_inv @ hp[:3]

        for kp, kp2d, confidence, label, color in [
                (left_kp[t], left_kp2d[t] if left_kp2d is not None else None,
                 left_conf[t], "L", (0, 255, 0)),
                (right_kp[t], right_kp2d[t] if right_kp2d is not None else None,
                 right_conf[t], "R", (0, 200, 255))]:
            if confidence <= 0.0 or not np.isfinite(kp).all() or np.any(np.abs(kp) > 1e8):
                continue
            if kp2d is not None and np.isfinite(kp2d).all():
                p0 = tuple(np.round(kp2d[THUMB]).astype(int))
                p1 = tuple(np.round(0.7 * kp2d[INDEX] + 0.3 * kp2d[MIDDLE]).astype(int))
                width_cm = float(np.linalg.norm(kp2d[THUMB] -
                                                 (0.7 * kp2d[INDEX] + 0.3 * kp2d[MIDDLE]))) / 100.0
                cv2.line(bgr, p0, p1, (220, 220, 220), 2, cv2.LINE_AA)
                cv2.circle(bgr, p0, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(bgr, p1, 6, color, -1, cv2.LINE_AA)
                mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
                cv2.putText(bgr, f"{label}:{width_cm:.1f}cm",
                            (mid[0] + 4, mid[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                continue
            k_thumb = kp[THUMB]
            k_vf = 0.7 * kp[INDEX] + 0.3 * kp[MIDDLE]
            width_cm = float(np.linalg.norm(k_thumb - k_vf)) * 100
            pts = np.stack([k_thumb, k_vf])
            pc = (R_inv @ pts.T).T + t_inv
            if np.any(pc[:, 2] <= 0.01):
                continue
            proj = K @ pc.T
            px = (proj[:2] / proj[2]).T
            p0 = (int(px[0, 0]), int(px[0, 1]))
            p1 = (int(px[1, 0]), int(px[1, 1]))
            cv2.line(bgr, p0, p1, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.circle(bgr, p0, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(bgr, p1, 6, color, -1, cv2.LINE_AA)
            mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            txt = f"{label}:{width_cm:.1f}cm"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(bgr, (mid[0] - tw // 2 - 2, mid[1] - th - 3),
                          (mid[0] + tw // 2 + 2, mid[1] + 3), (255, 255, 255), -1)
            cv2.putText(bgr, txt, (mid[0] - tw // 2, mid[1] - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        frames.append(resize_cell(bgr))
    return frames


def load_mask_frames(mask_dir, ep, orig_frames):
    """Overlay the SAM3 mask by tinting its area red and drawing its contour."""
    masks = np.load(f"{mask_dir}/{ep}/masks.npz")["masks"]
    frames = []
    n = min(len(masks), len(orig_frames))
    for t in range(n):
        base = orig_frames[t].copy()
        m = masks[t]
        if m.shape[:2] != (base.shape[0], base.shape[1]):
            m = cv2.resize(m, (base.shape[1], base.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        mb = (m > 0)
        red = base.copy()
        red[mb] = (0, 0, 255)
        out = cv2.addWeighted(red, 0.45, base, 0.55, 0)
        cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 255), 1)
        frames.append(out)
    return frames


def load_robot_only_frames(zarr_dir, ik_dir, ep, n_frames):
    """Render the robot only, on a black background."""
    import mujoco
    from steps.robot_retarget import (
        build_dual_model, hide_non_arm_geoms, render_rgb_and_mask,
        set_ego_camera, load_episode_world, make_renderer,
    )
    from steps.robot_registry import get_robot_spec

    ik = np.load(f"{ik_dir}/{ep}_ik.npz")
    qpos_all = ik["qpos"]
    left_base_pos = ik["left_base_pos"]
    right_base_pos = ik["right_base_pos"]
    left_base_quat = ik["left_base_quat"] if "left_base_quat" in ik else ik["base_quat"]
    right_base_quat = ik["right_base_quat"] if "right_base_quat" in ik else ik["base_quat"]
    robot_type = str(ik["robot_type"].item()) if "robot_type" in ik else "panda"
    spec = get_robot_spec(robot_type)
    N = len(qpos_all)

    head, left_kp, right_kp = load_episode_world(zarr_dir, ep, N)
    active_sides = tuple(
        side for side, kp in (("left", left_kp), ("right", right_kp))
        if np.isfinite(kp).all(axis=1).any()
    )

    fy, height = 490.19609999999994, 368
    # For visual alignment, derive fy from dataset intrinsics when possible,
    # supporting both new flat dictionaries and legacy 3x4 matrices.
    try:
        from steps.config import dataset_fy_for_height, fallback_episode_attrs
        _ep_a, _ = fallback_episode_attrs(f"{zarr_dir}/{ep}")
        _fy = dataset_fy_for_height(_ep_a or {}, height)
        if _fy is not None:
            fy = _fy
    except Exception:
        pass
    fovy = float(np.degrees(2.0 * np.arctan((height / 2.0) / fy)))
    dual = build_dual_model(left_base_pos, left_base_quat, right_base_pos, right_base_quat,
                            fovy, spec)
    cam_id = mujoco.mj_name2id(dual, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    hide_non_arm_geoms(dual, spec, active_sides=active_sides)
    renderer = make_renderer(dual, 1280, 736)  # Match retargeting with 2x supersampling for noise reduction.
    data = mujoco.MjData(dual)

    frames = []
    for t in range(min(n_frames, N)):
        data.qpos[:] = qpos_all[t]
        set_ego_camera(data, head[t])
        robot_rgb, mask, _ = render_rgb_and_mask(dual, data, renderer, cam_id)
        robot_rgb = cv2.medianBlur(robot_rgb, 3)  # Match retargeting's speckle-noise reduction.
        robot_bgr = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)
        out = np.zeros_like(robot_bgr)
        mb = mask.astype(bool)
        out[mb] = robot_bgr[mb]
        frames.append(resize_cell(out))
    del renderer
    return frames


# Main workflow

def build_demo(ep, zarr_dir, mask_dir, inpaint_dir, ik_dir, out_dir,
               depth_dir=None, fps=None):
    """Build a 3x3 split-screen demo video for one episode."""
    print(f"\n[{ep}] Building split-screen demo")
    bg_frames_full = read_video_cv2(f"{inpaint_dir}/{ep}/bg.mp4")
    n = len(bg_frames_full)
    print(f"  frame count n={n}")

    print("  1/7 original")
    orig = load_original_frames(zarr_dir, ep, n)
    print("  2/7 keypoints")
    kp = load_keypoint_frames(zarr_dir, ep, n)
    print("  3/7 gripper width")
    grip = load_gripper_frames(zarr_dir, ep, n)
    print("  4/7 SAM mask")
    mask = load_mask_frames(mask_dir, ep, orig)
    print("  5/7 inpaint bg")
    bg = [resize_cell(f) for f in bg_frames_full]
    print("  6/7 robot-only render")
    robot_only = load_robot_only_frames(zarr_dir, ik_dir, ep, n)
    print("  7/7 robot composite")
    robot_bg_full = read_video_cv2(f"{ik_dir}/{ep}_robot_on_bg.mp4")
    robot_bg = [resize_cell(f) for f in robot_bg_full]

    robot_type = "panda"
    try:
        ik_meta = np.load(f"{ik_dir}/{ep}_ik.npz", allow_pickle=True)
        if "robot_type" in ik_meta:
            robot_type = str(ik_meta["robot_type"].item())
    except Exception:
        pass

    depth_frames = None
    if depth_dir:
        print("  8/7 Depth (DA3)")
        depth_frames = load_depth_frames(depth_dir, ep, n)

    # Assemble the nine cells, with info in cell 8 and a blank cell 9.
    cells_static = [
        ("1. Original", orig),
        ("2. MANO Keypoints", kp),
        ("3. Gripper Width", grip),
        ("4. SAM3 Human/Hand Mask", mask),
        ("5. Inpainted BG", bg),
        ("6. Robot Only (sim)", robot_only),
        ("7. Robot Replacing Hand", robot_bg),
    ]

    out_path = str(Path(out_dir) / f"{ep}_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps or FPS), (OUT_W, OUT_H))

    n_out = min(n, min(len(c[1]) for c in cells_static))
    print(f"  compositing {n_out} frames -> {OUT_W}x{OUT_H}")
    for t in range(n_out):
        frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        for i, (label, frames) in enumerate(cells_static):
            r, c = i // GRID_COLS, i % GRID_COLS
            y0, x0 = r * CELL_H, c * CELL_W
            cell = frames[t] if t < len(frames) else np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            cell = add_label(cell, label)
            frame[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell
        # Cell 8 (index 7): depth visualization when --depth_dir is provided, otherwise info.
        r, c = 7 // GRID_COLS, 7 % GRID_COLS
        if depth_frames is not None:
            cell8 = depth_frames[t] if t < len(depth_frames) else np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            cell8 = add_label(cell8, "8. Depth (DA3)")
            frame[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = cell8
        else:
            info = make_info_cell(ep, t, n_out, robot_type)
            info = add_label(info, "8. Episode Info")
            frame[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = info
        # Cell 9 (index 8): info when depth is present, otherwise the pipeline title.
        r, c = 8 // GRID_COLS, 8 % GRID_COLS
        if depth_frames is not None:
            info9 = make_info_cell(ep, t, n_out, robot_type)
            info9 = add_label(info9, "9. Episode Info")
            frame[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = info9
        else:
            title = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            cv2.putText(title, "EgoVerse -> LeRobot", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(title, "H2R Synthesis", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(title, f"{robot_type.upper()} Dual-Arm", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(title, "Qwen-RobotManip", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            frame[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = title
        writer.write(frame)
    writer.release()
    try:
        from .config import reencode_h264
    except ImportError:
        from config import reencode_h264
    reencode_h264(out_path)
    print(f"  ✓ {out_path}")
    return out_path


def build_arg_parser():
    """Build the command-line argument parser."""
    ap = argparse.ArgumentParser(description="Generate 3x3 split-screen demo videos")
    ap.add_argument("--zarr_dir", required=True, help="Original EgoVerse Zarr directory, such as bimanual_sample")
    ap.add_argument("--mask_dir", required=True, help="Step 3 SAM3 mask output directory")
    ap.add_argument("--inpaint_dir", required=True, help="Step 4 inpainting output directory containing {ep}/bg.mp4")
    ap.add_argument(
        "--ik_dir",
        required=True,
        help="Retargeting output directory containing {ep}_ik.npz and {ep}_robot_on_bg.mp4",
    )
    ap.add_argument("--depth_dir", default=None,
                    help="Step 5 depth output directory containing {ep}/depth_vis.mp4; enables the demo depth cell")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--episodes", nargs="*", default=None,
                     help="Episode list; by default discover all --zarr_dir subdirectories containing zarr.json")
    return ap


def load_depth_frames(depth_dir, ep, n_frames):
    """Read Step 5 depth_vis.mp4 into a sequence of cell-sized frames."""
    import cv2
    vid = Path(depth_dir) / ep / "depth_vis.mp4"
    if not vid.exists():
        print(f"  ⚠️ depth_vis.mp4 missing: {vid}")
        return None
    cap = cv2.VideoCapture(str(vid))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n_frames)
    frames = []
    for t in range(min(n_frames, total)):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(resize_cell(f))
    cap.release()
    return frames


def run(args):
    """Build 3x3 split-screen demo videos for all episodes under --zarr_dir."""
    os.makedirs(args.output_dir, exist_ok=True)
    eps = args.episodes if args.episodes else discover_episodes(args.zarr_dir)
    print(f"Found {len(eps)} episodes")
    for ep in eps:
        try:
            build_demo(ep, args.zarr_dir, args.mask_dir, args.inpaint_dir, args.ik_dir, args.output_dir,
                       getattr(args, "depth_dir", None), getattr(args, "fps", None))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ {ep}: {e}")
    print(f"\nDone. → {args.output_dir}")


def main():
    """Entry point for the demo command."""
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
