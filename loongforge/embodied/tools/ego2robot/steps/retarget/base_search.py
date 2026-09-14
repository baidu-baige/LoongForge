# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Base-pose search, support-plane estimation, and collision screening."""

import time

import mujoco
import numpy as np

from .ik import (
    DLS_KEYFRAMES,
    DLS_MAX_ITER,
    DLS_POS_FEAS_GATE,
    DLS_POS_TOL,
    _prewarm,
    _project_qpos_to_limits,
    solve_arm_ik,
    solve_arm_ik_dls,
    solve_arm_ik_dls_robust,
    solve_arm_ik_position_only,
    solve_arm_ik_position_only_dls,
    solve_arm_ik_robust,
)
from .rendering import build_dual_model, mat_to_quat, name_to_dof
from .targets import _quat_to_mat


# ============ base_search（Ego2Robot Eq.4 / Appendix A.4） ============

# Candidate offsets are normalized by each morphology's maximum reach.  The
# paper searches these three translation axes and 3-D base orientation.
BASE_LATERAL = (0.1, 0.2, 0.3, 0.4)
# Keep the order/sign convention from Ego2Robot Appendix A.4.  ``fwd`` is the
# camera-facing direction used to construct the nominal base frame.
BASE_FORWARD = (0.0, -0.1, -0.2, -0.3, -0.4, -0.5)
BASE_VERTICAL = (0.3, 0.2, 0.0, -0.2, -0.3)
BASE_PITCH_DEG = (30.0, 45.0, 60.0)
BASE_YAW_DEG = (-45.0, -20.0, 0.0, 20.0, 45.0)
BASE_ROLL_DEG = (-15.0, 0.0, 15.0)
BASE_MAX_KEYFRAMES = 20
BASE_POSITION_SHORTLIST_FAST = 8
BASE_POSITION_SHORTLIST_BALANCED = 16
BASE_RESULT_TOPK = 5
BASE_POSITION_SCREEN_KEYFRAMES = 6
BASE_POSITION_SCREEN_MAX_ITER = 35
BASE_SLOW_COARSE_KEYFRAMES = 8
BASE_SLOW_COARSE_MAX_ITER = 40
BASE_SLOW_REFINE_TOPK = 64
BASE_FEAS_POS_TOL = 8e-3
BASE_FEAS_ROT_TOL = 0.25
BASE_REACH_TARGET = 0.65
BASE_CAMERA_MIN_DIST = 0.20
BASE_TRAJ_MAX_REACH = 0.90
BASE_TRAJ_MIN_REACH = 0.08
# Every registered morphology is rendered as two attached arm instances, so
# every base pair must pass the same bilateral collision screen.  Twelve
# samples were too sparse for short, fast hand crossings: the selected pair
# could be collision-free at the sampled frames but intersect between them.
BASE_PAIR_COLLISION_MAX_KEYFRAMES = 24
BASE_PAIR_COLLISION_MAX_ITER = 40
BASE_COLLISION_REPAIR_RESTARTS = 16

# Legacy DLS base search grid from the previous panda implementation.  Unlike
# the Ego2Robot grid above, these offsets are absolute metres and the base
# orientation stays at the nominal camera-facing frame.
DLS_HAND_ANCHOR_BACK = (0.0, 0.1, 0.2, 0.3, 0.4)
DLS_HAND_ANCHOR_LAT = (-0.1, 0.0, 0.1, 0.2, 0.3, 0.4)
DLS_HAND_ANCHOR_DZ = (-0.15, 0.0, 0.15, 0.3)

# SO-ARM101 scene-support estimation. The source depth is metric but noisy, so
# a sparse set of frames is fused in the SLAM world frame before fitting one
# static, gravity-aligned mounting plane.
SUPPORT_SURFACE_MAX_FRAMES = 12
SUPPORT_SURFACE_PIXEL_STRIDE = 8
SUPPORT_SURFACE_DEPTH_RANGE = (0.15, 3.0)
SUPPORT_SURFACE_RANSAC_ITERATIONS = 768
SUPPORT_SURFACE_INLIER_TOL = 0.05
SUPPORT_SURFACE_MAX_RMSE = 0.04
SUPPORT_SURFACE_MAX_SLOPE = 0.25
SUPPORT_SURFACE_MAX_LOCAL_GAP = 0.16
SUPPORT_SURFACE_MAX_POINTS = 6000


def estimate_scene_support_surface(scene_depth, head_seq, K, targets_p_world,
                                   reach):
    """Estimate a horizontal workspace support plane from metric ego depth.

    The returned plane is represented as ``z = ax + by + c``. Inlier XY points
    retain the observed support extent for diagnostics. Candidate placement may
    extrapolate the plane because ego video commonly crops out the mounts.
    """
    depth = np.asarray(scene_depth, dtype=np.float32)
    head_seq = np.asarray(head_seq, dtype=float)
    targets = np.asarray(targets_p_world, dtype=float).reshape(-1, 3)
    K = np.asarray(K, dtype=float)
    if depth.ndim != 3 or head_seq.ndim != 2 or head_seq.shape[1] < 7:
        return None
    if K.shape != (3, 3) or not np.isfinite(K).all():
        return None
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        return None
    targets = targets[np.isfinite(targets).all(axis=1)]
    n_frames = min(len(depth), len(head_seq))
    if n_frames == 0 or len(targets) == 0:
        return None

    frame_ids = np.unique(np.linspace(
        0, n_frames - 1, min(n_frames, SUPPORT_SURFACE_MAX_FRAMES),
        dtype=int))
    h, w = depth.shape[1:]
    rows = np.arange(0, h, SUPPORT_SURFACE_PIXEL_STRIDE, dtype=int)
    cols = np.arange(0, w, SUPPORT_SURFACE_PIXEL_STRIDE, dtype=int)
    uu, vv = np.meshgrid(cols.astype(float), rows.astype(float))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    xy_margin = max(0.55, 1.35 * float(reach))
    xy_lo = targets[:, :2].min(axis=0) - xy_margin
    xy_hi = targets[:, :2].max(axis=0) + xy_margin
    camera_down = np.mean([
        _quat_to_mat(pose[3:7])[:, 1] for pose in head_seq[frame_ids]
    ], axis=0)
    down_sign = 1.0 if camera_down[2] >= 0.0 else -1.0
    target_down = down_sign * targets[:, 2]
    # A tabletop-mounted base must lie on the gravity-down side of the hand
    # workspace. A median/upper-hand quantile is robust to brief lifts while
    # excluding horizontal slices that pass above most of the trajectory.
    down_lo = float(np.percentile(target_down, 60.0)) - 0.03
    down_hi = (float(np.percentile(target_down, 90.0)) +
               max(0.50, 1.20 * float(reach)))
    z_lo, z_hi = sorted((down_sign * down_lo, down_sign * down_hi))

    world_chunks = []
    depth_min, depth_max = SUPPORT_SURFACE_DEPTH_RANGE
    for frame_id in frame_ids:
        z_cam = depth[frame_id][::SUPPORT_SURFACE_PIXEL_STRIDE,
                                ::SUPPORT_SURFACE_PIXEL_STRIDE]
        valid = (np.isfinite(z_cam) & (z_cam >= depth_min) &
                 (z_cam <= depth_max))
        if not np.any(valid):
            continue
        z_values = z_cam[valid].astype(float)
        points_cam = np.column_stack([
            (uu[valid] - cx) * z_values / fx,
            (vv[valid] - cy) * z_values / fy,
            z_values,
        ])
        R_head = _quat_to_mat(head_seq[frame_id, 3:7])
        points_world = points_cam @ R_head.T + head_seq[frame_id, :3]
        keep = (
            (points_world[:, 0] >= xy_lo[0]) &
            (points_world[:, 0] <= xy_hi[0]) &
            (points_world[:, 1] >= xy_lo[1]) &
            (points_world[:, 1] <= xy_hi[1]) &
            (points_world[:, 2] >= z_lo) &
            (points_world[:, 2] <= z_hi)
        )
        if np.any(keep):
            world_chunks.append(points_world[keep])
    if not world_chunks:
        return None
    points = np.concatenate(world_chunks, axis=0)
    if len(points) < 200:
        return None

    # Fit gravity-aligned planes directly. Scoring includes the smaller XY
    # spread so a horizontal slice through a wall cannot beat a broad tabletop.
    rng = np.random.default_rng(7)
    min_vertical_component = 1.0 / np.sqrt(
        1.0 + SUPPORT_SURFACE_MAX_SLOPE ** 2)
    target_xy = np.median(targets[:, :2], axis=0)
    best_score = -np.inf
    best_inliers = None
    best_normal = None
    best_plane_d = None
    for _ in range(SUPPORT_SURFACE_RANSAC_ITERATIONS):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-8:
            continue
        normal /= normal_norm
        if abs(normal[2]) < min_vertical_component:
            continue
        plane_d = -float(normal @ sample[0])
        plane_z_at_targets = -(
            normal[0] * target_xy[0] + normal[1] * target_xy[1] + plane_d
        ) / normal[2]
        plane_down = down_sign * plane_z_at_targets
        if plane_down < down_lo or plane_down > down_hi:
            continue
        distances = np.abs(points @ normal + plane_d)
        candidate_inliers = distances <= SUPPORT_SURFACE_INLIER_TOL
        if int(candidate_inliers.sum()) < 200:
            continue
        candidate_xy = points[candidate_inliers, :2]
        centered_xy = candidate_xy - candidate_xy.mean(axis=0)
        xy_eigenvalues = np.linalg.eigvalsh(
            centered_xy.T @ centered_xy / len(centered_xy))
        xy_spread = float(np.sqrt(max(xy_eigenvalues[0], 0.0)))
        score = float(candidate_inliers.sum()) * xy_spread
        if score > best_score:
            best_score = score
            best_inliers = candidate_inliers
            best_normal = normal.copy()
            best_plane_d = plane_d
    if best_inliers is None or best_normal is None or best_plane_d is None:
        return None

    # Copy the narrowed values into concrete numeric locals so static
    # analyzers do not treat the optional accumulators as nullable here.
    normal = np.asarray(best_normal, dtype=float)
    plane_d = float(best_plane_d)
    if normal[2] < 0.0:
        normal = -normal
        plane_d = -plane_d
    plane_d = -float(np.median(points[best_inliers] @ normal))
    orthogonal_residual = np.abs(points @ normal + plane_d)
    inliers = orthogonal_residual <= SUPPORT_SURFACE_INLIER_TOL
    if int(inliers.sum()) < 200:
        return None
    coefficients = np.array([
        -normal[0] / normal[2],
        -normal[1] / normal[2],
        -plane_d / normal[2],
    ])

    support_points = points[inliers]
    residual = support_points[:, 2] - (
        support_points[:, 0] * coefficients[0] +
        support_points[:, 1] * coefficients[1] + coefficients[2])
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    centered_xy = support_points[:, :2] - support_points[:, :2].mean(axis=0)
    xy_eigenvalues = np.linalg.eigvalsh(
        centered_xy.T @ centered_xy / max(len(centered_xy), 1))
    if rmse > SUPPORT_SURFACE_MAX_RMSE or xy_eigenvalues[0] < 0.0025:
        return None

    if len(support_points) > SUPPORT_SURFACE_MAX_POINTS:
        keep_ids = np.linspace(
            0, len(support_points) - 1, SUPPORT_SURFACE_MAX_POINTS,
            dtype=int)
        support_points = support_points[keep_ids]
    confidence = min(1.0, float(inliers.sum()) / 1500.0)
    confidence *= float(np.exp(-rmse / SUPPORT_SURFACE_INLIER_TOL))
    return {
        "coefficients": np.asarray(coefficients, dtype=float),
        "points_xy": np.asarray(support_points[:, :2], dtype=float),
        "rmse": rmse,
        "confidence": confidence,
        "inlier_count": int(inliers.sum()),
        "sample_count": int(len(points)),
        "frame_count": int(len(frame_ids)),
        "down_sign": down_sign,
        "max_local_gap": SUPPORT_SURFACE_MAX_LOCAL_GAP,
        "base_min_z": 0.0,
    }


def _base_visual_min_z(model, base_body):
    """Return the lowest visible base vertex in the model's base frame."""
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, str(base_body))
    if base_id < 0:
        raise ValueError(f"base body not found while finding support: {base_body}")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    min_z = float("inf")
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != base_id:
            continue
        if int(model.geom_group[geom_id]) == 3:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            min_z = min(
                min_z,
                float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id]))
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vertex_start:vertex_start + vertex_count]
        R_geom = data.geom_xmat[geom_id].reshape(3, 3)
        z_values = (vertices @ R_geom.T + data.geom_xpos[geom_id])[:, 2]
        min_z = min(min_z, float(z_values.min()))
    if not np.isfinite(min_z):
        raise ValueError(f"base body has no visible geometry: {base_body}")
    return min_z


def _snap_base_to_support(base_pos, support_surface):
    """Place a base's visible sole on a fitted plane and report local gap."""
    base_pos = np.asarray(base_pos, dtype=float).copy()
    coefficients = np.asarray(support_surface["coefficients"], dtype=float)
    plane_z = (coefficients[0] * base_pos[0] +
               coefficients[1] * base_pos[1] + coefficients[2])
    base_up_z = float(support_surface.get("base_up_z", 1.0))
    base_pos[2] = (plane_z - base_up_z *
                   float(support_surface.get("base_min_z", 0.0)))
    points_xy = np.asarray(support_surface["points_xy"], dtype=float)
    if len(points_xy):
        local_gap = float(np.sqrt(np.min(np.sum(
            (points_xy - base_pos[:2]) ** 2, axis=1))))
    else:
        local_gap = float("inf")
    return base_pos, local_gap


def _print_progress(label, done, total, started, last_print,
                    force=False, width=28):
    """Print a throttled single-line progress bar and return its timestamp."""
    now = time.perf_counter()
    if not force and done < total and now - last_print < 1.0:
        return last_print
    total = max(int(total), 1)
    done = min(int(done), total)
    fraction = done / total
    filled = int(width * fraction)
    bar = "#" * filled + "." * (width - filled)
    elapsed = max(now - started, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else float("inf")
    eta_text = "--:--" if not np.isfinite(eta) else time.strftime(
        "%H:%M:%S", time.gmtime(max(0, int(eta))))
    print(f"\r    {label} [{bar}] {done}/{total} "
          f"{fraction:6.1%} {rate:6.1f}/s ETA {eta_text}",
          end="\n" if force else "", flush=True)
    return now


def base_frame(fwd_flat, up_axis=None):
    """Build a base frame whose local +Z is physical up and +X faces ahead."""
    z = (np.array([0.0, 0.0, 1.0]) if up_axis is None else
         np.asarray(up_axis, dtype=float))
    z /= max(np.linalg.norm(z), 1e-8)
    x = np.asarray(fwd_flat, dtype=float)
    x = x - z * float(x @ z)
    x /= max(np.linalg.norm(x), 1e-8)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return mat_to_quat(R), R


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_angle(R0, R1):
    trace = np.clip((np.trace(R0.T @ R1) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(trace))


def select_base_keyframes(targets_p_world, targets_R_world,
                          max_keyframes=BASE_MAX_KEYFRAMES):
    """Select endpoints, spatial extremes, and motion extremes like A.4."""
    n = len(targets_p_world)
    if n <= max_keyframes:
        return np.arange(n, dtype=int)
    selected = {0, n - 1}
    for axis in range(3):
        selected.add(int(np.argmin(targets_p_world[:, axis])))
        selected.add(int(np.argmax(targets_p_world[:, axis])))
    if n > 1:
        pos_delta = np.linalg.norm(np.diff(targets_p_world, axis=0), axis=1)
        rot_delta = np.array([
            _rotation_angle(targets_R_world[i], targets_R_world[i + 1])
            for i in range(n - 1)
        ])
        selected.add(int(np.argmax(pos_delta)) + 1)
        selected.add(int(np.argmax(rot_delta)) + 1)

    # Fill the remaining budget with farthest-point samples so a long motion
    # is represented even when its extrema occur in the same few frames.
    while len(selected) < max_keyframes:
        remaining = [i for i in range(n) if i not in selected]
        if not remaining:
            break
        chosen = np.array(sorted(selected), dtype=int)
        pos_scale = max(float(np.ptp(targets_p_world, axis=0).max()), 1e-6)
        scores = []
        for i in remaining:
            pos_score = np.min(np.linalg.norm(
                targets_p_world[i] - targets_p_world[chosen], axis=1)) / pos_scale
            rot_score = np.min([
                _rotation_angle(targets_R_world[i], targets_R_world[j])
                for j in chosen
            ]) / np.pi
            scores.append(pos_score + 0.25 * rot_score)
        selected.add(remaining[int(np.argmax(scores))])
    return np.array(sorted(selected), dtype=int)


def _base_orientation_candidates(R_nominal):
    for pitch in BASE_PITCH_DEG:
        for yaw in BASE_YAW_DEG:
            for roll in BASE_ROLL_DEG:
                delta = (_rot_z(np.deg2rad(yaw)) @
                         _rot_y(np.deg2rad(pitch)) @
                         _rot_x(np.deg2rad(roll)))
                yield R_nominal @ delta, (pitch, yaw, roll)


def _screen_orientation_candidates(R_nominal):
    """Return a small orientation cover for the balanced position screen."""
    wanted = {
        (30.0, 0.0, 0.0),
        (45.0, 0.0, 0.0),
        (60.0, 0.0, 0.0),
        (45.0, -45.0, 0.0),
        (45.0, 45.0, 0.0),
    }
    return [
        (R, euler) for R, euler in _base_orientation_candidates(R_nominal)
        if euler in wanted
    ]


def _candidate_score(record):
    return record["feasibility_rate"] - 5.0 * abs(
        record["reach_ratio"] - BASE_REACH_TARGET)


def _sort_base_records(records):
    return sorted(records, key=lambda r: (
        r["score"], r["feasibility_rate"], r["position_rate"],
        -r["mean_pos_error"],
    ), reverse=True)


def search_base_pose_original(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                              targets_p_world, targets_R_world,
                              R_nominal, fwd, sign, reach,
                              max_keyframes=DLS_KEYFRAMES, ik_solver="mink",
                              mink_context=None, mink_position_context=None,
                              support_surface=None):
    """Run the previous hand-anchored base search.

    The old implementation searched 120 absolute hand-anchored positions,
    kept the nominal base orientation, used position-only DLS as a gate, and
    only then ran full IK on the surviving positions. DLS reproduces the old
    behavior; the optional Mink backend changes only the IK calculations.
    """
    use_mink = ik_solver == "mink"
    position_solver = (solve_arm_ik_position_only if use_mink
                       else solve_arm_ik_position_only_dls)
    targets_p_world = np.asarray(targets_p_world, dtype=float)
    targets_R_world = np.asarray(targets_R_world, dtype=float)
    n_frames = len(targets_p_world)
    idx = (np.arange(n_frames, dtype=int) if n_frames <= max_keyframes else
           np.linspace(0, n_frames - 1, max_keyframes).astype(int))
    kf_p, kf_R = targets_p_world[idx], targets_R_world[idx]
    anchor = targets_p_world.mean(axis=0)
    fwd = np.asarray(fwd, dtype=float)
    lateral = np.array([-fwd[1], fwd[0], 0.0])

    def position_feasibility(base_pos):
        successes = 0
        errors = []
        for p_world in kf_p:
            ok, error = position_solver(
                model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                R_nominal.T @ (p_world - base_pos),
                max_iter=100, tol_pos=DLS_POS_TOL,
                lmbda=1e-2, step=0.5,
                mink_context=mink_position_context)
            successes += int(ok)
            errors.append(error)
        return successes / len(kf_p), float(np.mean(errors))

    def full_feasibility(base_pos):
        warm = None
        errors_pos, errors_rot = [], []
        successes = 0
        for p_world, R_world in zip(kf_p, kf_R):
            target_pos = R_nominal.T @ (p_world - base_pos)
            target_R = R_nominal.T @ R_world
            if use_mink:
                q, ok, err_pos, err_rot = solve_arm_ik_robust(
                    model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                    target_pos, target_R, q_warm=warm,
                    tol_pos=BASE_FEAS_POS_TOL,
                    tol_rot=BASE_FEAS_ROT_TOL,
                    mink_context=mink_context,
                    mink_position_context=mink_position_context,
                    # Base-search keyframes are sparse and can be far apart;
                    # do not apply the video-frame trust region here.
                    continuity_max_step=0.0)
            else:
                q, ok, err_pos, err_rot = solve_arm_ik_dls_robust(
                    model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                    target_pos, target_R, q_warm=warm,
                    max_iter=DLS_MAX_ITER, tol_pos=BASE_FEAS_POS_TOL,
                    tol_rot=BASE_FEAS_ROT_TOL)
            warm = q
            errors_pos.append(err_pos)
            errors_rot.append(err_rot)
            successes += int(ok)
        return (successes / len(kf_p), float(np.mean(errors_pos)),
                float(np.mean(errors_rot)))

    def make_record(base_pos, position_rate, mean_pos_error,
                    feasibility_rate, mean_full_pos_error,
                    mean_rot_error):
        return {
            "base_pos": np.asarray(base_pos, dtype=float).copy(),
            "base_R": np.asarray(R_nominal, dtype=float).copy(),
            "base_quat": mat_to_quat(R_nominal),
            "orientation": (0.0, 0.0, 0.0),
            "position_rate": position_rate,
            "feasibility_rate": feasibility_rate,
            "mean_pos_error": mean_full_pos_error,
            "position_mean_pos_error": mean_pos_error,
            "mean_rot_error": mean_rot_error,
            "reach_ratio": float(np.mean(
                np.linalg.norm(kf_p - base_pos, axis=1)) /
                max(float(reach), 1e-6)),
            "score": feasibility_rate,
        }

    # Preserve the legacy call order exactly: for each hand-anchored candidate,
    # run position feasibility first, then full IK immediately if it passes.
    best_record = None
    best_score = None
    dz_offsets = ((0.0,) if support_surface is not None else
                  DLS_HAND_ANCHOR_DZ)
    for back in DLS_HAND_ANCHOR_BACK:
        for lat in DLS_HAND_ANCHOR_LAT:
            for dz in dz_offsets:
                base_pos = (anchor - fwd * back +
                            lateral * (lat * sign) +
                            np.array([0.0, 0.0, dz]))
                if support_surface is not None:
                    base_pos, _ = _snap_base_to_support(
                        base_pos, support_surface)
                position_rate, position_error = position_feasibility(base_pos)
                if position_rate < DLS_POS_FEAS_GATE:
                    continue
                feasibility, pose_error, rot_error = full_feasibility(base_pos)
                record = make_record(
                    base_pos, position_rate, position_error,
                    feasibility, pose_error, rot_error)
                score = (feasibility, position_rate)
                if best_score is None or score > best_score:
                    best_record, best_score = record, score
                if feasibility >= 1.0 and position_rate >= 1.0:
                    return [best_record], idx

    if best_record is None:
        base_pos = anchor + np.array([0.0, 0.0, 0.15])
        if support_surface is not None:
            base_pos, _ = _snap_base_to_support(base_pos, support_surface)
        position_rate, position_error = position_feasibility(base_pos)
        feasibility, pose_error, rot_error = full_feasibility(base_pos)
        best_record = make_record(
            base_pos, position_rate, position_error,
            feasibility, pose_error, rot_error)
    return [best_record], idx


def search_base_pose(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                     targets_p_world, targets_R_world, R_nominal, fwd,
                     sign, reach, camera_pos=None,
                     max_keyframes=BASE_MAX_KEYFRAMES,
                     search_mode="fast", ik_solver="mink",
                     mink_context=None, mink_position_context=None,
                     base_orientation_mode="free",
                     enable_base_orientation_search=False,
                     base_height_anchor=None,
                     base_forward_anchor=None, base_forward_offsets=None,
                     max_target_distance=None, support_surface=None):
    """Search base poses using the Ego2Robot candidate grid and scoring.

    ``slow`` evaluates every surviving translation/orientation combination.
    With ``enable_base_orientation_search``, ``balanced`` preserves the
    paper's full orientation grid but uses five representative orientations
    to screen translations before full IK. It is disabled by default so the
    base remains at the nominal camera-facing orientation. ``fast`` is the
    previous single-orientation screen with a smaller shortlist. ``original``
    always restores the hand-anchored fixed-orientation search. The IK backend
    is selected independently.
    """
    search_mode = str(search_mode).lower()
    if search_mode not in {"original", "slow", "balanced", "fast"}:
        raise ValueError(f"unknown base search mode: {search_mode}")
    if ik_solver not in {"dls", "mink"}:
        raise ValueError(f"unknown IK solver: {ik_solver}")
    if base_orientation_mode not in {"free", "upright"}:
        raise ValueError(
            f"unknown base orientation mode: {base_orientation_mode}")
    if search_mode == "original":
        return search_base_pose_original(
            model, arm_qadr, arm_vadr, arm_ids, ee_ref,
            targets_p_world, targets_R_world, R_nominal, fwd, sign, reach,
            ik_solver=ik_solver, mink_context=mink_context,
            mink_position_context=mink_position_context,
            support_surface=support_surface)
    targets_p_world = np.asarray(targets_p_world, dtype=float)
    targets_R_world = np.asarray(targets_R_world, dtype=float)
    idx = select_base_keyframes(targets_p_world, targets_R_world, max_keyframes)
    kf_p, kf_R = targets_p_world[idx], targets_R_world[idx]
    anchor = targets_p_world.mean(axis=0)
    if base_height_anchor is not None:
        anchor[2] = float(base_height_anchor)
    fwd = np.asarray(fwd, dtype=float)
    fwd /= max(np.linalg.norm(fwd), 1e-8)
    if base_forward_anchor is not None:
        anchor += fwd * (float(base_forward_anchor) - float(anchor @ fwd))
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.array([-fwd[1], fwd[0], 0.0])
    lateral /= max(np.linalg.norm(lateral), 1e-8)
    reach = max(float(reach), 1e-6)
    max_target_distance = (
        BASE_TRAJ_MAX_REACH * reach if max_target_distance is None
        else float(max_target_distance)
    )
    if not np.isfinite(max_target_distance) or max_target_distance <= 0.0:
        raise ValueError("max_target_distance must be finite and positive")
    camera_pos = anchor if camera_pos is None else np.asarray(camera_pos, dtype=float)

    fixed_base_orientation = (
        base_orientation_mode == "upright" or
        (search_mode == "balanced" and
         not bool(enable_base_orientation_search))
    )
    all_orientations = ([(R_nominal, (0.0, 0.0, 0.0))]
                        if fixed_base_orientation else
                        list(_base_orientation_candidates(R_nominal)))
    if search_mode == "slow":
        # The paper grid is still enumerated below.  Its expensive evaluation
        # is coarse-to-fine; doing another all-orientation position QP here
        # would duplicate most of the final work.
        screen_orientations = []
        position_shortlist = None
        coarse_idx = select_base_keyframes(
            targets_p_world, targets_R_world, BASE_SLOW_COARSE_KEYFRAMES)
        eval_idx = coarse_idx
        eval_max_iter = BASE_SLOW_COARSE_MAX_ITER
    elif search_mode == "balanced":
        screen_orientations = (all_orientations if fixed_base_orientation else
                               _screen_orientation_candidates(R_nominal))
        position_shortlist = BASE_POSITION_SHORTLIST_BALANCED
        eval_idx = idx
        eval_max_iter = 100
    else:
        screen_orientations = [(R_nominal, (0.0, 0.0, 0.0))]
        position_shortlist = BASE_POSITION_SHORTLIST_FAST
        eval_idx = idx
        eval_max_iter = 100
    use_mink = ik_solver == "mink"
    position_solver = (solve_arm_ik_position_only if use_mink
                       else solve_arm_ik_position_only_dls)
    pose_solver = solve_arm_ik if use_mink else solve_arm_ik_dls
    screen_idx = select_base_keyframes(
        targets_p_world, targets_R_world, BASE_POSITION_SCREEN_KEYFRAMES)
    screen_p = targets_p_world[screen_idx]

    position_records = []
    forward_offsets = (BASE_FORWARD if base_forward_offsets is None
                       else tuple(base_forward_offsets))
    vertical_offsets = ((0.0,) if support_surface is not None else
                        BASE_VERTICAL)
    screen_total = (len(BASE_LATERAL) * len(forward_offsets) *
                    len(vertical_offsets) * len(screen_orientations) *
                    len(screen_p))
    screen_done = 0
    screen_started = time.perf_counter()
    screen_last_print = screen_started
    for lat in BASE_LATERAL:
        for forward in forward_offsets:
            for vertical in vertical_offsets:
                base_pos = (anchor + lateral * (sign * lat * reach) +
                            fwd * (forward * reach) + up * (vertical * reach))
                if support_surface is not None:
                    base_pos, _ = _snap_base_to_support(
                        base_pos, support_surface)
                camera_dist = float(np.linalg.norm(base_pos - camera_pos))
                traj_dist = np.linalg.norm(targets_p_world - base_pos, axis=1)
                if camera_dist < BASE_CAMERA_MIN_DIST:
                    continue
                if np.any(traj_dist > max_target_distance):
                    continue
                if np.any(traj_dist < BASE_TRAJ_MIN_REACH):
                    continue
                if search_mode == "slow":
                    # Geometric filtering above is the inexpensive first
                    # stage. Every surviving translation remains in the full
                    # 7x7x5x3x5x3 candidate enumeration below.
                    best_successes, best_mean_error = len(kf_p), 0.0
                else:
                    # Position feasibility depends on base orientation once
                    # joint limits are present. Use a small orientation cover
                    # for the non-exact fast screening modes.
                    screen_results = []
                    for screen_R, _ in screen_orientations:
                        errors = []
                        successes = 0
                        for p_world in screen_p:
                            ok, error = position_solver(
                                model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                                screen_R.T @ (p_world - base_pos),
                                max_iter=BASE_POSITION_SCREEN_MAX_ITER,
                                tol_pos=BASE_FEAS_POS_TOL,
                                mink_context=mink_position_context)
                            successes += int(ok)
                            errors.append(error)
                            screen_done += 1
                            screen_last_print = _print_progress(
                                "base_search position screen", screen_done,
                                screen_total, screen_started, screen_last_print)
                        screen_results.append((successes, float(np.mean(errors))))
                    best_successes, best_mean_error = max(
                        screen_results, key=lambda x: (x[0], -x[1]))
                distances = np.linalg.norm(kf_p - base_pos, axis=1)
                position_records.append({
                    "base_pos": base_pos,
                    "position_rate": best_successes / len(screen_p),
                    "mean_pos_error": best_mean_error,
                    "reach_ratio": float(np.mean(distances) / reach),
                })
    if screen_total:
        _print_progress("base_search position screen", screen_total,
                        screen_total, screen_started, screen_last_print,
                        force=True)

    if not position_records:
        # Keep the pipeline usable for an unusually short/degenerate trajectory.
        fallback_pos = (anchor - 0.3 * fwd * reach +
                        sign * 0.4 * lateral * reach + 0.15 * up * reach)
        if support_surface is not None:
            fallback_pos, _ = _snap_base_to_support(
                fallback_pos, support_surface)
        position_records = [{
            "base_pos": fallback_pos,
            "position_rate": 0.0,
            "mean_pos_error": float("inf"),
            "reach_ratio": float(np.mean(np.linalg.norm(kf_p - anchor, axis=1)) / reach),
        }]

    position_records.sort(key=lambda r: (
        r["position_rate"],
        -abs(r["reach_ratio"] - BASE_REACH_TARGET),
        -r["mean_pos_error"],
    ), reverse=True)
    if position_shortlist is not None:
        position_records = position_records[:position_shortlist]

    if search_mode == "slow":
        print(f"    base_search coarse grid: {len(position_records)} translations x "
              f"{len(all_orientations)} orientations = "
              f"{len(position_records) * len(all_orientations)} candidates; "
              f"{len(eval_idx)} keyframes, {eval_max_iter} IK iterations")

    records = []
    coarse_total = len(position_records) * len(all_orientations)
    coarse_done = 0
    coarse_started = time.perf_counter()
    coarse_last_print = coarse_started
    for position in position_records:
        base_pos = position["base_pos"]
        for base_R, euler in all_orientations:
            warm = None
            errors_pos, errors_rot = [], []
            successes = 0
            for frame_idx in eval_idx:
                p_world, R_world = targets_p_world[frame_idx], targets_R_world[frame_idx]
                p_base = base_R.T @ (p_world - base_pos)
                R_base_target = base_R.T @ R_world
                q, _, err_pos, err_rot = pose_solver(
                    model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                    p_base, R_base_target, q_init=warm,
                    max_iter=eval_max_iter, tol_pos=BASE_FEAS_POS_TOL,
                    tol_rot=BASE_FEAS_ROT_TOL, mink_context=mink_context)
                warm = q
                errors_pos.append(err_pos)
                errors_rot.append(err_rot)
                successes += int(err_pos < BASE_FEAS_POS_TOL and
                                 err_rot < BASE_FEAS_ROT_TOL)
            record = {
                "base_pos": base_pos.copy(),
                "base_R": base_R,
                "base_quat": mat_to_quat(base_R),
                "orientation": euler,
                "position_rate": position["position_rate"],
                "feasibility_rate": successes / len(eval_idx),
                "mean_pos_error": float(np.mean(errors_pos)),
                "mean_rot_error": float(np.mean(errors_rot)),
                "reach_ratio": position["reach_ratio"],
            }
            record["score"] = _candidate_score(record)
            records.append(record)
            coarse_done += 1
            coarse_last_print = _print_progress(
                "base_search coarse", coarse_done, coarse_total,
                coarse_started, coarse_last_print)
    if coarse_total:
        _print_progress("base_search coarse", coarse_total, coarse_total,
                        coarse_started, coarse_last_print, force=True)

    if search_mode == "slow":
        # Refine only the strongest candidates with the full A.4 keyframe set.
        # This preserves the complete candidate grid while avoiding up to 100
        # QP iterations on every keyframe for candidates already far behind.
        coarse_records = _sort_base_records(records)[:BASE_SLOW_REFINE_TOPK]
        refined = []
        refine_total = len(coarse_records)
        refine_started = time.perf_counter()
        refine_last_print = refine_started
        for refine_done, coarse in enumerate(coarse_records, 1):
            base_pos = coarse["base_pos"]
            base_R = coarse["base_R"]
            warm = None
            errors_pos, errors_rot = [], []
            successes = 0
            for p_world, R_world in zip(kf_p, kf_R):
                p_base = base_R.T @ (p_world - base_pos)
                R_base_target = base_R.T @ R_world
                q, _, err_pos, err_rot = pose_solver(
                    model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                    p_base, R_base_target, q_init=warm,
                    max_iter=100, tol_pos=BASE_FEAS_POS_TOL,
                    tol_rot=BASE_FEAS_ROT_TOL, mink_context=mink_context)
                warm = q
                errors_pos.append(err_pos)
                errors_rot.append(err_rot)
                successes += int(err_pos < BASE_FEAS_POS_TOL and
                                 err_rot < BASE_FEAS_ROT_TOL)
            record = dict(coarse)
            record["feasibility_rate"] = successes / len(kf_p)
            record["mean_pos_error"] = float(np.mean(errors_pos))
            record["mean_rot_error"] = float(np.mean(errors_rot))
            record["score"] = _candidate_score(record)
            refined.append(record)
            refine_last_print = _print_progress(
                "base_search refine", refine_done, refine_total,
                refine_started, refine_last_print)
        if refine_total:
            _print_progress("base_search refine", refine_total, refine_total,
                            refine_started, refine_last_print, force=True)
        records = refined
    records = _sort_base_records(records)
    return records[:BASE_RESULT_TOPK], idx


def _is_cross_arm_pair(body1, body2):
    return ((body1.startswith("left_") and body2.startswith("right_")) or
            (body1.startswith("right_") and body2.startswith("left_")))


def _is_known_baseline_self_contact(body1, body2):
    """Ignore fixed mesh overlap present in the Kinova Gen3 XML.

    The menagerie model has a persistent -0.012 m overlap between each
    ``base_link`` and its child ``shoulder_link``. It is present at every
    joint configuration and therefore is a model-geometry artifact rather
    than a trajectory collision. The names are checked after the arm prefix,
    so this applies independently to the left and right copy.
    """
    names = {
        body1.removeprefix("left_").removeprefix("right_"),
        body2.removeprefix("left_").removeprefix("right_"),
    }
    return names == {"base_link", "shoulder_link"}


def _cross_arm_collision_metrics(model, data):
    """Return cross-arm contacts and penetration for the current qpos."""
    contacts = 0
    penetration = 0.0
    min_distance = float("inf")
    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[contact.geom1])) or ""
        body2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[contact.geom2])) or ""
        if not _is_cross_arm_pair(body1, body2):
            continue
        distance = float(contact.dist)
        contacts += 1
        penetration += max(0.0, -distance)
        min_distance = min(min_distance, distance)
    return contacts, penetration, min_distance


def _quality_collision_metrics(model, data):
    """Return conservative per-frame self/cross-arm collision metrics.

    MuJoCo can report near contacts within a contact margin.  The quality gate
    therefore counts only penetrating contacts (``dist < 0``), so normal
    gripper closure at or near contact is not rejected.  Self collision is
    restricted to contacts between bodies belonging to the same attached arm;
    contacts involving the camera/world are ignored.
    """
    self_contacts = 0
    cross_contacts = 0
    self_penetration = 0.0
    cross_penetration = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[contact.geom1])) or ""
        body2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[contact.geom2])) or ""
        left_pair = body1.startswith("left_") and body2.startswith("left_")
        right_pair = body1.startswith("right_") and body2.startswith("right_")
        distance = float(contact.dist)
        if distance >= 0.0:
            continue
        if _is_known_baseline_self_contact(body1, body2):
            continue
        if _is_cross_arm_pair(body1, body2):
            cross_contacts += 1
            cross_penetration += max(0.0, -distance)
        elif left_pair or right_pair:
            self_contacts += 1
            self_penetration += max(0.0, -distance)
    return {
        "self_collision": bool(self_contacts),
        "self_contact_count": int(self_contacts),
        "self_penetration": float(self_penetration),
        "cross_arm_contact_count": int(cross_contacts),
        "cross_arm_penetration": float(cross_penetration),
    }


def _collision_branch_repair(
        dual, qpos_all, Q_l, Q_r, ik_pose_ok, ik_err_pos, ik_err_rot,
        single, arm_qadr, arm_ids, l_qadr, r_qadr, left_targets_p,
        left_targets_R, right_targets_p, right_targets_R, left_base_pos,
        left_base_R, right_base_pos, right_base_R, ee_ref, mink_context,
        mink_position_context):
    """Repair cross-arm intersections using alternative per-arm IK branches.

    Base-pair screening cannot see every frame and the final robust IK may pick
    a different branch.  Only frames that actually collide are retried, so
    collision-free frames keep their original trajectory and cost.
    """
    data = mujoco.MjData(dual)
    repaired = 0
    checked = 0

    def arm_seed(q_values):
        q = single.qpos0.copy()
        for i, qa in enumerate(arm_qadr):
            q[qa] = q_values[i]
        return q

    def random_seed(frame_idx, salt):
        rng = np.random.default_rng(1009 * (frame_idx + 1) + salt)
        q = _project_qpos_to_limits(single, single.qpos0)
        for i, jid in enumerate(arm_ids):
            if single.jnt_limited[jid]:
                lo, hi = single.jnt_range[jid]
                q[arm_qadr[i]] = rng.uniform(lo, hi)
        return q

    def solve_candidates(target_p, target_R, base_pos, base_R, current, frame_idx, salt):
        target_p = base_R.T @ (target_p - base_pos)
        target_R = base_R.T @ target_R
        seeds = [arm_seed(current)]
        # The VX300S has several useful elbow/wrist branches.  A couple of
        # random restarts are not enough when both wrists target the same
        # small workspace, so sample a deterministic, bounded branch set only
        # for frames that already collide.
        seeds.extend(
            random_seed(frame_idx, salt + i)
            for i in range(BASE_COLLISION_REPAIR_RESTARTS))
        out = []
        for seed_idx, seed in enumerate(seeds):
            continuity_q = arm_seed(current) if seed_idx == 0 else None
            q, ok, ep, er = solve_arm_ik(
                single, arm_qadr, None, arm_ids, ee_ref, target_p, target_R,
                q_init=seed, max_iter=100, tol_pos=BASE_FEAS_POS_TOL,
                tol_rot=BASE_FEAS_ROT_TOL, mink_context=mink_context,
                continuity_q=continuity_q)
            out.append((q, bool(ok), float(ep), float(er)))
        return out

    for t in range(len(qpos_all)):
        data.qpos[:] = qpos_all[t]
        mujoco.mj_forward(dual, data)
        contacts, current_penetration, current_min_distance = \
            _cross_arm_collision_metrics(dual, data)
        if not contacts:
            continue
        checked += 1
        left_current = Q_l[t].copy()
        right_current = Q_r[t].copy()
        left_options = solve_candidates(
            left_targets_p[t], left_targets_R[t], left_base_pos, left_base_R,
            left_current, t, 11)
        right_options = solve_candidates(
            right_targets_p[t], right_targets_R[t], right_base_pos, right_base_R,
            right_current, t, 29)

        candidates = []
        for left in left_options:
            for right in right_options:
                q_candidate = qpos_all[t].copy()
                for i, qa in enumerate(l_qadr):
                    q_candidate[qa] = left[0][arm_qadr[i]]
                for i, qa in enumerate(r_qadr):
                    q_candidate[qa] = right[0][arm_qadr[i]]
                data.qpos[:] = q_candidate
                mujoco.mj_forward(dual, data)
                c, penetration, min_distance = _cross_arm_collision_metrics(dual, data)
                continuity = float(
                    np.linalg.norm(left[0][arm_qadr] - left_current) +
                    np.linalg.norm(right[0][arm_qadr] - right_current))
                candidates.append((
                    c, penetration, -min_distance,
                    not (left[1] and right[1]), left[2] + right[2],
                    continuity, q_candidate, left, right,
                ))

        chosen = min(candidates, key=lambda x: x[:6])
        current_penalty = (
            contacts, current_penetration, -current_min_distance,
            True, float("inf"), float("inf"))
        if chosen[:6] < current_penalty:
            _, _, _, _, _, _, q_candidate, left, right = chosen
            qpos_all[t] = q_candidate
            Q_l[t] = left[0][arm_qadr]
            Q_r[t] = right[0][arm_qadr]
            ik_pose_ok[t] = [left[1], right[1]]
            ik_err_pos[t] = [left[2], right[2]]
            ik_err_rot[t] = [left[3], right[3]]
            repaired += 1

    if checked:
        print(f"    cross-arm collision repair: checked={checked} "
              f"collision frames, repaired={repaired}")
    return repaired


def _evaluate_base_pair_collisions(
        single, arm_qadr, arm_vadr, arm_ids, ee_ref,
        left_targets_p, left_targets_R, right_targets_p, right_targets_R,
        left_choice, right_choice, left_keyframes, right_keyframes, spec,
        mink_context, mink_position_context=None,
        ik_solver=solve_arm_ik, use_robust_ik=False):
    """Screen one base pair using representative dual-arm IK poses.

    The helper re-solves representative keyframes for each arm independently,
    inserts both solutions into the actual dual model, and lets MuJoCo report
    cross-arm geometry penetration.
    """
    frame_idx = np.unique(np.concatenate((
        np.asarray(left_keyframes, dtype=int),
        np.asarray(right_keyframes, dtype=int),
    )))
    if len(frame_idx) > BASE_PAIR_COLLISION_MAX_KEYFRAMES:
        picks = np.linspace(0, len(frame_idx) - 1,
                            BASE_PAIR_COLLISION_MAX_KEYFRAMES).round().astype(int)
        frame_idx = frame_idx[picks]

    dual = build_dual_model(
        left_choice["base_pos"], left_choice["base_quat"],
        right_choice["base_pos"], right_choice["base_quat"],
        fovy_deg=60.0, spec=spec)
    _, left_qadr, _ = name_to_dof(dual, spec.arm_joints, "left_")
    _, right_qadr, _ = name_to_dof(dual, spec.arm_joints, "right_")
    data = mujoco.MjData(dual)
    qpos = dual.qpos0.copy()
    # Keep the collision screen's initial state aligned with the production
    # trajectory solver.  Starting from None here can select a different IK
    # branch from the one used by the final frame-by-frame pass.
    if use_robust_ik:
        warm_left = _prewarm(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref,
            left_choice["base_R"].T @ (left_targets_p[0] - left_choice["base_pos"]),
            mink_context=mink_position_context or mink_context)
        warm_right = _prewarm(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref,
            right_choice["base_R"].T @ (right_targets_p[0] - right_choice["base_pos"]),
            mink_context=mink_position_context or mink_context)
    else:
        warm_left = None
        warm_right = None
    collision_frames = 0
    contact_count = 0
    penetration = 0.0
    min_distance = float("inf")

    for t in frame_idx:
        left_base_R = left_choice["base_R"]
        right_base_R = right_choice["base_R"]
        if use_robust_ik:
            left_q, left_ok, left_ep, left_er = solve_arm_ik_robust(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref,
                left_base_R.T @ (left_targets_p[t] - left_choice["base_pos"]),
                left_base_R.T @ left_targets_R[t], q_warm=warm_left,
                tol_pos=BASE_FEAS_POS_TOL, tol_rot=BASE_FEAS_ROT_TOL,
                mink_context=mink_context,
                mink_position_context=mink_position_context or mink_context,
                continuity_max_step=0.0)
            right_q, right_ok, right_ep, right_er = solve_arm_ik_robust(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref,
                right_base_R.T @ (right_targets_p[t] - right_choice["base_pos"]),
                right_base_R.T @ right_targets_R[t], q_warm=warm_right,
                tol_pos=BASE_FEAS_POS_TOL, tol_rot=BASE_FEAS_ROT_TOL,
                mink_context=mink_context,
                mink_position_context=mink_position_context or mink_context,
                continuity_max_step=0.0)
        else:
            left_q, left_ok, left_ep, left_er = ik_solver(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref,
                left_base_R.T @ (left_targets_p[t] - left_choice["base_pos"]),
                left_base_R.T @ left_targets_R[t], q_init=warm_left,
                max_iter=BASE_PAIR_COLLISION_MAX_ITER,
                tol_pos=BASE_FEAS_POS_TOL, tol_rot=BASE_FEAS_ROT_TOL,
                mink_context=mink_context)
            right_q, right_ok, right_ep, right_er = ik_solver(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref,
                right_base_R.T @ (right_targets_p[t] - right_choice["base_pos"]),
                right_base_R.T @ right_targets_R[t], q_init=warm_right,
                max_iter=BASE_PAIR_COLLISION_MAX_ITER,
                tol_pos=BASE_FEAS_POS_TOL, tol_rot=BASE_FEAS_ROT_TOL,
                mink_context=mink_context)
        warm_left, warm_right = left_q, right_q
        qpos[:] = dual.qpos0
        for i, qa in enumerate(left_qadr):
            qpos[qa] = left_q[arm_qadr[i]]
        for i, qa in enumerate(right_qadr):
            qpos[qa] = right_q[arm_qadr[i]]
        data.qpos[:] = qpos
        mujoco.mj_forward(dual, data)
        contacts, frame_penetration, frame_min_distance = \
            _cross_arm_collision_metrics(dual, data)
        if contacts:
            collision_frames += 1
        contact_count += contacts
        penetration += frame_penetration
        min_distance = min(min_distance, frame_min_distance)

    return {
        "collision_frames": collision_frames,
        "contact_count": contact_count,
        "penetration": penetration,
        "min_distance": min_distance,
        "checked_frames": len(frame_idx),
    }


def select_joint_base_pair(
        left_candidates, right_candidates, single=None,
        arm_qadr=None, arm_vadr=None, arm_ids=None, ee_ref=None,
        left_targets_p=None, left_targets_R=None,
        right_targets_p=None, right_targets_R=None,
        left_keyframes=None, right_keyframes=None, spec=None,
        mink_context=None, mink_position_context=None,
        ik_solver=solve_arm_ik, use_robust_ik=False,
        enable_collision_screen=True, base_alignment_axis=None):
    """Select the best top-5 x top-5 pair, with optional bilateral screening."""
    if not left_candidates or not right_candidates:
        raise ValueError("base search returned no candidates")
    alignment_axis = None
    if base_alignment_axis is not None:
        alignment_axis = np.asarray(base_alignment_axis, dtype=float)
        alignment_axis /= max(np.linalg.norm(alignment_axis), 1e-8)
    pairs = []
    for left in left_candidates[:BASE_RESULT_TOPK]:
        for right in right_candidates[:BASE_RESULT_TOPK]:
            pairs.append({
                "score": left["score"] + right["score"],
                "feasibility": left["feasibility_rate"] + right["feasibility_rate"],
                "left": left,
                "right": right,
                "base_height_delta": abs(float(
                    left["base_pos"][2] - right["base_pos"][2])),
                "base_depth_delta": (0.0 if alignment_axis is None else abs(float(
                    (left["base_pos"] - right["base_pos"]) @ alignment_axis))),
            })

    if spec is not None and spec.coplanar_bases:
        # A desktop dual-arm setup cannot realize independently floating base
        # heights. Keep pairs on one mounting plane; if the discrete grids do
        # not contain a pair within tolerance, retain only the closest pair(s).
        coplanar = [p for p in pairs if p["base_height_delta"] <= 0.05]
        if coplanar:
            pairs = coplanar
        else:
            min_delta = min(p["base_height_delta"] for p in pairs)
            pairs = [p for p in pairs
                     if p["base_height_delta"] <= min_delta + 1e-9]

    if spec is not None and spec.aligned_base_depths:
        aligned = [p for p in pairs if p["base_depth_delta"] <= 0.05]
        if aligned:
            pairs = aligned
        else:
            min_delta = min(p["base_depth_delta"] for p in pairs)
            pairs = [p for p in pairs
                     if p["base_depth_delta"] <= min_delta + 1e-9]

    collision_enabled = (
        spec is not None and
        single is not None and arm_qadr is not None and arm_vadr is not None and
        arm_ids is not None and ee_ref is not None and
        left_targets_p is not None and left_targets_R is not None and
        right_targets_p is not None and right_targets_R is not None and
        left_keyframes is not None and right_keyframes is not None and
        enable_collision_screen)
    if not collision_enabled:
        best = max(pairs, key=lambda p: (p["score"], p["feasibility"]))
        return best["left"], best["right"]

    started = time.perf_counter()
    for pair in pairs:
        pair["collision"] = _evaluate_base_pair_collisions(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref,
            left_targets_p, left_targets_R, right_targets_p, right_targets_R,
            pair["left"], pair["right"], left_keyframes, right_keyframes,
            spec, mink_context, mink_position_context=mink_position_context,
            ik_solver=ik_solver, use_robust_ik=use_robust_ik)
    # Collision-free remains a hard preference, but the old ordering compared
    # penetration and distance before IK feasibility.  It could therefore pick
    # a collision-free pair whose right arm was unreachable for much of the
    # trajectory.  Among equally collision-safe pairs, preserve the base-search
    # feasibility/score as the primary quality signal.
    # Collision quality must be ordered before IK quality.  Otherwise, when
    # all top-k pairs collide, the old ordering selected the most reachable
    # pair even if it had substantially more penetration than another pair.
    best = max(pairs, key=lambda p: (
        p["collision"]["collision_frames"] == 0,
        -p["collision"]["collision_frames"],
        -p["collision"]["penetration"],
        p["collision"]["min_distance"],
        p["feasibility"],
        p["score"],
    ))
    c = best["collision"]
    elapsed = time.perf_counter() - started
    print(f"    dual-arm collision screen: {len(pairs)} pairs, "
          f"{c['checked_frames']} keyframes, selected "
          f"{c['collision_frames']} collision frames, "
          f"min_dist={c['min_distance'] * 1000:.1f}mm, "
          f"base_dz={best['base_height_delta'] * 1000:.1f}mm, "
          f"base_depth_delta={best['base_depth_delta'] * 1000:.1f}mm, "
          f"{elapsed:.1f}s")
    return best["left"], best["right"]
