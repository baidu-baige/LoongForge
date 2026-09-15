# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Single-episode retargeting orchestration and artifact generation."""

from pathlib import Path

import cv2
import mujoco
import numpy as np
import zarr

try:
    from .. import config
    from ..robot_registry import get_robot_spec, resolve_ee_ref
except ImportError:
    import config
    from robot_registry import get_robot_spec, resolve_ee_ref

from .base_search import (
    BASE_FEAS_POS_TOL,
    BASE_FEAS_ROT_TOL,
    _base_visual_min_z,
    _collision_branch_repair,
    _quality_collision_metrics,
    _snap_base_to_support,
    base_frame,
    estimate_scene_support_surface,
    search_base_pose,
    select_joint_base_pair,
)
from .ik import (
    DLS_MAX_ITER,
    MINK_CONTINUITY_COST,
    MINK_CONTINUITY_MAX_STEP,
    POSE_REFINEMENT_MAX_ITER,
    POSITION_FIRST_TOL,
    MinkIKContext,
    _prewarm_dls,
    _project_qpos_to_limits,
    refine_arm_ik_pose,
    solve_arm_ik,
    solve_arm_ik_dls,
    solve_arm_ik_dls_robust,
    solve_arm_ik_position_first,
    solve_arm_ik_robust,
)
from .rendering import (
    build_dual_model,
    build_single_arm_model,
    feather_mask,
    hide_non_arm_geoms,
    load_scene_depth,
    make_renderer,
    name_to_dof,
    patch_arm_holes,
    read_video_frames,
    render_rgb_and_mask,
    set_ego_camera,
    set_gripper_qpos,
    set_wrist_cameras,
    tone_map_robot,
)
from .targets import (
    _fill_invalid_keypoints,
    eq12_pose_world,
    smooth_retarget_targets,
    target_ref_pose,
)


# ============ Data loading ============

def load_episode_world(zarr_dir, ep, n_frames):
    """Read world-frame head and bimanual keypoints for the first N valid frames. Returns:
        head(N,7), left_kp(N,63), right_kp(N,63)
    """
    z = zarr.open_group(str(Path(zarr_dir) / ep), mode="r")
    head = np.array(z["obs_head_pose"][:n_frames])
    lk = np.array(z["left.obs_keypoints"][:n_frames])
    rk = np.array(z["right.obs_keypoints"][:n_frames])
    return head, lk, rk


def head_forward_flat(head_seq):
    """Project the human gaze onto the horizontal plane to obtain robot yaw."""
    def q2m(q):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
        return R.reshape(3, 3)
    fwds = []
    for i in range(len(head_seq)):
        R = q2m(head_seq[i, 3:7])
        fwds.append(R @ np.array([0.0, 0.0, 1.0]))  # Camera +z is the viewing direction.
    f = np.mean(fwds, axis=0)
    f[2] = 0.0
    if np.linalg.norm(f) < 1e-6:
        # Path-B's synthetic identity head has no horizontal gaze component.
        # Pick the camera's image-up plane as a stable forward axis so the
        # bilateral lateral axis remains world X rather than world Y (which
        # would place both mounts at the image edges).
        f = np.array([0.0, -1.0, 0.0])
    return f / np.linalg.norm(f)


def repair_branch_jumps(model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                        Q, targets_p_world, targets_R_world, base_pos, R_base,
                        pose_ok_flags, position_ok_flags=None,
                        tol_pos=8e-3, tol_rot=0.25,
                        jump_thresh=0.45, n_sweeps=3, mink_context=None,
                        ik_solver=solve_arm_ik):
    """Re-solve IK branch jumps to enforce temporal continuity.

    A separate pass is needed because `solve_arm_ik_robust` can apply jump_penalty only
    among candidates for the current frame. A common failure is an unreachable target
    breaking the warm chain, prewarm selecting another branch, and subsequent frames
    remaining on it. The penalty then locks the sequence onto that poor branch, visibly
    flipping the arm while the base and end effector remain still.

    A forward scan from t=1 to N-1 re-solves frames whose joint delta exceeds
    jump_thresh, using the potentially repaired Q[t-1] as q_init. A result is
    accepted when it meets the position tolerance and reduces the jump. Full
    pose success is tracked separately: orientation-only failures may therefore
    be repaired without sacrificing pinch-center alignment. Because Q[t-1] is
    already repaired, the correction naturally propagates forward. A reverse
    scan then handles jumps near the beginning of the sequence.

    Q is an (N, 7) arm-joint sequence modified in place. Returns (fixed frame count,
    updated pose and position success flags)."""
    N = len(Q)
    n_fixed = 0
    pose_ok_flags = np.asarray(pose_ok_flags, dtype=bool).copy()
    if position_ok_flags is None:
        position_ok_flags = pose_ok_flags.copy()
    else:
        position_ok_flags = np.asarray(position_ok_flags, dtype=bool).copy()

    def try_resolve(t, q_ref):
        """Re-solve frame t from q_ref. Return (q_arm, ep, er), or None."""
        p_b = R_base.T @ (targets_p_world[t] - base_pos)
        R_b = R_base.T @ targets_R_world[t]
        use_legacy_dls = ik_solver is solve_arm_ik_dls
        q0 = (np.zeros(model.nq) if use_legacy_dls else
              _project_qpos_to_limits(model, model.qpos0))
        for i, qa in enumerate(arm_qadr):
            q0[qa] = q_ref[i]
        if use_legacy_dls:
            # Legacy branch repair kept the DLS solver's stricter default
            # orientation stopping tolerance (0.15), then accepted 0.25 here.
            q, _, ep_, er_ = ik_solver(
                model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                p_b, R_b, q_init=q0, tol_pos=tol_pos)
        else:
            q, _, ep_, er_ = ik_solver(
                model, arm_qadr, arm_vadr, arm_ids, ee_ref,
                p_b, R_b, q_init=q0, tol_pos=tol_pos, tol_rot=tol_rot,
                # Mink tasks operate on the complete model configuration
                # (nq), while q_ref is the 7-joint arm vector used by Q.
                continuity_q=q0)
        # Preserve the position-aligned solution even when this morphology
        # cannot realize the requested human wrist orientation continuously.
        if ep_ < tol_pos:
            return np.array([q[qa] for qa in arm_qadr]), ep_, er_
        return None

    for _ in range(n_sweeps):
        changed = 0
        # Forward pass: use the repaired previous frame to guide the next frame.
        for t in range(1, N):
            if np.linalg.norm(Q[t] - Q[t - 1]) <= jump_thresh:
                continue
            r = try_resolve(t, Q[t - 1])
            if r is not None and np.linalg.norm(r[0] - Q[t - 1]) < \
                    np.linalg.norm(Q[t] - Q[t - 1]):
                Q[t] = r[0]
                position_ok_flags[t] = True
                pose_ok_flags[t] = r[2] < tol_rot
                changed += 1
        # Reverse pass: handle initial jumps that the forward pass cannot repair at t=0.
        for t in range(N - 2, -1, -1):
            if np.linalg.norm(Q[t + 1] - Q[t]) <= jump_thresh:
                continue
            r = try_resolve(t, Q[t + 1])
            if r is not None and np.linalg.norm(r[0] - Q[t + 1]) < \
                    np.linalg.norm(Q[t] - Q[t + 1]):
                Q[t] = r[0]
                position_ok_flags[t] = True
                pose_ok_flags[t] = r[2] < tol_rot
                changed += 1
        n_fixed += changed
        if changed == 0:
            break
    return n_fixed, pose_ok_flags, position_ok_flags


def interp_failed_frames(qpos_all, ik_position_ok, l_qadr, r_qadr,
                         branch_jump_thresh=0.0):
    """Fill position-failed frames without blending incompatible branches."""
    n_fixed = 0
    for arm, qadr in ((0, l_qadr), (1, r_qadr)):
        ok = ik_position_ok[:, arm]
        bad_idx = np.where(~ok)[0]
        if len(bad_idx) == 0:
            continue
        good = np.where(ok)[0]
        if len(good) < 2:
            continue
        # ``searchsorted(..., side="left")`` returns the first good frame at
        # or after each failed frame.  Use it as the upper endpoint and step
        # back for the lower endpoint; treating it as ``lo`` would give a
        # negative interpolation weight and extrapolate beyond the trajectory.
        hi = np.searchsorted(good, bad_idx, side="left")
        lo = hi - 1
        # At either sequence boundary there is only one usable endpoint. Hold
        # that nearest solution instead of extrapolating outside the good span.
        before_first = hi <= 0
        after_last = hi >= len(good)
        lo = np.clip(lo, 0, len(good) - 1)
        hi = np.clip(hi, 0, len(good) - 1)
        lo[before_first] = hi[before_first]
        hi[after_last] = lo[after_last]
        lo_t = good[lo]
        hi_t = good[hi]
        for i, t in enumerate(bad_idx):
            if hi_t[i] == lo_t[i]:
                weight = 0.0
            else:
                weight = (t - lo_t[i]) / max(hi_t[i] - lo_t[i], 1)
            endpoints_cross_branches = (
                branch_jump_thresh > 0.0 and hi_t[i] != lo_t[i] and
                np.max(np.abs(
                    qpos_all[hi_t[i], qadr] -
                    qpos_all[lo_t[i], qadr])) > branch_jump_thresh)
            if endpoints_cross_branches:
                # A linear blend between two valid but incompatible IK
                # branches is generally not a solution at all. Preserve the
                # preceding branch until a later frame is solved explicitly.
                qpos_all[t, qadr] = qpos_all[lo_t[i], qadr]
            else:
                qpos_all[t, qadr] = (
                    (1.0 - weight) * qpos_all[lo_t[i], qadr] +
                    weight * qpos_all[hi_t[i], qadr])
            n_fixed += 1
    return n_fixed


def clamp_joint_speed(qpos_all, qadr, vmax_rad_per_frame, iters=2):
    """Limit the arm's joint-space displacement between adjacent frames."""
    if vmax_rad_per_frame <= 0.0:
        return 0
    n_clamped = 0
    for _ in range(iters):
        for t in range(1, len(qpos_all)):
            delta = qpos_all[t, qadr] - qpos_all[t - 1, qadr]
            norm = float(np.linalg.norm(delta))
            if norm > vmax_rad_per_frame:
                qpos_all[t, qadr] = (
                    qpos_all[t - 1, qadr] +
                    delta * (vmax_rad_per_frame / norm))
                n_clamped += 1
    return n_clamped


def process_episode(ep, zarr_dir, bg_path, state, out_dir,
                    fps, feather, fy, height, debug,
                    scene_depth_path=None, depth_epsilon=0.02,
                    depth_mode="depth-aware", base_search_mode="balanced",
                    enable_base_orientation_search=False,
                    ik_solver="mink", base_pullback=0.0,
                    max_jump_rad=0.0,
                    mink_continuity_max_step=MINK_CONTINUITY_MAX_STEP,
                    mink_continuity_cost=MINK_CONTINUITY_COST, spec=None,
                    target_smooth_window=11, target_smooth_polyorder=3,
                    target_orientation_sigma=6.0):
    """Run one episode for a registered dual-arm morphology."""
    spec = spec or get_robot_spec("panda")
    base_pullback = float(base_pullback or 0.0)
    if not np.isfinite(base_pullback) or base_pullback < 0.0:
        raise ValueError(f"base_pullback must be a finite non-negative distance, got {base_pullback}")
    base_search_mode = str(base_search_mode).lower()
    if base_search_mode not in {"original", "balanced", "slow", "fast"}:
        raise ValueError(f"unknown base search mode: {base_search_mode}")
    ik_solver = str(ik_solver).lower()
    if ik_solver not in {"dls", "mink"}:
        raise ValueError(f"unknown IK solver: {ik_solver}")
    use_mink_ik = ik_solver == "mink"
    max_jump_rad = float(max_jump_rad)
    if not np.isfinite(max_jump_rad) or max_jump_rad < 0.0:
        raise ValueError(f"max_jump_rad must be finite and non-negative, got {max_jump_rad}")
    mink_continuity_max_step = float(mink_continuity_max_step)
    mink_continuity_cost = float(mink_continuity_cost)
    if (not np.isfinite(mink_continuity_max_step) or
            mink_continuity_max_step < 0.0):
        raise ValueError("mink_continuity_max_step must be finite and non-negative")
    if not np.isfinite(mink_continuity_cost) or mink_continuity_cost < 0.0:
        raise ValueError("mink_continuity_cost must be finite and non-negative")
    target_smooth_window = int(target_smooth_window)
    target_smooth_polyorder = int(target_smooth_polyorder)
    target_orientation_sigma = float(target_orientation_sigma)
    if target_smooth_window < 0:
        raise ValueError("target_smooth_window must be non-negative")
    if target_smooth_polyorder < 0:
        raise ValueError("target_smooth_polyorder must be non-negative")
    if not np.isfinite(target_orientation_sigma) or target_orientation_sigma < 0.0:
        raise ValueError("target_orientation_sigma must be finite and non-negative")
    N = len(state)
    head, left_kp, right_kp = load_episode_world(zarr_dir, ep, N)
    assert len(head) == N, f"head has {len(head)} frames, state {N}"
    scene_depth = None
    scene_depth_metadata = None
    ep_attrs = None

    # Quality curation uses the original hand tracks as the hand-detection
    # signal.  SAM3's arm/person mask is a visual-removal mask, not a hand
    # detector, so it must not be used for this gate.
    def valid_keypoint_frame(keypoints):
        values = np.asarray(keypoints).reshape(len(keypoints), -1)
        return np.isfinite(values).all(axis=1) & (np.abs(values) < 1e8).all(axis=1)

    hand_detected = valid_keypoint_frame(left_kp) & valid_keypoint_frame(right_kp)

    # Resolve the morphology before converting TCP targets.  Body-referenced
    # models receive an explicit TCP site during model construction; using the
    # resolved spec here prevents applying the body-to-TCP offset twice.
    single, spec = build_single_arm_model(spec, return_spec=True)

    # 1. Common human TCP pose -> this morphology's IK reference pose.  The
    # reference implementation smooths Eq.1/Eq.2 outputs before IK; apply the
    # same treatment here so noisy fingertip cross-products cannot drive the
    # redundant arm through alternating IK branches.
    left_kp_smooth = _fill_invalid_keypoints(left_kp)
    right_kp_smooth = _fill_invalid_keypoints(right_kp)
    left_tcp_p, left_tcp_R, left_w, _ = eq12_pose_world(left_kp_smooth, hand_sign=-1.0)
    right_tcp_p, right_tcp_R, right_w, _ = eq12_pose_world(right_kp_smooth, hand_sign=+1.0)
    (left_tcp_p, left_tcp_R, left_w) = smooth_retarget_targets(
        left_tcp_p, left_tcp_R, left_w,
        window=target_smooth_window, polyorder=target_smooth_polyorder,
        orientation_sigma=target_orientation_sigma, width_max=spec.gripper_max)
    (right_tcp_p, right_tcp_R, right_w) = smooth_retarget_targets(
        right_tcp_p, right_tcp_R, right_w,
        window=target_smooth_window, polyorder=target_smooth_polyorder,
        orientation_sigma=target_orientation_sigma, width_max=spec.gripper_max)
    left_p, left_R = target_ref_pose(
        spec, left_tcp_p, left_tcp_R, spec.tcp_rot_site_left,
        opening_width=left_w)
    right_p, right_R = target_ref_pose(
        spec, right_tcp_p, right_tcp_R, spec.tcp_rot_site_right,
        opening_width=right_w)

    # 2. Nominal camera-facing frame; the paper search varies translation and
    # orientation around it independently for both arms.
    fwd = head_forward_flat(head)
    _, nominal_R_base = base_frame(fwd)
    camera_pos = np.mean(head[:, :3], axis=0)
    # Reuse the single-arm model for base_search and later IK; the dual-arm model is for rendering only.
    arm_ids, arm_qadr, arm_vadr = name_to_dof(single, spec.arm_joints)
    ee_ref_single = resolve_ee_ref(single, spec)
    mink_context = (MinkIKContext(
                        single, arm_ids, ee_ref_single,
                        orientation_cost=spec.ik_orientation_cost,
                        continuity_max_step=mink_continuity_max_step,
                        continuity_cost=mink_continuity_cost)
                    if use_mink_ik else None)
    mink_position_context = (
        MinkIKContext(single, arm_ids, ee_ref_single, orientation_cost=0.0)
        if use_mink_ik else None)
    support_surface = None
    if spec.scene_support_surface:
        if depth_mode != "depth-aware":
            print(f"    [{ep}] scene support disabled: metric depth is unavailable")
        elif scene_depth_path is None:
            raise ValueError(
                f"SO-ARM101 scene-support search requires scene depth for {ep}")
        else:
            scene_depth, scene_depth_metadata = load_scene_depth(
                scene_depth_path)
            ep_attrs, _ = config.fallback_episode_attrs(
                str(Path(zarr_dir) / ep))
            K = config.dataset_intrinsics_k(
                ep_attrs or {}, camera="front_1",
                img_shape=scene_depth.shape[1:])
            if K is None:
                print(f"    [{ep}] scene support fallback: camera intrinsics unavailable")
            else:
                support_surface = estimate_scene_support_surface(
                    scene_depth, head, K, np.concatenate([left_p, right_p]),
                    spec.reach)
                if support_surface is None:
                    print(f"    [{ep}] scene support fallback: no reliable horizontal plane")
                else:
                    gravity_up = np.array([
                        0.0, 0.0, -support_surface["down_sign"]
                    ])
                    _, nominal_R_base = base_frame(fwd, gravity_up)
                    support_surface["base_min_z"] = _base_visual_min_z(
                        single, spec.base_body)
                    support_surface["base_up_z"] = nominal_R_base[2, 2]
                    plane = support_surface["coefficients"]
                    print(
                        f"    [{ep}] scene support: z={plane[0]:+.4f}x "
                        f"{plane[1]:+.4f}y {plane[2]:+.4f}, "
                        f"rmse={support_surface['rmse']:.3f}m "
                        f"confidence={support_surface['confidence']:.2f} "
                        f"points={support_surface['inlier_count']}, "
                        f"base_min_z={support_surface['base_min_z']:.4f}m")
    shared_base_height = (float(np.mean(np.concatenate([
        left_p[:, 2], right_p[:, 2]]))) if spec.coplanar_bases else None)
    shared_base_forward = (float(np.mean(np.concatenate([
        left_p @ fwd, right_p @ fwd]))) if spec.aligned_base_depths else None)

    left_candidates, left_kf_idx = search_base_pose(
        single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
        left_p, left_R, nominal_R_base, fwd, sign=-1.0,
        reach=spec.reach, camera_pos=camera_pos,
        search_mode=base_search_mode, ik_solver=ik_solver,
        mink_context=mink_context,
        mink_position_context=mink_position_context,
        base_orientation_mode=spec.base_orientation_mode,
        enable_base_orientation_search=enable_base_orientation_search,
        base_height_anchor=shared_base_height,
        base_forward_anchor=shared_base_forward,
        base_forward_offsets=spec.base_forward_offsets,
        max_target_distance=spec.base_max_target_distance,
        support_surface=support_surface)
    right_candidates, right_kf_idx = search_base_pose(
        single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
        right_p, right_R, nominal_R_base, fwd, sign=+1.0,
        reach=spec.reach, camera_pos=camera_pos,
        search_mode=base_search_mode, ik_solver=ik_solver,
        mink_context=mink_context,
        mink_position_context=mink_position_context,
        base_orientation_mode=spec.base_orientation_mode,
        enable_base_orientation_search=enable_base_orientation_search,
        base_height_anchor=shared_base_height,
        base_forward_anchor=shared_base_forward,
        base_forward_offsets=spec.base_forward_offsets,
        max_target_distance=spec.base_max_target_distance,
        support_surface=support_surface)
    left_choice, right_choice = select_joint_base_pair(
        left_candidates, right_candidates,
        single=single, arm_qadr=arm_qadr, arm_vadr=arm_vadr,
        arm_ids=arm_ids, ee_ref=ee_ref_single,
        left_targets_p=left_p, left_targets_R=left_R,
        right_targets_p=right_p, right_targets_R=right_R,
        left_keyframes=left_kf_idx, right_keyframes=right_kf_idx,
        spec=spec, mink_context=mink_context,
        mink_position_context=mink_position_context,
        ik_solver=(solve_arm_ik if use_mink_ik else solve_arm_ik_dls),
        use_robust_ik=use_mink_ik,
        enable_collision_screen=base_search_mode != "original",
        base_alignment_axis=fwd)
    left_base_pos, left_R_base = left_choice["base_pos"], left_choice["base_R"]
    right_base_pos, right_R_base = right_choice["base_pos"], right_choice["base_R"]
    left_base_quat, right_base_quat = left_choice["base_quat"], right_choice["base_quat"]
    print(f"  [{ep}] base_search({base_search_mode}): "
          f"left feas={left_choice['feasibility_rate']:.0%} "
          f"reach={left_choice['reach_ratio']:.2f} score={left_choice['score']:.2f} "
          f"euler={np.round(left_choice['orientation'],1)} at {np.round(left_base_pos,3)} | "
          f"right feas={right_choice['feasibility_rate']:.0%} "
          f"reach={right_choice['reach_ratio']:.2f} score={right_choice['score']:.2f} "
          f"euler={np.round(right_choice['orientation'],1)} at {np.round(right_base_pos,3)}")

    # Optional visual calibration: move both mounts away from the ego camera
    # along the opposite horizontal viewing direction.  The target TCP poses
    # stay fixed, so the subsequent trajectory IK is re-solved against the
    # adjusted bases.  Keeping this after base search preserves the search
    # result when the option is disabled (the default).
    if base_pullback > 0.0:
        pull_dir = -np.asarray(fwd, dtype=float)
        pull_dir[2] = 0.0
        pull_norm = np.linalg.norm(pull_dir)
        if pull_norm < 1e-8:
            raise ValueError("cannot apply base_pullback: head viewing direction has no horizontal component")
        pull = pull_dir / pull_norm * base_pullback
        head_pts = head[:, :3]
        before = min(
            np.linalg.norm(head_pts - left_base_pos, axis=1).min(),
            np.linalg.norm(head_pts - right_base_pos, axis=1).min(),
        )
        left_base_pos = left_base_pos + pull
        right_base_pos = right_base_pos + pull
        if support_surface is not None:
            left_base_pos, left_support_gap = _snap_base_to_support(
                left_base_pos, support_surface)
            right_base_pos, right_support_gap = _snap_base_to_support(
                right_base_pos, support_surface)
            max_gap = max(left_support_gap, right_support_gap)
            if max_gap > support_surface["max_local_gap"]:
                print(f"    WARNING: adjusted base is {max_gap:.3f}m from "
                      "the nearest observed support point; using plane extrapolation")
        after = min(
            np.linalg.norm(head_pts - left_base_pos, axis=1).min(),
            np.linalg.norm(head_pts - right_base_pos, axis=1).min(),
        )
        print(f"    base_pullback={base_pullback:.3f}m along {pull_dir / pull_norm}: "
              f"min head/base distance {before:.3f} -> {after:.3f}m")

    # 3. Build the dual-arm wrapper, deriving fovy from fy. None prefers dataset
    # intrinsics and supports both the legacy front_1 matrix and flat {fl_y, h} format.
    if fy is None:
        if ep_attrs is None:
            ep_attrs, _ = config.fallback_episode_attrs(str(Path(zarr_dir) / ep))
        inferred_fy = config.dataset_fy_for_height(ep_attrs or {}, height)
        fy = inferred_fy if inferred_fy is not None else 490.1961
    fy = float(fy)
    if not np.isfinite(fy) or fy <= 0.0:
        raise ValueError(f"fy must be a finite positive focal length, got {fy}")
    fovy = float(np.degrees(2.0 * np.arctan((height / 2.0) / fy)))
    dual = build_dual_model(left_base_pos, left_base_quat,
                            right_base_pos, right_base_quat, fovy, spec)
    cam_id = mujoco.mj_name2id(dual, mujoco.mjtObj.mjOBJ_CAMERA, "ego")

    # DOF indices in the dual-arm model (left_/right_ prefixes).
    l_ids, l_qadr, l_vadr = name_to_dof(dual, spec.arm_joints, "left_")
    r_ids, r_qadr, r_vadr = name_to_dof(dual, spec.arm_joints, "right_")
    _, lf_qadr, _ = name_to_dof(dual, spec.finger_joints, "left_")
    _, rf_qadr, _ = name_to_dof(dual, spec.finger_joints, "right_")
    all_finger_joints = spec.gripper_all_joints or spec.finger_joints
    _, lf_all_qadr, _ = name_to_dof(dual, all_finger_joints, "left_")
    _, rf_all_qadr, _ = name_to_dof(dual, all_finger_joints, "right_")

    # 4. Full-episode 6-DOF IK (world targets -> each arm's base frame).
    #    R_target_base = R_base.T @ R_target_world, independently per arm.
    qpos_all = np.zeros((N, dual.nq))
    # Keep position reachability separate from full-pose success. A frame can
    # place the pinch center accurately while missing the requested wrist
    # orientation; such a frame must not be replaced by joint interpolation.
    ik_position_ok = np.zeros((N, 2), dtype=bool)
    ik_pose_ok = np.zeros((N, 2), dtype=bool)
    ik_err_pos = np.zeros((N, 2))
    ik_err_rot = np.zeros((N, 2))
    ik_position_fallback = np.zeros((N, 2), dtype=bool)
    ik_branch_jump_avoided = np.zeros((N, 2), dtype=bool)
    ik_dls_fallback = np.zeros((N, 2), dtype=bool)
    ik_pose_refined = np.zeros((N, 2), dtype=bool)
    # Keep separate (N, 7) arm-joint sequences for repair_branch_jumps post-processing.
    Q_l = np.zeros((N, len(arm_qadr)))
    Q_r = np.zeros((N, len(arm_qadr)))

    # Warm-start single-arm IK frame by frame to avoid jitter from switching IK branches.
    if not use_mink_ik:
        warm_l = _prewarm_dls(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            left_R_base.T @ (left_p[0] - left_base_pos))
        warm_r = _prewarm_dls(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            right_R_base.T @ (right_p[0] - right_base_pos))
    else:
        # Let the first full-pose solve start without a hard continuity bound.
        # A position-only prewarm can be on a wrist branch that is over 0.35 rad
        # from any orientation-compliant solution; constraining that first frame
        # makes the entire trajectory report IK failure despite tiny position
        # residuals. Subsequent frames are warm-started continuously below.
        warm_l = None
        warm_r = None

    def solve_trajectory_ik(target_pos, target_R, warm):
        if not use_mink_ik:
            return (*solve_arm_ik_dls_robust(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
                target_pos, target_R, q_warm=warm,
                continuity_max_step=mink_continuity_max_step),
                    False, False, False, False)

        # Follow EgoDex's continuous control loop before trying any restart:
        # start from the previous configuration, keep orientation as a soft
        # objective, and stop once the pinch-center position is reached.
        continuous = solve_arm_ik_position_first(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            target_pos, target_R, q_init=warm, max_iter=500,
            tol_pos=POSITION_FIRST_TOL, tol_rot=BASE_FEAS_ROT_TOL,
            mink_context=mink_context, continuity_q=warm,
            continuity_max_step=mink_continuity_max_step)
        if continuous[2] < BASE_FEAS_POS_TOL:
            refined = refine_arm_ik_pose(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
                target_pos, target_R, continuous, previous_q=warm,
                max_iter=POSE_REFINEMENT_MAX_ITER,
                tol_pos=POSITION_FIRST_TOL, tol_rot=BASE_FEAS_ROT_TOL,
                mink_context=mink_context,
                continuity_max_step=mink_continuity_max_step)
            q, pose_ok, err_pos, err_rot, pose_refined = refined
            return (q, pose_ok, err_pos, err_rot, False, False, False,
                    pose_refined)

        def continuous_with_warm(q):
            if warm is None or mink_continuity_max_step <= 0.0:
                return True
            return bool(np.max(np.abs(
                q[arm_qadr] - warm[arm_qadr])) <=
                mink_continuity_max_step + 1e-6)

        # Position is genuinely unresolved on the current branch. Only now
        # allow the robust full-pose solver and DLS to act as recovery paths.
        mink_result = solve_arm_ik_robust(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            target_pos, target_R, warm, mink_context=mink_context,
            mink_position_context=mink_position_context,
            continuity_max_step=mink_continuity_max_step,
            max_iter=300)
        if (mink_result[2] < BASE_FEAS_POS_TOL and
                continuous_with_warm(mink_result[0])):
            return (*mink_result, not mink_result[1], False, False, False)

        # Mink's hard continuity limit can preserve a position-reachable but
        # orientation-infeasible branch for a long run. Recover that frame
        # with the existing damped-least-squares solver, which evaluates the
        # same TCP site and enforces the same joint limits. Prefer the DLS
        # candidate when it is compliant, or when it materially improves the
        # failed Mink residuals; later frames remain warm-started from it.
        dls_result = solve_arm_ik_dls_robust(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            target_pos, target_R, q_warm=warm,
            max_iter=DLS_MAX_ITER,
            continuity_max_step=mink_continuity_max_step)
        if (dls_result[2] < BASE_FEAS_POS_TOL and
                continuous_with_warm(dls_result[0])):
            return (*dls_result, not dls_result[1], False, True, False)

        # Never promote a distant recovery branch into the next frame's warm
        # start. The continuous result obeys the trust region and is therefore
        # the least damaging failed-position state for later interpolation.
        return (*continuous, False, False, False, False)

    for t in range(N):
        # left
        p_base_l = left_R_base.T @ (left_p[t] - left_base_pos)
        R_base_l = left_R_base.T @ left_R[t]
        # Solve with the single-arm model, then write results to the matching dual-model qadr.
        q_l, okl, epl, erl, fallback_l, branch_l, dls_l, refined_l = \
            solve_trajectory_ik(p_base_l, R_base_l, warm_l)
        # right
        p_base_r = right_R_base.T @ (right_p[t] - right_base_pos)
        R_base_r = right_R_base.T @ right_R[t]
        q_r, okr, epr, err, fallback_r, branch_r, dls_r, refined_r = \
            solve_trajectory_ik(p_base_r, R_base_r, warm_r)
        warm_l = q_l.copy()
        warm_r = q_r.copy()

        # Store arm-joint sequences using qadr values from the single-arm model.
        Q_l[t] = [q_l[qa] for qa in arm_qadr]
        Q_r[t] = [q_r[qa] for qa in arm_qadr]

        # Write back to the dual-arm model.
        for i, qa in enumerate(l_qadr):
            qpos_all[t, qa] = q_l[arm_qadr[i]]
        for i, qa in enumerate(r_qadr):
            qpos_all[t, qa] = q_r[arm_qadr[i]]
        set_gripper_qpos(qpos_all[t], lf_all_qadr, spec, left_w[t])
        set_gripper_qpos(qpos_all[t], rf_all_qadr, spec, right_w[t])
        ik_err_pos[t] = [epl, epr]
        ik_err_rot[t] = [erl, err]
        ik_position_ok[t] = ik_err_pos[t] < BASE_FEAS_POS_TOL
        ik_pose_ok[t] = [okl, okr]
        ik_position_fallback[t] = [fallback_l, fallback_r]
        ik_branch_jump_avoided[t] = [branch_l, branch_r]
        ik_dls_fallback[t] = [dls_l, dls_r]
        ik_pose_refined[t] = [refined_l, refined_r]

    both_position_ok = ik_position_ok.all(axis=1).mean()
    both_pose_ok = ik_pose_ok.all(axis=1).mean()
    ik_label = "DLS" if not use_mink_ik else "Mink"
    ik_task_dim = (6 if not use_mink_ik else
                   3 + int(np.count_nonzero(spec.ik_orientation_cost)))
    print(f"    IK {ik_label} {ik_task_dim}D task: "
          f"both_position_ok={both_position_ok:.1%} "
          f"both_pose_ok={both_pose_ok:.1%} "
          f"pos_err L={ik_err_pos[:,0].mean()*1000:.1f}mm R={ik_err_pos[:,1].mean()*1000:.1f}mm "
          f"rot_err L={np.degrees(ik_err_rot[:,0].mean()):.1f}° R={np.degrees(ik_err_rot[:,1].mean()):.1f}°")
    if np.any(ik_position_fallback):
        print("    position-priority fallback: "
              f"L={np.count_nonzero(ik_position_fallback[:, 0])} "
              f"R={np.count_nonzero(ik_position_fallback[:, 1])} frames, "
              "branch jumps avoided "
              f"L={np.count_nonzero(ik_branch_jump_avoided[:, 0])} "
              f"R={np.count_nonzero(ik_branch_jump_avoided[:, 1])}")
    if np.any(ik_dls_fallback):
        print("    DLS fallback after Mink failure: "
              f"L={np.count_nonzero(ik_dls_fallback[:, 0])} "
              f"R={np.count_nonzero(ik_dls_fallback[:, 1])}")
    if np.any(ik_pose_refined):
        print("    pose refinement accepted: "
              f"L={np.count_nonzero(ik_pose_refined[:, 0])} "
              f"R={np.count_nonzero(ik_pose_refined[:, 1])}")

    # 4b. Repair branch jumps by re-solving jump frames from neighboring configurations.
    n_fix_l, pose_ok_l, position_ok_l = repair_branch_jumps(
        single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
        Q_l, left_p, left_R, left_base_pos, left_R_base, ik_pose_ok[:, 0],
        position_ok_flags=ik_position_ok[:, 0],
        mink_context=mink_context,
        ik_solver=(solve_arm_ik_position_first
                   if use_mink_ik else solve_arm_ik_dls))
    n_fix_r, pose_ok_r, position_ok_r = repair_branch_jumps(
        single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
        Q_r, right_p, right_R, right_base_pos, right_R_base, ik_pose_ok[:, 1],
        position_ok_flags=ik_position_ok[:, 1],
        mink_context=mink_context,
        ik_solver=(solve_arm_ik_position_first
                   if use_mink_ik else solve_arm_ik_dls))
    if n_fix_l + n_fix_r > 0:
        # Write repaired joints back to qpos_all.
        for t in range(N):
            for i, qa in enumerate(l_qadr):
                qpos_all[t, qa] = Q_l[t, i]
            for i, qa in enumerate(r_qadr):
                qpos_all[t, qa] = Q_r[t, i]
        ik_pose_ok[:, 0] = pose_ok_l
        ik_pose_ok[:, 1] = pose_ok_r
        ik_position_ok[:, 0] = position_ok_l
        ik_position_ok[:, 1] = position_ok_r
        both_pose_ok2 = ik_pose_ok.all(axis=1).mean()
        print(f"    branch-repair: fixed L={n_fix_l} R={n_fix_r} "
              f"frames → both_pose_ok={both_pose_ok2:.1%}")

    # The final robust IK can choose a different valid branch than the
    # sparse base-pair screen.  Validate the actual rendered qpos and retry
    # only intersecting frames before writing the video.
    if use_mink_ik:
        _collision_branch_repair(
            dual, qpos_all, Q_l, Q_r, ik_pose_ok, ik_err_pos, ik_err_rot,
            single, arm_qadr, arm_ids, l_qadr, r_qadr,
            left_p, left_R, right_p, right_R,
            left_base_pos, left_R_base, right_base_pos, right_R_base,
            ee_ref_single, mink_context, mink_position_context)
        # Collision repair may replace either arm with a new IK candidate.
        # Refresh position success from the candidate residuals it recorded;
        # the repair function updates full-pose success directly.
        ik_position_ok[:] = ik_err_pos < BASE_FEAS_POS_TOL

    # Replace only position-unreachable frames. Orientation-only failures keep
    # their accurately aligned pinch-center solution instead of being replaced
    # by joint interpolation that no longer tracks the current hand position.
    n_interpolated = interp_failed_frames(
        qpos_all, ik_position_ok, l_qadr, r_qadr,
        branch_jump_thresh=(spec.ik_branch_jump_threshold
                            if spec.ik_position_priority else 0.0))
    n_clamped_left = clamp_joint_speed(qpos_all, l_qadr, max_jump_rad)
    n_clamped_right = clamp_joint_speed(qpos_all, r_qadr, max_jump_rad)
    if n_interpolated or n_clamped_left or n_clamped_right:
        print(f"    post-ik: interp position-failed={n_interpolated} frames, "
              f"speed-clamp L={n_clamped_left} R={n_clamped_right}")

    # 5. Render and composite.
    bg_frames = read_video_frames(bg_path)
    if not bg_frames:
        print("    [SKIP] no bg frames")
        return None
    Nrender = min(N, len(bg_frames))
    H, W = bg_frames[0].shape[:2]
    if depth_mode == "depth-aware":
        if scene_depth_path is None:
            raise ValueError(f"depth-aware mode requires scene depth for {ep}")
        if scene_depth is None:
            scene_depth, scene_depth_metadata = load_scene_depth(
                scene_depth_path)
        saved_h = scene_depth_metadata["height"]
        saved_w = scene_depth_metadata["width"]
        crop_top = scene_depth_metadata["crop_top"]
        crop_bottom = scene_depth_metadata["crop_bottom"]
        # Depth Anything consumes the valid image area; align encoded video padding.
        if (H, W) != (saved_h, saved_w) and H == saved_h + crop_top + crop_bottom and W == saved_w:
            bg_frames = [f[crop_top:H - crop_bottom, :saved_w] for f in bg_frames]
            H, W = bg_frames[0].shape[:2]
        if scene_depth.ndim != 3 or scene_depth.shape[1:] != (H, W):
            raise ValueError(f"depth shape {scene_depth.shape} does not match video {(H, W)}")
        if (saved_h, saved_w) != (H, W) or scene_depth.shape[0] < Nrender:
            raise ValueError(f"invalid scene depth metadata/frames for {ep}")
    hide_non_arm_geoms(dual, spec)
    # MuJoCo's offscreen renderer has no MSAA.  The legacy pipeline rendered
    # at 2x and reduced with area filtering to suppress edge noise.
    SS = 2
    renderer = make_renderer(dual, W * SS, H * SS)
    data = mujoco.MjData(dual)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_final = str(out_dir / f"{ep}_robot_on_bg.mp4")
    w_final = cv2.VideoWriter(out_final, fourcc, fps, (W, H))
    out_mask = str(out_dir / f"{ep}_arm_mask.mp4")
    w_mask = cv2.VideoWriter(out_mask, fourcc, fps, (W, H))
    wrist_cam_ids = {
        name: mujoco.mj_name2id(dual, mujoco.mjtObj.mjOBJ_CAMERA, name)
        for name in ("wrist_l", "wrist_r")
    }
    w_wrist = {
        name: cv2.VideoWriter(
            str(out_dir / f"{ep}_{name}.mp4"), fourcc, fps, (W, H))
        for name in wrist_cam_ids
    }
    video_writers = {"robot_on_bg": w_final, "arm_mask": w_mask, **w_wrist}
    failed_writers = [name for name, writer in video_writers.items()
                      if not writer.isOpened()]
    if failed_writers:
        for writer in video_writers.values():
            writer.release()
        raise RuntimeError(f"failed to open video writer(s): {failed_writers}")
    missing_cameras = [name for name, cam in wrist_cam_ids.items() if cam < 0]
    if missing_cameras:
        for writer in video_writers.values():
            writer.release()
        raise ValueError(f"wrist camera(s) not found: {missing_cameras}")
    w_debug = None
    if debug:
        w_debug = cv2.VideoWriter(str(out_dir / f"{ep}_debug4.mp4"),
                                  fourcc, fps, (W * 2, H))

    rendered = np.zeros(N, dtype=bool)
    robot_pixel_count = np.zeros(N, dtype=np.int64)
    robot_mask_ratio = np.zeros(N, dtype=np.float32)
    self_collision = np.zeros(N, dtype=bool)
    self_contact_count = np.zeros(N, dtype=np.int32)
    self_penetration = np.zeros(N, dtype=np.float32)
    cross_arm_contact_count = np.zeros(N, dtype=np.int32)
    cross_arm_penetration = np.zeros(N, dtype=np.float32)

    for t in range(Nrender):
        data.qpos[:] = qpos_all[t]
        set_ego_camera(data, head[t])
        # First update the robot frames from qpos, then place the mocap wrist
        # cameras. render_rgb_and_mask performs the second forward pass that
        # propagates those mocap poses to the cameras.
        mujoco.mj_forward(dual, data)
        set_wrist_cameras(data, dual, spec, (left_w[t], right_w[t]))
        robot_rgb, mask, robot_depth = render_rgb_and_mask(
            dual, data, renderer, cam_id, with_depth=scene_depth is not None)
        collision = _quality_collision_metrics(dual, data)
        rendered[t] = True
        robot_pixel_count[t] = int(mask.sum())
        robot_mask_ratio[t] = float(mask.mean())
        self_collision[t] = collision["self_collision"]
        self_contact_count[t] = collision["self_contact_count"]
        self_penetration[t] = collision["self_penetration"]
        cross_arm_contact_count[t] = collision["cross_arm_contact_count"]
        cross_arm_penetration[t] = collision["cross_arm_penetration"]
        # Keep the legacy robot-image cleanup order.  Apply it before the
        # 2x-to-output reduction so the result remains visually compatible.
        robot_rgb = cv2.medianBlur(robot_rgb, 3)
        robot_rgb = tone_map_robot(robot_rgb)
        robot_rgb = patch_arm_holes(robot_rgb, mask)
        if robot_rgb.shape[:2] != (H, W):
            robot_rgb = cv2.resize(robot_rgb, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.float32), (W, H),
                              interpolation=cv2.INTER_LINEAR) > 0.5
        # Closing removes isolated one-pixel z-fight holes in the instance
        # mask, matching the old compositing path.
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8), iterations=1) > 0
        if robot_depth is not None and robot_depth.shape[:2] != (H, W):
            robot_depth = cv2.resize(robot_depth, (W, H),
                                     interpolation=cv2.INTER_NEAREST)
        robot_bgr = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)
        m = feather_mask(mask, feather)
        if scene_depth is not None:
            visible = mask & np.isfinite(robot_depth) & (robot_depth > 0)
            visible &= robot_depth < (scene_depth[t] - depth_epsilon)
            m *= feather_mask(visible, feather)
        composed = (m * robot_bgr + (1.0 - m) * bg_frames[t]).astype(np.uint8)
        w_final.write(composed)
        mask_u8 = mask.astype(np.uint8) * 255
        w_mask.write(np.repeat(mask_u8[:, :, None], 3, axis=2))
        for name, wrist_cam_id in wrist_cam_ids.items():
            renderer.disable_segmentation_rendering()
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=wrist_cam_id)
            wrist_rgb = renderer.render().copy()
            if wrist_rgb.shape[:2] != (H, W):
                wrist_rgb = cv2.resize(
                    wrist_rgb, (W, H), interpolation=cv2.INTER_AREA)
            wrist_rgb = cv2.medianBlur(wrist_rgb, 3)
            w_wrist[name].write(cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
        if w_debug is not None:
            panel = np.hstack([bg_frames[t], composed])
            cv2.putText(panel, "INPAINT-BG", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.putText(panel, f"{spec.name.upper()}-DUAL", (W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            w_debug.write(panel)
        if (t + 1) % 100 == 0:
            print(f"      rendered {t+1}/{Nrender}")

    w_final.release()
    w_mask.release()
    for writer in w_wrist.values():
        writer.release()
    if w_debug:
        w_debug.release()

    # OpenCV's mp4v output is not consistently playable in browsers and
    # VSCode.  Preserve the legacy final H.264 conversion for every video.
    config.reencode_h264(out_final)
    config.reencode_h264(str(out_dir / f"{ep}_arm_mask.mp4"))
    config.reencode_h264(str(out_dir / f"{ep}_wrist_l.mp4"))
    config.reencode_h264(str(out_dir / f"{ep}_wrist_r.mp4"))
    if w_debug:
        config.reencode_h264(str(out_dir / f"{ep}_debug4.mp4"))

    state_qpos = np.concatenate([
        qpos_all[:, l_qadr], qpos_all[:, lf_qadr],
        qpos_all[:, r_qadr], qpos_all[:, rf_qadr],
    ], axis=1).astype(np.float32)
    support_metadata = {}
    if spec.scene_support_surface:
        support_metadata = {
            "scene_support_applied": np.asarray(support_surface is not None),
            "scene_support_plane": np.asarray(
                support_surface["coefficients"]
                if support_surface is not None else
                [np.nan, np.nan, np.nan], dtype=np.float32),
            "scene_support_rmse": np.asarray(
                support_surface["rmse"]
                if support_surface is not None else np.nan,
                dtype=np.float32),
            "base_visual_min_z": np.asarray(
                support_surface["base_min_z"]
                if support_surface is not None else np.nan,
                dtype=np.float32),
        }
    np.savez_compressed(str(out_dir / f"{ep}_ik.npz"),
                        qpos=qpos_all, state=state_qpos,
                        robot_type=np.asarray(spec.name),
                        state_names=np.asarray([
                            f"left_{n}" for n in spec.state_names
                        ] + [f"right_{n}" for n in spec.state_names]),
                        # ``ik_ok`` remains a compatibility alias for callers
                        # that historically interpreted it as full-pose IK.
                        ik_ok=ik_pose_ok,
                        ik_position_ok=ik_position_ok,
                        ik_pose_ok=ik_pose_ok,
                        ik_err_pos=ik_err_pos, ik_err_rot=ik_err_rot,
                        ik_position_fallback=ik_position_fallback,
                        ik_branch_jump_avoided=ik_branch_jump_avoided,
                        ik_dls_fallback=ik_dls_fallback,
                        ik_pose_refined=ik_pose_refined,
                        left_base_pos=left_base_pos, right_base_pos=right_base_pos,
                        base_quat=left_base_quat,
                        left_base_quat=left_base_quat,
                        right_base_quat=right_base_quat,
                        left_base_search_keyframes=left_kf_idx,
                        right_base_search_keyframes=right_kf_idx,
                        base_pullback=np.asarray(base_pullback, dtype=np.float32),
                        base_search_mode=np.asarray(base_search_mode),
                        ik_solver=np.asarray(ik_solver),
                        ik_orientation_cost=np.asarray(
                            spec.ik_orientation_cost, dtype=np.float32),
                        ik_position_priority=np.asarray(
                            spec.ik_position_priority),
                        ik_branch_jump_threshold=np.asarray(
                            spec.ik_branch_jump_threshold, dtype=np.float32),
                        max_jump_rad=np.asarray(max_jump_rad, dtype=np.float32),
                        mink_continuity_max_step=np.asarray(
                            mink_continuity_max_step, dtype=np.float32),
                        mink_continuity_cost=np.asarray(
                            mink_continuity_cost, dtype=np.float32),
                        target_smooth_window=np.asarray(
                            target_smooth_window, dtype=np.int32),
                        target_smooth_polyorder=np.asarray(
                            target_smooth_polyorder, dtype=np.int32),
                        target_orientation_sigma=np.asarray(
                            target_orientation_sigma, dtype=np.float32),
                        **support_metadata)
    np.savez_compressed(str(out_dir / f"{ep}_quality.npz"),
                        hand_detected=hand_detected,
                        rendered=rendered,
                        robot_pixel_count=robot_pixel_count,
                        robot_mask_ratio=robot_mask_ratio,
                        self_collision=self_collision,
                        self_contact_count=self_contact_count,
                        self_penetration=self_penetration,
                        cross_arm_contact_count=cross_arm_contact_count,
                        cross_arm_penetration=cross_arm_penetration,
                        fps=np.asarray(float(fps)),
                        robot_type=np.asarray(spec.name))
    print(f"    ✓ {out_final}")
    return out_final
