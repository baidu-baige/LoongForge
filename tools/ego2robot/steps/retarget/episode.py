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
    depth_visibility_alpha,
    feather_mask,
    gripper_geom_ids,
    hide_non_arm_geoms,
    load_scene_depth,
    mat_to_quat,
    make_renderer,
    name_to_dof,
    patch_arm_holes,
    read_video_frames,
    render_rgb_and_mask,
    render_geom_subset_mask,
    set_ego_camera,
    set_gripper_qpos,
    set_wrist_cameras,
    temporal_median_depth_frame,
    tone_map_robot,
)
from .targets import (
    _fill_invalid_keypoints,
    _quat_to_mat,
    eq12_pose_world,
    opening_projection_plane_normals,
    projection_plane_directions,
    smooth_retarget_targets,
    target_axis_projection_plane_normals,
    target_ref_pose,
)
from .trajectory_refinement import (
    TrajectoryRefinementConfig,
    evaluate_arm_trajectory,
    refine_arm_trajectory,
    refine_dual_arm_trajectory,
)


# Occlusion tolerance gain.  The DA3->world depth alignment carries a residual
# uncertainty (median absolute residual of the hand-keypoint fit); the depth
# occlusion test is only meaningful beyond that uncertainty.  We widen the
# occlusion margin by this multiple of the residual so robot pixels that fall
# within alignment noise (e.g. an arm grazing the scene plane) are not culled.
DEPTH_ALIGN_TOL_GAIN = 1.5


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


def load_episode_keypoints_2d(zarr_dir, ep, n_frames):
    """Read optional full-image hand observations used by Path B."""
    z = zarr.open_group(str(Path(zarr_dir) / ep), mode="r")
    result = []
    for side in ("left", "right"):
        name = f"{side}.obs_keypoints_2d"
        result.append(
            np.array(z[name][:n_frames]).reshape(n_frames, 21, 2)
            if name in z else None)
    return tuple(result)


def load_human_hand_masks(mask_dir, ep, n_frames, dilation_px=8):
    """Load and dilate the original human hand/arm mask for compositing."""
    if mask_dir is None:
        return None
    path = Path(mask_dir) / ep / "hand_arm_masks.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"depth-aware gripper compositing requires human hand masks: {path}")
    with np.load(path, allow_pickle=False) as archive:
        masks = np.asarray(archive["masks"][:n_frames], dtype=np.uint8)
    if masks.ndim != 3 or len(masks) < n_frames:
        raise ValueError(
            f"invalid human hand masks for {ep}: {masks.shape}, need {n_frames} frames")
    dilation_px = int(dilation_px)
    if dilation_px < 0:
        raise ValueError("human_hand_mask_dilation must be non-negative")
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_px + 1, 2 * dilation_px + 1))
        masks = np.stack([
            cv2.dilate(mask, kernel) for mask in masks
        ])
    return masks > 0


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


# ============ DA3 scene-depth alignment to EgoVerse world scale ============

def _project_hand_depth_samples(scene_depth, head, hands, K):
    """Collect (da3_depth, world_depth) pairs at projected MANO keypoints.

    The scene depth from Depth Anything 3 is monocular metric depth in its own
    scale, while the MuJoCo robot depth is rendered in the EgoVerse world scale.
    The hand keypoints are known in the world frame, so projecting them into the
    ego camera and reading DA3 at those pixels gives per-pixel correspondences
    between the two depth scales without any extra sensing.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if not (np.isfinite([fx, fy, cx, cy]).all() and fx > 0 and fy > 0):
        return np.empty(0), np.empty(0)
    frames, height, width = scene_depth.shape
    da3_samples = []
    world_samples = []
    for kp_seq, valid in hands:
        kp_seq = np.asarray(kp_seq, dtype=float).reshape(len(kp_seq), 21, 3)
        valid = np.asarray(valid, dtype=bool)
        for t in np.flatnonzero(valid):
            if t >= frames:
                break
            R = _quat_to_mat(head[t, 3:7])          # camera-to-world
            cam = (kp_seq[t] - head[t, :3]) @ R      # world->camera
            z = cam[:, 2]
            good = np.isfinite(cam).all(axis=1) & (z > 1e-3)
            if not np.any(good):
                continue
            u = np.rint(fx * cam[good, 0] / z[good] + cx).astype(int)
            v = np.rint(fy * cam[good, 1] / z[good] + cy).astype(int)
            in_view = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not np.any(in_view):
                continue
            da3 = scene_depth[t, v[in_view], u[in_view]].astype(float)
            wz = z[good][in_view]
            finite = np.isfinite(da3) & (da3 > 1e-3) & np.isfinite(wz)
            if np.any(finite):
                da3_samples.append(da3[finite])
                world_samples.append(wz[finite])
    if not da3_samples:
        return np.empty(0), np.empty(0)
    return np.concatenate(da3_samples), np.concatenate(world_samples)


def _estimate_scene_depth_alignment(scene_depth, head, hands, K,
                                    min_pairs=100, ratio_quantile=0.7,
                                    scale_bounds=(0.2, 5.0)):
    """Robustly fit ``world ~= scale * da3 + shift`` from hand correspondences.

    Hand keypoints tend to cluster in a narrow depth band, so a full affine fit
    is ill-conditioned; scale-only is the well-conditioned default. Because the
    hand sits at or in front of the inpainted surface, the median world/da3
    ratio is biased low by non-contact (floating) frames, so a high quantile of
    the ratio recovers the contact scale. An affine refit is only accepted when
    the sampled depth has real spread and it lowers the robust residual.
    """
    stats = {"mode": "identity", "n_pairs": 0}
    if K is None:
        stats["reason"] = "no intrinsics"
        return 1.0, 0.0, stats
    da3, world = _project_hand_depth_samples(scene_depth, head, hands, K)
    stats["n_pairs"] = int(da3.size)
    if da3.size < min_pairs:
        stats["reason"] = f"only {da3.size} pairs (<{min_pairs})"
        return 1.0, 0.0, stats

    ratio = world / da3
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < min_pairs:
        stats["reason"] = f"only {ratio.size} valid ratios"
        return 1.0, 0.0, stats
    lo, hi = float(scale_bounds[0]), float(scale_bounds[1])
    scale = float(np.clip(np.quantile(ratio, ratio_quantile), lo, hi))
    shift = 0.0
    mode = "scale"
    resid_scale = float(np.median(np.abs(world - scale * da3)))

    # Affine refit only when depth diversity is sufficient to identify a shift.
    da3_med = float(np.median(da3))
    spread = float(np.quantile(da3, 0.9) - np.quantile(da3, 0.1))
    if spread > max(0.25, 0.35 * da3_med):
        try:
            from scipy.stats import theilslopes
            # Bias the fit toward contact frames (nearer/at the surface).
            contact = world >= np.median(world)
            if np.count_nonzero(contact) >= min_pairs // 2:
                a, b, _, _ = theilslopes(world[contact], da3[contact])
            else:
                a, b, _, _ = theilslopes(world, da3)
            a = float(a)
            b = float(b)
            resid_affine = float(np.median(np.abs(world - (a * da3 + b))))
            if (lo <= a <= hi and np.isfinite(b) and
                    resid_affine < 0.9 * resid_scale):
                scale, shift, mode = a, b, "affine"
                resid_scale = resid_affine
        except Exception as exc:  # pragma: no cover - defensive
            stats["affine_error"] = str(exc)

    stats.update({
        "mode": mode,
        "ratio_median": float(np.median(ratio)),
        "ratio_q": float(np.quantile(ratio, ratio_quantile)),
        "da3_spread": spread,
        "resid": resid_scale,
    })
    return scale, shift, stats


def process_episode(ep, zarr_dir, bg_path, state, out_dir,
                    fps, feather, fy, height, debug,
                    mask_dir=None, human_hand_mask_dilation=8,
                    scene_depth_path=None, depth_epsilon=0.02,
                    depth_temporal_window=5, depth_transition_width=0.04,
                    depth_mode="depth-aware", base_search_mode="balanced",
                    enable_base_orientation_search=False,
                    trajectory_orientation_base_search=False,
                    ik_solver="mink", base_pullback=0.0,
                    max_jump_rad=0.0,
                    mink_continuity_max_step=MINK_CONTINUITY_MAX_STEP,
                    mink_continuity_cost=MINK_CONTINUITY_COST, spec=None,
                    target_smooth_window=11, target_smooth_polyorder=3,
                    target_orientation_sigma=6.0, use_scene_support=True,
                    depth_align=False, trajectory_refine=True,
                    trajectory_refine_position_weight=1.0,
                    trajectory_refine_orientation_weight=0.05,
                    trajectory_refine_velocity_weight=0.2,
                    trajectory_refine_acceleration_weight=1.0,
                    trajectory_refine_home_weight=0.002,
                    trajectory_refine_joint_margin_weight=0.05,
                    trajectory_refine_joint_margin_fraction=0.10,
                    trajectory_refine_collision_weight=4.0,
                    trajectory_refine_collision_safe_distance=0.015,
                    trajectory_refine_position_tolerance=0.005,
                    trajectory_refine_max_nfev=20,
                    primary_trajectory_opt=False,
                    trajectory_refine_primary_max_nfev=100):
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
    trajectory_refine = bool(trajectory_refine)
    # Scope C gray-scale switch: when set (and Mink IK is active on a multi-frame
    # episode) the whole-trajectory optimizer becomes the primary producer of the
    # final trajectory. The frame-wise pass is then only a feasible-branch
    # initializer, and the branch-jump/speed-clamp post-processing that the
    # whole-trajectory cost already absorbs is skipped. Default False keeps the
    # existing frame-wise-primary flow byte-for-byte.
    primary_trajectory_opt = bool(primary_trajectory_opt)
    trajectory_refine_primary_max_nfev = int(trajectory_refine_primary_max_nfev)
    if trajectory_refine_primary_max_nfev < 1:
        raise ValueError("trajectory_refine_primary_max_nfev must be positive")
    primary = primary_trajectory_opt and use_mink_ik and len(state) > 1
    trajectory_refine_config = TrajectoryRefinementConfig(
        position_weight=float(trajectory_refine_position_weight),
        orientation_weight=float(trajectory_refine_orientation_weight),
        velocity_weight=float(trajectory_refine_velocity_weight),
        acceleration_weight=float(trajectory_refine_acceleration_weight),
        home_weight=float(trajectory_refine_home_weight),
        joint_margin_weight=float(trajectory_refine_joint_margin_weight),
        joint_margin_fraction=float(trajectory_refine_joint_margin_fraction),
        collision_weight=float(trajectory_refine_collision_weight),
        collision_safe_distance=float(
            trajectory_refine_collision_safe_distance),
        position_tolerance=float(trajectory_refine_position_tolerance),
        max_nfev=(trajectory_refine_primary_max_nfev if primary
                  else int(trajectory_refine_max_nfev)),
        primary=primary,
        use_analytic_jacobian=primary,
    )
    trajectory_refine_config.validate()
    depth_temporal_window = int(depth_temporal_window)
    if depth_temporal_window < 1 or depth_temporal_window % 2 == 0:
        raise ValueError("depth_temporal_window must be a positive odd integer")
    depth_transition_width = float(depth_transition_width)
    if not np.isfinite(depth_transition_width) or depth_transition_width < 0.0:
        raise ValueError(
            "depth_transition_width must be finite and non-negative")
    N = len(state)
    head, left_kp, right_kp = load_episode_world(zarr_dir, ep, N)
    left_kp2d, right_kp2d = load_episode_keypoints_2d(
        zarr_dir, ep, N)
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

    left_valid = valid_keypoint_frame(left_kp)
    right_valid = valid_keypoint_frame(right_kp)
    has_left, has_right = bool(left_valid.any()), bool(right_valid.any())
    if not (has_left or has_right):
        raise ValueError(f"episode {ep} contains no valid hand keypoints")
    # Quality is per-frame and should report an observed hand when either side
    # is present. Missing sides are intentionally left invalid.
    hand_detected = left_valid | right_valid

    # Resolve the morphology before converting TCP targets.  Body-referenced
    # models receive an explicit TCP site during model construction; using the
    # resolved spec here prevents applying the body-to-TCP offset twice.
    single, spec = build_single_arm_model(spec, return_spec=True)

    # 1. Common human TCP pose -> this morphology's IK reference pose.  The
    # reference implementation smooths Eq.1/Eq.2 outputs before IK; apply the
    # same treatment here so noisy fingertip cross-products cannot drive the
    # redundant arm through alternating IK branches.
    def make_targets(keypoints, keypoints_2d, hand_sign, active):
        if active:
            kp_smooth = _fill_invalid_keypoints(keypoints)
            p, R, w, _ = eq12_pose_world(
                kp_smooth, hand_sign=hand_sign,
                keypoints_2d=keypoints_2d, head_pose=head)
            smoothed_p, smoothed_R, smoothed_w = smooth_retarget_targets(
                p, R, w, window=target_smooth_window,
                polyorder=target_smooth_polyorder,
                orientation_sigma=target_orientation_sigma,
                width_max=spec.gripper_max,
                keypoints_2d=keypoints_2d, hand_sign=hand_sign,
                head_pose=head)
            return smoothed_p, smoothed_R, smoothed_w
        # No target exists for an unobserved hand. Keep a finite neutral target
        # only for shared array plumbing; the IK branch below skips this side.
        p = np.zeros((N, 3), dtype=float)
        R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        return p, R, np.zeros(N, dtype=float)

    left_tcp_p, left_tcp_R, left_w = make_targets(
        left_kp, left_kp2d, -1.0, has_left)
    right_tcp_p, right_tcp_R, right_w = make_targets(
        right_kp, right_kp2d, +1.0, has_right)
    left_p, left_R = target_ref_pose(
        spec, left_tcp_p, left_tcp_R, spec.tcp_rot_site_left,
        opening_width=left_w)
    right_p, right_R = target_ref_pose(
        spec, right_tcp_p, right_tcp_R, spec.tcp_rot_site_right,
        opening_width=right_w)
    left_projection_planes = (
        opening_projection_plane_normals(
            left_p, left_kp2d, -1.0, head_pose=head)
        if has_left and left_kp2d is not None else None)
    right_projection_planes = (
        opening_projection_plane_normals(
            right_p, right_kp2d, +1.0, head_pose=head)
        if has_right and right_kp2d is not None else None)
    use_secondary_projection = bool(
        spec.ik_secondary_projection_axis is not None and
        spec.ik_secondary_projection_cost > 0.0)
    left_secondary_planes = (
        target_axis_projection_plane_normals(
            left_p, left_R, spec.ik_secondary_projection_axis,
            head_pose=head)
        if has_left and left_projection_planes is not None and
        use_secondary_projection else None)
    right_secondary_planes = (
        target_axis_projection_plane_normals(
            right_p, right_R, spec.ik_secondary_projection_axis,
            head_pose=head)
        if has_right and right_projection_planes is not None and
        use_secondary_projection else None)

    def plane_directions(positions, primary, secondary):
        if primary is None:
            return None
        directions = [projection_plane_directions(
            positions, primary, head_pose=head)]
        if secondary is not None:
            directions.append(projection_plane_directions(
                positions, secondary, head_pose=head))
        return np.stack(directions, axis=1)

    def projection_plane_stack(primary, secondary):
        if primary is None:
            return None
        planes = [primary]
        if secondary is not None:
            planes.append(secondary)
        return np.stack(planes, axis=1)

    left_projection_stack = projection_plane_stack(
        left_projection_planes, left_secondary_planes)
    right_projection_stack = projection_plane_stack(
        right_projection_planes, right_secondary_planes)
    left_projection_directions = plane_directions(
        left_p, left_projection_planes, left_secondary_planes)
    right_projection_directions = plane_directions(
        right_p, right_projection_planes, right_secondary_planes)

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
    use_projection_ik = bool(
        use_mink_ik and spec.ik_projection_axis is not None and
        spec.ik_projection_cost > 0.0 and
        (left_projection_planes is not None or
         right_projection_planes is not None))
    mink_projection_context = (MinkIKContext(
        single, arm_ids, ee_ref_single, orientation_cost=0.0,
        projection_axes=([
            spec.ik_projection_axis,
            spec.ik_secondary_projection_axis,
        ] if use_secondary_projection else [spec.ik_projection_axis]),
        projection_costs=([
            spec.ik_projection_cost,
            spec.ik_secondary_projection_cost,
        ] if use_secondary_projection else [spec.ik_projection_cost]),
        continuity_max_step=mink_continuity_max_step,
        # The hard per-frame bound already prevents branch jumps. Keep this
        # soft term below the projection cost so it cannot freeze a visibly
        # incorrect opening direction merely to retain the previous wrist.
        continuity_cost=min(mink_continuity_cost, 0.01))
        if use_projection_ik else None)
    support_surface = None
    if spec.scene_support_surface and not use_scene_support:
        print(f"    [{ep}] scene support disabled: base search ignores the "
              f"support surface (--no_scene_support)")
    elif spec.scene_support_surface:
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
                    scene_depth, head, K,
                    np.concatenate([p for p, active in ((left_p, has_left),
                                                         (right_p, has_right)) if active]),
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
    active_positions = [p for p, active in ((left_p, has_left), (right_p, has_right)) if active]
    shared_base_height = (float(np.mean(np.concatenate([p[:, 2] for p in active_positions])))
                          if spec.coplanar_bases else None)
    shared_base_forward = (float(np.mean(np.concatenate([p @ fwd for p in active_positions])))
                           if spec.aligned_base_depths else None)

    def neutral_base_choice(sign):
        lateral = np.array([-fwd[1], fwd[0], 0.0])
        lateral /= max(np.linalg.norm(lateral), 1e-8)
        base_pos = (camera_pos + sign * 0.4 * lateral * spec.reach
                    - 0.3 * fwd * spec.reach
                    + np.array([0.0, 0.0, 0.15 * spec.reach]))
        if support_surface is not None:
            base_pos, _ = _snap_base_to_support(base_pos, support_surface)
        return {
            "base_pos": base_pos, "base_R": nominal_R_base.copy(),
            "base_quat": mat_to_quat(nominal_R_base),
            "feasibility_rate": 0.0, "reach_ratio": 0.0, "score": 0.0,
            "orientation": (0.0, 0.0, 0.0),
        }

    empty_kf = np.empty(0, dtype=np.int64)
    if has_left:
        left_candidates, left_kf_idx = search_base_pose(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            left_p, left_R, nominal_R_base, fwd, sign=-1.0,
            reach=spec.reach, camera_pos=camera_pos,
            search_mode=base_search_mode, ik_solver=ik_solver,
            mink_context=mink_context, mink_position_context=mink_position_context,
            base_orientation_mode=spec.base_orientation_mode,
            enable_base_orientation_search=enable_base_orientation_search,
            trajectory_orientation_search=trajectory_orientation_base_search,
            base_height_anchor=shared_base_height, base_forward_anchor=shared_base_forward,
            base_forward_offsets=spec.base_forward_offsets,
            max_target_distance=spec.base_max_target_distance,
            support_surface=support_surface,
            mink_projection_context=mink_projection_context,
            projection_planes_world=left_projection_stack,
            projection_directions_world=left_projection_directions)
        left_choice = max(left_candidates, key=lambda x: (x["score"], x["feasibility_rate"]))
    else:
        left_kf_idx, left_choice = empty_kf, neutral_base_choice(-1.0)
    if has_right:
        right_candidates, right_kf_idx = search_base_pose(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            right_p, right_R, nominal_R_base, fwd, sign=+1.0,
            reach=spec.reach, camera_pos=camera_pos,
            search_mode=base_search_mode, ik_solver=ik_solver,
            mink_context=mink_context, mink_position_context=mink_position_context,
            base_orientation_mode=spec.base_orientation_mode,
            enable_base_orientation_search=enable_base_orientation_search,
            trajectory_orientation_search=trajectory_orientation_base_search,
            base_height_anchor=shared_base_height, base_forward_anchor=shared_base_forward,
            base_forward_offsets=spec.base_forward_offsets,
            max_target_distance=spec.base_max_target_distance,
            support_surface=support_surface,
            mink_projection_context=mink_projection_context,
            projection_planes_world=right_projection_stack,
            projection_directions_world=right_projection_directions)
        right_choice = max(right_candidates, key=lambda x: (x["score"], x["feasibility_rate"]))
    else:
        right_kf_idx, right_choice = empty_kf, neutral_base_choice(+1.0)
    if has_left and has_right:
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
        warm_l = (_prewarm_dls(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            left_R_base.T @ (left_p[0] - left_base_pos)) if has_left else None)
        warm_r = (_prewarm_dls(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            right_R_base.T @ (right_p[0] - right_base_pos)) if has_right else None)
    else:
        # Let the first full-pose solve start without a hard continuity bound.
        # A position-only prewarm can be on a wrist branch that is over 0.35 rad
        # from any orientation-compliant solution; constraining that first frame
        # makes the entire trajectory report IK failure despite tiny position
        # residuals. Subsequent frames are warm-started continuously below.
        warm_l = None
        warm_r = None

    def solve_trajectory_ik(target_pos, target_R, warm,
                            projection_plane_normals=None,
                            projection_plane_directions=None):
        if not use_mink_ik:
            return (*solve_arm_ik_dls_robust(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
                target_pos, target_R, q_warm=warm,
                continuity_max_step=mink_continuity_max_step),
                    False, False, False, False)

        projection_valid = bool(
            mink_projection_context is not None and
            projection_plane_normals is not None and
            np.isfinite(projection_plane_normals).all())
        projected_result = None
        if projection_valid:
            # Path B observes the opening line but cannot recover its 3D depth
            # component. Solve position and that one image-space orientation
            # constraint jointly; a position-first pass can settle on a wrist
            # limit from which the later orientation refinement cannot move.
            dual_projection = len(
                mink_projection_context.projection_tasks) > 1
            projected = mink_projection_context.solve(
                target_pos, target_R, q_init=warm,
                max_iterations=300 if dual_projection else 600,
                dt=0.02 if dual_projection else 0.01,
                tol_pos=POSITION_FIRST_TOL,
                tol_rot=0.08 if dual_projection else 0.02,
                continuity_q=warm,
                continuity_max_step=mink_continuity_max_step,
                position_first=False,
                projection_plane_normals=projection_plane_normals,
                projection_plane_directions=projection_plane_directions)
            projected_result = projected
            if projected[2] < BASE_FEAS_POS_TOL:
                q, _, err_pos, err_rot = projected
                return (q, bool(err_rot < BASE_FEAS_ROT_TOL),
                        err_pos, err_rot, False, False, False, True)

        # Follow EgoDex's continuous control loop before trying any restart:
        # start from the previous configuration, keep orientation as a soft
        # objective, and stop once the pinch-center position is reached.
        # Underactuated morphologies (for example SO-ARM101) cannot satisfy
        # the human wrist orientation and position simultaneously. Their
        # position-priority pass must therefore remove orientation costs;
        # otherwise the wrist-roll limit can freeze the arm several frames
        # before the position target becomes unreachable.
        position_context = (
            mink_position_context if spec.ik_position_priority else mink_context)
        continuous = solve_arm_ik_position_first(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            target_pos, target_R, q_init=warm, max_iter=500,
            tol_pos=POSITION_FIRST_TOL, tol_rot=BASE_FEAS_ROT_TOL,
            mink_context=position_context, continuity_q=warm,
            continuity_max_step=mink_continuity_max_step)
        if continuous[2] < BASE_FEAS_POS_TOL:
            if primary:
                # Branch-only mode: the frame-wise pass only needs to find a
                # feasible, continuous position branch. Skip per-frame pose
                # refinement and hand orientation quality to the whole-trajectory
                # optimizer, which owns the final trajectory.
                q, _, err_pos, err_rot = continuous
                return (q, bool(err_pos < BASE_FEAS_POS_TOL and
                                err_rot < BASE_FEAS_ROT_TOL),
                        err_pos, err_rot, False, False, False, False)
            refinement_context = (
                mink_projection_context if projection_valid else mink_context)
            refined = refine_arm_ik_pose(
                single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
                target_pos, target_R, continuous, previous_q=warm,
                max_iter=POSE_REFINEMENT_MAX_ITER,
                tol_pos=POSITION_FIRST_TOL, tol_rot=BASE_FEAS_ROT_TOL,
                mink_context=refinement_context,
                continuity_max_step=mink_continuity_max_step,
                projection_plane_normals=(
                    projection_plane_normals
                    if refinement_context is mink_projection_context else None),
                projection_plane_directions=(
                    projection_plane_directions
                    if refinement_context is mink_projection_context else None))
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

        # If every position recovery failed, preserve the projection-aware
        # branch as the next warm start. A position-only failure here would
        # otherwise rotate the wrist away from both image axes and contaminate
        # several subsequent, reachable frames. The failed frame itself is
        # still replaced by position interpolation below.
        if projected_result is not None:
            q, _, err_pos, err_rot = projected_result
            return (q, False, err_pos, err_rot,
                    False, False, False, False)

        # Never promote a distant recovery branch into the next frame's warm
        # start. The continuous result obeys the trust region and is therefore
        # the least damaging failed-position state for later interpolation.
        return (*continuous, False, False, False, False)

    for t in range(N):
        if has_left:
            p_base_l = left_R_base.T @ (left_p[t] - left_base_pos)
            R_base_l = left_R_base.T @ left_R[t]
            planes_base_l = (
                left_projection_stack[t] @ left_R_base
                if left_projection_stack is not None else None)
            directions_base_l = (
                left_projection_directions[t] @ left_R_base
                if left_projection_directions is not None else None)
            q_l, okl, epl, erl, fallback_l, branch_l, dls_l, refined_l = \
                solve_trajectory_ik(
                    p_base_l, R_base_l, warm_l, planes_base_l,
                    directions_base_l)
            warm_l = q_l.copy()
        else:
            q_l = np.zeros(single.nq)
            okl, epl, erl = True, 0.0, 0.0
            fallback_l = branch_l = dls_l = refined_l = False
        if has_right:
            p_base_r = right_R_base.T @ (right_p[t] - right_base_pos)
            R_base_r = right_R_base.T @ right_R[t]
            planes_base_r = (
                right_projection_stack[t] @ right_R_base
                if right_projection_stack is not None else None)
            directions_base_r = (
                right_projection_directions[t] @ right_R_base
                if right_projection_directions is not None else None)
            q_r, okr, epr, err, fallback_r, branch_r, dls_r, refined_r = \
                solve_trajectory_ik(
                    p_base_r, R_base_r, warm_r, planes_base_r,
                    directions_base_r)
            warm_r = q_r.copy()
        else:
            q_r = np.zeros(single.nq)
            okr, epr, err = True, 0.0, 0.0
            fallback_r = branch_r = dls_r = refined_r = False

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
    # In primary mode this discontinuity repair is absorbed by the smoothness /
    # acceleration cost of the whole-trajectory optimizer, so it is skipped.
    if has_left and left_projection_stack is None and not primary:
        n_fix_l, pose_ok_l, position_ok_l = repair_branch_jumps(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            Q_l, left_p, left_R, left_base_pos, left_R_base, ik_pose_ok[:, 0],
            position_ok_flags=ik_position_ok[:, 0],
            mink_context=mink_context,
            ik_solver=(solve_arm_ik_position_first if use_mink_ik else solve_arm_ik_dls))
    else:
        n_fix_l, pose_ok_l, position_ok_l = 0, ik_pose_ok[:, 0], ik_position_ok[:, 0]
    if has_right and right_projection_stack is None and not primary:
        n_fix_r, pose_ok_r, position_ok_r = repair_branch_jumps(
            single, arm_qadr, arm_vadr, arm_ids, ee_ref_single,
            Q_r, right_p, right_R, right_base_pos, right_R_base, ik_pose_ok[:, 1],
            position_ok_flags=ik_position_ok[:, 1],
            mink_context=mink_context,
            ik_solver=(solve_arm_ik_position_first if use_mink_ik else solve_arm_ik_dls))
    else:
        n_fix_r, pose_ok_r, position_ok_r = 0, ik_pose_ok[:, 1], ik_position_ok[:, 1]
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
    if use_mink_ik and has_left and has_right:
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
    # Speed clamping is likewise absorbed by the velocity cost in primary mode.
    n_clamped_left = (0 if primary else
                      clamp_joint_speed(qpos_all, l_qadr, max_jump_rad))
    n_clamped_right = (0 if primary else
                       clamp_joint_speed(qpos_all, r_qadr, max_jump_rad))
    if n_interpolated or n_clamped_left or n_clamped_right:
        print(f"    post-ik: interp position-failed={n_interpolated} frames, "
              f"speed-clamp L={n_clamped_left} R={n_clamped_right}")
    # Snapshot the exact trajectory that refinement is allowed to fall back to.
    # Q_l/Q_r predate failed-frame interpolation and optional speed clamping.
    pre_refine_arms = [qpos_all[:, l_qadr].copy(),
                       qpos_all[:, r_qadr].copy()]

    # The frame-wise Mink chain supplies a feasible branch and remains the
    # fallback. Refine each complete arm trajectory against the same TCP
    # targets while jointly penalizing velocity, acceleration, deviation from
    # the morphology home pose, and proximity to joint limits. A refined arm is
    # accepted only if the total objective improves without materially
    # regressing its mean or p95 TCP position error.
    refinement_results = [None, None]
    dual_refinement_result = None
    if trajectory_refine and use_mink_ik and N > 1:
        projection_axes = None
        projection_costs = None
        if mink_projection_context is not None:
            projection_axes = np.stack([
                task.axis for task in mink_projection_context.projection_tasks
            ])
            projection_costs = np.asarray([
                spec.ik_projection_cost,
                *([spec.ik_secondary_projection_cost]
                  if use_secondary_projection else []),
            ], dtype=float)

        def refine_side(side_index, active, qadr, target_p, target_R,
                        base_pos, base_R, valid, projection_stack,
                        projection_directions):
            if not active:
                return None
            targets_p_base = np.einsum(
                "ij,nj->ni", base_R.T, target_p - base_pos)
            targets_R_base = np.einsum(
                "ij,njk->nik", base_R.T, target_R)
            normals_base = (None if projection_stack is None else
                            np.einsum("nkj,ji->nki",
                                      projection_stack, base_R))
            directions_base = (None if projection_directions is None else
                               np.einsum("nkj,ji->nki",
                                         projection_directions, base_R))
            result = refine_arm_trajectory(
                single, arm_qadr, arm_ids, ee_ref_single,
                qpos_all[:, qadr], targets_p_base, targets_R_base, valid,
                config=trajectory_refine_config,
                orientation_cost=spec.ik_orientation_cost,
                projection_axes=projection_axes,
                projection_normals=normals_base,
                projection_directions=directions_base,
                projection_costs=projection_costs)
            label = "L" if side_index == 0 else "R"
            before, after = result.metrics_before, result.metrics_after
            if result.accepted:
                qpos_all[:, qadr] = result.qpos
                print(
                    f"    trajectory-refine {label}: accepted, "
                    f"cost {result.initial_cost:.2f}->{result.final_cost:.2f}, "
                    f"pos {before['position_mean']*1000:.1f}->"
                    f"{after['position_mean']*1000:.1f}mm, "
                    f"vel {before['velocity_rms']:.4f}->"
                    f"{after['velocity_rms']:.4f}, acc "
                    f"{before['acceleration_rms']:.4f}->"
                    f"{after['acceleration_rms']:.4f}, margin "
                    f"{before['joint_margin_min']:.3f}->"
                    f"{after['joint_margin_min']:.3f}")
            else:
                print(f"    trajectory-refine {label}: kept Mink trajectory "
                      f"({result.rejection_reason})")
            return result

        run_independent_refinement = not (
            has_left and has_right and
            trajectory_refine_config.collision_weight > 0.0)
        if run_independent_refinement:
            refinement_results[0] = refine_side(
                0, has_left, l_qadr, left_p, left_R, left_base_pos, left_R_base,
                left_valid, left_projection_stack, left_projection_directions)
            refinement_results[1] = refine_side(
                1, has_right, r_qadr, right_p, right_R, right_base_pos, right_R_base,
                right_valid, right_projection_stack, right_projection_directions)

        # When both hands are active, replace the two independent candidates
        # with one coupled solve. Cross-arm signed-distance residuals are active
        # inside the optimizer, so pose/smoothness improvements cannot silently
        # move one arm through the other. The independent results above remain
        # useful diagnostics and a fallback if the coupled solve regresses TCP.
        if has_left and has_right and trajectory_refine_config.collision_weight > 0.0:
            left_targets_p_base = np.einsum(
                "ij,nj->ni", left_R_base.T, left_p - left_base_pos)
            right_targets_p_base = np.einsum(
                "ij,nj->ni", right_R_base.T, right_p - right_base_pos)
            left_targets_R_base = np.einsum(
                "ij,njk->nik", left_R_base.T, left_R)
            right_targets_R_base = np.einsum(
                "ij,njk->nik", right_R_base.T, right_R)
            left_normals_base = (None if left_projection_stack is None else
                np.einsum("nkj,ji->nki", left_projection_stack, left_R_base))
            right_normals_base = (None if right_projection_stack is None else
                np.einsum("nkj,ji->nki", right_projection_stack, right_R_base))
            left_directions_base = (None if left_projection_directions is None else
                np.einsum("nkj,ji->nki", left_projection_directions, left_R_base))
            right_directions_base = (None if right_projection_directions is None else
                np.einsum("nkj,ji->nki", right_projection_directions, right_R_base))
            dual_refinement_result = refine_dual_arm_trajectory(
                single, dual, arm_qadr, arm_ids, ee_ref_single,
                l_qadr, r_qadr, pre_refine_arms[0], pre_refine_arms[1],
                left_targets_p_base, left_targets_R_base,
                right_targets_p_base, right_targets_R_base,
                left_valid, right_valid, config=trajectory_refine_config,
                orientation_cost=spec.ik_orientation_cost,
                left_projection_axes=projection_axes,
                left_projection_normals=left_normals_base,
                left_projection_directions=left_directions_base,
                left_projection_costs=projection_costs,
                right_projection_axes=projection_axes,
                right_projection_normals=right_normals_base,
                right_projection_directions=right_directions_base,
                right_projection_costs=projection_costs,
                baseline_qpos=qpos_all[0])
            if dual_refinement_result["accepted"]:
                qpos_all[:, l_qadr] = dual_refinement_result["left_qpos"]
                qpos_all[:, r_qadr] = dual_refinement_result["right_qpos"]
            else:
                print(
                    "    trajectory-refine dual: kept Mink trajectory "
                    f"({dual_refinement_result['rejection_reason']})")
            print(
                "    trajectory-refine dual: "
                f"cost {dual_refinement_result['initial_cost']:.2f}->"
                f"{dual_refinement_result['final_cost']:.2f}, collision "
                f"{dual_refinement_result['initial_collision_cost']:.2f}->"
                f"{dual_refinement_result['final_collision_cost']:.2f}, "
                f"pairs={dual_refinement_result['geom_pair_count']} "
                f"nfev={dual_refinement_result['nfev']}")
            projection_counts = [
                dual_refinement_result[f"position_projection_{side}"]
                ["projected_frames"] for side in ("left", "right")]
            restore_counts = [
                len(dual_refinement_result[f"position_restore_{side}"]
                    ["restored_frames"]) for side in ("left", "right")]
            print(
                "    trajectory-refine position feasibility: "
                f"preproject L/R={projection_counts[0]}/{projection_counts[1]}, "
                f"restored L/R={restore_counts[0]}/{restore_counts[1]}, "
                f"max={dual_refinement_result['max_position_error']*1000:.2f}mm")

            # Keep a hard full-model safety check after the soft signed-distance
            # objective. Non-smooth closest-pair switches can leave a residual
            # penetration at a local optimum; never publish that trajectory.
            if dual_refinement_result["accepted"]:
                probe = mujoco.MjData(dual)
                dual_collision_frames = 0
                dual_max_penetration = 0.0
                for frame in range(N):
                    probe.qpos[:] = qpos_all[frame]
                    mujoco.mj_forward(dual, probe)
                    collision = _quality_collision_metrics(dual, probe)
                    penetrating = (collision["self_collision"] or
                                   collision["cross_arm_contact_count"] > 0)
                    dual_collision_frames += int(penetrating)
                    dual_max_penetration = max(
                        dual_max_penetration, collision["self_penetration"],
                        collision["cross_arm_penetration"])
                if dual_collision_frames:
                    qpos_all[:, l_qadr] = pre_refine_arms[0]
                    qpos_all[:, r_qadr] = pre_refine_arms[1]
                    dual_refinement_result["accepted"] = False
                    dual_refinement_result["rejection_reason"] = (
                        f"{dual_collision_frames} penetrating frames, max "
                        f"{dual_max_penetration * 1000.0:.1f}mm")
                    print(
                        "    trajectory-refine dual: hard collision guard kept "
                        f"Mink trajectory ({dual_refinement_result['rejection_reason']})")

        # Independent arm refinements can introduce collisions that the
        # pre-refinement branch repair never saw. Reject only the changed side
        # when it increases the number of penetrating self/cross-arm contacts.
        if (dual_refinement_result is None and
                any(result is not None and result.accepted
                    for result in refinement_results)):
            refined_arms = [
                None if refinement_results[0] is None else
                refinement_results[0].qpos.copy(),
                None if refinement_results[1] is None else
                refinement_results[1].qpos.copy(),
            ]
            baseline_arms = pre_refine_arms

            def collision_counts():
                probe = mujoco.MjData(dual)
                self_frames = 0
                cross_frames = 0
                for frame in range(N):
                    probe.qpos[:] = qpos_all[frame]
                    mujoco.mj_forward(dual, probe)
                    collision = _quality_collision_metrics(dual, probe)
                    self_frames += int(collision["self_collision"])
                    cross_frames += int(
                        collision["cross_arm_contact_count"] > 0)
                return self_frames, cross_frames

            refined_collision = collision_counts()
            accepted_sides = [
                index for index, result in enumerate(refinement_results)
                if result is not None and result.accepted
            ]
            for index in accepted_sides:
                qadr = l_qadr if index == 0 else r_qadr
                qpos_all[:, qadr] = baseline_arms[index]
            baseline_collision = collision_counts()
            for index in accepted_sides:
                qadr = l_qadr if index == 0 else r_qadr
                qpos_all[:, qadr] = refined_arms[index]

            if (refined_collision[0] > baseline_collision[0] or
                    refined_collision[1] > baseline_collision[1]):
                # With two changed sides, test each alone before rejecting both.
                survivors = []
                for index in accepted_sides:
                    for other in accepted_sides:
                        qadr = l_qadr if other == 0 else r_qadr
                        qpos_all[:, qadr] = baseline_arms[other]
                    qadr = l_qadr if index == 0 else r_qadr
                    qpos_all[:, qadr] = refined_arms[index]
                    side_collision = collision_counts()
                    if (side_collision[0] <= baseline_collision[0] and
                            side_collision[1] <= baseline_collision[1]):
                        survivors.append(index)
                for index in accepted_sides:
                    qadr = l_qadr if index == 0 else r_qadr
                    if index in survivors:
                        qpos_all[:, qadr] = refined_arms[index]
                    else:
                        qpos_all[:, qadr] = baseline_arms[index]
                        result = refinement_results[index]
                        result.accepted = False
                        result.qpos = baseline_arms[index]
                        result.rejection_reason = (
                            "collision frames increased from "
                            f"{baseline_collision} to {refined_collision}")
                print("    trajectory-refine collision guard: "
                      f"baseline={baseline_collision}, both={refined_collision}, "
                      f"kept sides={survivors}")

        # Refresh FK diagnostics against the final accepted qpos. This also
        # makes saved IK quality consistent with interpolation/refinement.
        def refresh_side(side_index, active, qadr, target_p, target_R,
                         base_pos, base_R, valid, projection_stack,
                         projection_directions):
            if not active:
                return
            targets_p_base = np.einsum(
                "ij,nj->ni", base_R.T, target_p - base_pos)
            targets_R_base = np.einsum(
                "ij,njk->nik", base_R.T, target_R)
            normals_base = (None if projection_stack is None else
                            np.einsum("nkj,ji->nki",
                                      projection_stack, base_R))
            directions_base = (None if projection_directions is None else
                               np.einsum("nkj,ji->nki",
                                         projection_directions, base_R))
            pos_error, rot_error = evaluate_arm_trajectory(
                single, arm_qadr, ee_ref_single, qpos_all[:, qadr],
                targets_p_base, targets_R_base, valid,
                orientation_axes=np.asarray(spec.ik_orientation_cost) > 0.0,
                projection_axes=projection_axes,
                projection_normals=normals_base,
                projection_directions=directions_base,
                projection_costs=projection_costs)
            ik_err_pos[valid, side_index] = pos_error[valid]
            ik_err_rot[valid, side_index] = rot_error[valid]
            ik_position_ok[valid, side_index] = (
                pos_error[valid] < BASE_FEAS_POS_TOL)
            ik_pose_ok[valid, side_index] = (
                ik_position_ok[valid, side_index] &
                (rot_error[valid] < BASE_FEAS_ROT_TOL))

        refresh_side(
            0, has_left, l_qadr, left_p, left_R, left_base_pos, left_R_base,
            left_valid, left_projection_stack, left_projection_directions)
        refresh_side(
            1, has_right, r_qadr, right_p, right_R, right_base_pos, right_R_base,
            right_valid, right_projection_stack, right_projection_directions)

    # Persist final-trajectory FK metrics even when refinement is disabled, so
    # interpolation or optional speed clamping cannot leave stale diagnostics.
    if not (trajectory_refine and use_mink_ik and N > 1):
        def refresh_without_refinement(side_index, active, qadr, target_p,
                                       target_R, base_pos, base_R, valid,
                                       projection_stack,
                                       projection_directions):
            if not active:
                return
            targets_p_base = np.einsum(
                "ij,nj->ni", base_R.T, target_p - base_pos)
            targets_R_base = np.einsum(
                "ij,njk->nik", base_R.T, target_R)
            normals_base = (None if projection_stack is None else
                            np.einsum("nkj,ji->nki",
                                      projection_stack, base_R))
            directions_base = (None if projection_directions is None else
                               np.einsum("nkj,ji->nki",
                                         projection_directions, base_R))
            local_projection_axes = None
            local_projection_costs = None
            if mink_projection_context is not None:
                local_projection_axes = np.stack([
                    task.axis for task in
                    mink_projection_context.projection_tasks])
                local_projection_costs = np.asarray([
                    spec.ik_projection_cost,
                    *([spec.ik_secondary_projection_cost]
                      if use_secondary_projection else []),
                ], dtype=float)
            pos_error, rot_error = evaluate_arm_trajectory(
                single, arm_qadr, ee_ref_single, qpos_all[:, qadr],
                targets_p_base, targets_R_base, valid,
                orientation_axes=np.asarray(spec.ik_orientation_cost) > 0.0,
                projection_axes=local_projection_axes,
                projection_normals=normals_base,
                projection_directions=directions_base,
                projection_costs=local_projection_costs)
            ik_err_pos[valid, side_index] = pos_error[valid]
            ik_err_rot[valid, side_index] = rot_error[valid]
            ik_position_ok[valid, side_index] = (
                pos_error[valid] < BASE_FEAS_POS_TOL)
            ik_pose_ok[valid, side_index] = (
                ik_position_ok[valid, side_index] &
                (rot_error[valid] < BASE_FEAS_ROT_TOL))

        refresh_without_refinement(
            0, has_left, l_qadr, left_p, left_R, left_base_pos, left_R_base,
            left_valid, left_projection_stack, left_projection_directions)
        refresh_without_refinement(
            1, has_right, r_qadr, right_p, right_R, right_base_pos, right_R_base,
            right_valid, right_projection_stack, right_projection_directions)

    # 5. Render and composite.
    bg_frames = read_video_frames(bg_path)
    if not bg_frames:
        print("    [SKIP] no bg frames")
        return None
    Nrender = min(N, len(bg_frames))
    H, W = bg_frames[0].shape[:2]
    human_hand_masks = None
    # Occlusion margin: robot pixels are kept when they are at least this far in
    # front of the aligned scene depth.  Adapted below to the alignment residual.
    depth_occ_margin = depth_epsilon
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
        human_hand_masks = load_human_hand_masks(
            mask_dir, ep, Nrender, dilation_px=human_hand_mask_dilation)
        if human_hand_masks.shape[1:] != (H, W):
            human_hand_masks = np.stack([
                cv2.resize(mask.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST) > 0
                for mask in human_hand_masks
            ])
        # DA3 depth is monocular metric depth in its own scale; align it to the
        # EgoVerse world scale (matching the MuJoCo robot depth) before the
        # occlusion test, otherwise the depth comparison culls robot pixels at
        # an arbitrary offset.
        if depth_align:
            if ep_attrs is None:
                ep_attrs, _ = config.fallback_episode_attrs(
                    str(Path(zarr_dir) / ep))
            K = config.dataset_intrinsics_k(
                ep_attrs or {}, camera="front_1",
                img_shape=scene_depth.shape[1:])
            hands = [(left_kp, left_valid)] if has_left else []
            if has_right:
                hands.append((right_kp, right_valid))
            scale, shift, align_stats = _estimate_scene_depth_alignment(
                scene_depth, head, hands, K)
            if align_stats["mode"] != "identity":
                scene_depth = np.clip(
                    scale * scene_depth.astype(np.float32) + shift,
                    1e-3, None).astype(np.float32)
            print(f"    [{ep}] depth align: mode={align_stats['mode']} "
                  f"scale={scale:.4f} shift={shift:.4f} "
                  f"pairs={align_stats['n_pairs']}"
                  + (f" reason={align_stats['reason']}"
                     if "reason" in align_stats else ""))
            # The occlusion decision compares aligned scene depth to rendered
            # robot depth.  When the DA3->world alignment is uncertain, that
            # comparison is only trustworthy beyond the alignment residual, so
            # a robot pixel that sits within the residual of the scene surface
            # (e.g. an arm grazing the table plane) must not be culled.  Fold
            # the residual into the occlusion margin as a negative (tolerant)
            # bias, which can relax the strict depth_epsilon requirement.
            align_resid = align_stats.get("resid")
            if (align_stats["mode"] != "identity" and align_resid is not None
                    and np.isfinite(align_resid) and align_resid > 0):
                depth_occ_margin = depth_epsilon - DEPTH_ALIGN_TOL_GAIN * float(
                    align_resid)
                print(f"    [{ep}] depth occlusion margin: "
                      f"{depth_occ_margin:+.4f} m "
                      f"(epsilon={depth_epsilon:.4f} - "
                      f"{DEPTH_ALIGN_TOL_GAIN:.1f}*resid={align_resid:.4f})")

    active_sides = tuple(side for side, active in
                         (("left", has_left), ("right", has_right)) if active)
    hide_non_arm_geoms(dual, spec, active_sides=active_sides)
    depth_test_geom_ids = gripper_geom_ids(
        dual, spec, active_sides=active_sides)
    if scene_depth is not None and depth_test_geom_ids.size == 0:
        raise ValueError(f"no visible gripper geometry found for {spec.name}")
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
    depth_visibility_ratio = np.ones(N, dtype=np.float32)
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
        gripper_mask = (render_geom_subset_mask(
            dual, data, renderer, cam_id, depth_test_geom_ids)
            if scene_depth is not None else None)
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
        if gripper_mask is not None and gripper_mask.shape[:2] != (H, W):
            gripper_mask = cv2.resize(
                gripper_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST) > 0
        robot_bgr = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)
        m = feather_mask(mask, feather)
        if scene_depth is not None:
            stable_scene_depth = temporal_median_depth_frame(
                scene_depth, t, window=depth_temporal_window)
            depth_alpha = depth_visibility_alpha(
                gripper_mask, robot_depth, stable_scene_depth,
                margin=depth_occ_margin,
                transition_width=depth_transition_width)
            # The arm body is always visible. Only gripper pixels take the
            # scene-depth result. Around the original human hand boundary, the
            # dilated protection mask restores robot alpha to prevent erosion
            # caused by SAM/inpainting/depth edge disagreement.
            protected_gripper = gripper_mask & human_hand_masks[t]
            depth_alpha[protected_gripper] = 1.0
            composite_depth_alpha = np.ones(mask.shape, dtype=np.float32)
            composite_depth_alpha[gripper_mask] = depth_alpha[gripper_mask]
            gripper_pixels = int(gripper_mask.sum())
            if gripper_pixels:
                depth_visibility_ratio[t] = float(
                    composite_depth_alpha[gripper_mask].mean())
            m *= composite_depth_alpha[..., None]
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
                        trajectory_orientation_base_search=np.asarray(
                            trajectory_orientation_base_search),
                        ik_solver=np.asarray(ik_solver),
                        ik_orientation_cost=np.asarray(
                            spec.ik_orientation_cost, dtype=np.float32),
                        ik_projection_axis=np.asarray(
                            spec.ik_projection_axis
                            if spec.ik_projection_axis is not None else
                            [np.nan, np.nan, np.nan], dtype=np.float32),
                        ik_projection_cost=np.asarray(
                            spec.ik_projection_cost, dtype=np.float32),
                        ik_secondary_projection_axis=np.asarray(
                            spec.ik_secondary_projection_axis
                            if spec.ik_secondary_projection_axis is not None else
                            [np.nan, np.nan, np.nan], dtype=np.float32),
                        ik_secondary_projection_cost=np.asarray(
                            spec.ik_secondary_projection_cost,
                            dtype=np.float32),
                        ik_projection_enabled=np.asarray(use_projection_ik),
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
                        trajectory_refine_enabled=np.asarray(
                            trajectory_refine and use_mink_ik),
                        trajectory_refine_accepted=np.asarray([
                            result is not None and result.accepted
                            for result in refinement_results], dtype=bool),
                        trajectory_refine_initial_cost=np.asarray([
                            np.nan if result is None else result.initial_cost
                            for result in refinement_results], dtype=np.float32),
                        trajectory_refine_final_cost=np.asarray([
                            np.nan if result is None else result.final_cost
                            for result in refinement_results], dtype=np.float32),
                        trajectory_refine_position_weight=np.asarray(
                            trajectory_refine_config.position_weight, dtype=np.float32),
                        trajectory_refine_orientation_weight=np.asarray(
                            trajectory_refine_config.orientation_weight, dtype=np.float32),
                        trajectory_refine_velocity_weight=np.asarray(
                            trajectory_refine_config.velocity_weight, dtype=np.float32),
                        trajectory_refine_acceleration_weight=np.asarray(
                            trajectory_refine_config.acceleration_weight, dtype=np.float32),
                        trajectory_refine_home_weight=np.asarray(
                            trajectory_refine_config.home_weight, dtype=np.float32),
                        trajectory_refine_joint_margin_weight=np.asarray(
                            trajectory_refine_config.joint_margin_weight, dtype=np.float32),
                        trajectory_refine_joint_margin_fraction=np.asarray(
                            trajectory_refine_config.joint_margin_fraction, dtype=np.float32),
                        trajectory_refine_collision_weight=np.asarray(
                            trajectory_refine_config.collision_weight, dtype=np.float32),
                        trajectory_refine_collision_safe_distance=np.asarray(
                            trajectory_refine_config.collision_safe_distance, dtype=np.float32),
                        trajectory_refine_position_tolerance=np.asarray(
                            trajectory_refine_config.position_tolerance, dtype=np.float32),
                        trajectory_refine_dual_enabled=np.asarray(
                            dual_refinement_result is not None),
                        trajectory_refine_dual_initial_cost=np.asarray(
                            dual_refinement_result["initial_cost"]
                            if dual_refinement_result is not None else np.nan,
                            dtype=np.float32),
                        trajectory_refine_dual_final_cost=np.asarray(
                            dual_refinement_result["final_cost"]
                            if dual_refinement_result is not None else np.nan,
                            dtype=np.float32),
                        trajectory_refine_dual_final_collision_cost=np.asarray(
                            dual_refinement_result["final_collision_cost"]
                            if dual_refinement_result is not None else np.nan,
                            dtype=np.float32),
                        trajectory_refine_dual_max_position_error=np.asarray(
                            dual_refinement_result["max_position_error"]
                            if dual_refinement_result is not None else np.nan,
                            dtype=np.float32),
                        trajectory_refine_max_nfev=np.asarray(
                            trajectory_refine_config.max_nfev, dtype=np.int32),
                        primary_trajectory_opt=np.asarray(primary),
                        use_analytic_jacobian=np.asarray(
                            trajectory_refine_config.use_analytic_jacobian),
                        **support_metadata)
    np.savez_compressed(str(out_dir / f"{ep}_quality.npz"),
                        hand_detected=hand_detected,
                        rendered=rendered,
                        robot_pixel_count=robot_pixel_count,
                        robot_mask_ratio=robot_mask_ratio,
                        depth_visibility_ratio=depth_visibility_ratio,
                        self_collision=self_collision,
                        self_contact_count=self_contact_count,
                        self_penetration=self_penetration,
                        cross_arm_contact_count=cross_arm_contact_count,
                        cross_arm_penetration=cross_arm_penetration,
                        fps=np.asarray(float(fps)),
                        robot_type=np.asarray(spec.name))
    print(f"    ✓ {out_final}")
    return out_final
