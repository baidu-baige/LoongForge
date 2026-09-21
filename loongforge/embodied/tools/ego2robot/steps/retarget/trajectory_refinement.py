# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Episode-level refinement of frame-wise IK trajectories.

The frame-wise Mink pass remains responsible for finding a feasible IK branch.
This module treats that result as an initial guess and improves the complete
trajectory with sparse least squares while retaining the end-effector targets.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TrajectoryRefinementConfig:
    """Weights and guards for whole-trajectory refinement."""

    position_weight: float = 1.0
    orientation_weight: float = 0.05
    velocity_weight: float = 0.2
    acceleration_weight: float = 1.0
    home_weight: float = 0.002
    joint_margin_weight: float = 0.05
    joint_margin_fraction: float = 0.10
    collision_weight: float = 4.0
    collision_safe_distance: float = 0.015
    collision_scale: float = 0.01
    position_tolerance: float = 0.005
    position_constraint_weight: float = 1000.0
    position_scale: float = 0.01
    orientation_scale: float = 0.25
    max_nfev: int = 20
    max_mean_position_regression: float = 0.002
    max_p95_position_regression: float = 0.003
    # When ``primary`` is set the whole-trajectory optimizer produces the final
    # trajectory (the frame-wise pass is only an initial guess), so the
    # "objective did not improve" rejection is dropped and the position
    # regression thresholds are relaxed to the primary_* values below.
    primary: bool = False
    primary_max_mean_position_regression: float = 0.010
    primary_max_p95_position_regression: float = 0.015
    # Feed scipy an exact MuJoCo-Jacobian ``jac`` instead of the numerical
    # estimate driven by ``jac_sparsity``.
    use_analytic_jacobian: bool = False

    def validate(self):
        for name in (
            "position_weight", "orientation_weight", "velocity_weight",
            "acceleration_weight", "home_weight",
            "joint_margin_weight", "collision_weight",
            "position_constraint_weight",
            "primary_max_mean_position_regression",
            "primary_max_p95_position_regression",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < float(self.joint_margin_fraction) < 0.5:
            raise ValueError("joint_margin_fraction must be in (0, 0.5)")
        if not np.isfinite(self.position_scale) or self.position_scale <= 0.0:
            raise ValueError("position_scale must be finite and positive")
        if not np.isfinite(self.orientation_scale) or self.orientation_scale <= 0.0:
            raise ValueError("orientation_scale must be finite and positive")
        if int(self.max_nfev) < 1:
            raise ValueError("max_nfev must be positive")
        if (not np.isfinite(self.collision_safe_distance) or
                self.collision_safe_distance < 0.0):
            raise ValueError(
                "collision_safe_distance must be finite and non-negative")
        if not np.isfinite(self.collision_scale) or self.collision_scale <= 0.0:
            raise ValueError("collision_scale must be finite and positive")
        if (not np.isfinite(self.position_tolerance) or
                self.position_tolerance <= 0.0):
            raise ValueError("position_tolerance must be finite and positive")


@dataclass
class TrajectoryRefinementResult:
    qpos: np.ndarray
    accepted: bool
    status: int
    nfev: int
    initial_cost: float
    final_cost: float
    metrics_before: dict
    metrics_after: dict
    rejection_reason: str = ""


def _ee_pose(model, data, ee_ref):
    kind, index = ee_ref
    if kind == "site":
        return data.site_xpos[index].copy(), data.site_xmat[index].reshape(3, 3).copy()
    return data.xpos[index].copy(), data.xmat[index].reshape(3, 3).copy()


def _ee_pose_and_jacobian(model, data, ee_ref, jacp, jacr):
    """Read EE pose and its MuJoCo geometric Jacobian (world = base frame).

    ``jacp``/``jacr`` are (3, model.nv) buffers filled in place. The columns
    correspond to velocity (DOF) addresses; the caller slices the arm DOFs.
    """
    kind, index = ee_ref
    if kind == "site":
        pos = data.site_xpos[index].copy()
        rotation = data.site_xmat[index].reshape(3, 3).copy()
        mujoco.mj_jacSite(model, data, jacp, jacr, index)
    else:
        pos = data.xpos[index].copy()
        rotation = data.xmat[index].reshape(3, 3).copy()
        mujoco.mj_jac(model, data, jacp, jacr, pos, index)
    return pos, rotation


def _set_arm_configuration(model, data, arm_qadr, arm_q):
    data.qpos[:] = model.qpos0
    for qpos_address, value in zip(arm_qadr, arm_q):
        data.qpos[int(qpos_address)] = float(value)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def _rotation_error(current, target):
    """Target-frame rotation vector from target to current."""
    return Rotation.from_matrix(np.asarray(target).T @ np.asarray(current)).as_rotvec()


def _normalize_projection_inputs(axes, normals, directions, costs, n_frames):
    if normals is None or axes is None:
        return None, None, None, None
    axes = np.asarray(axes, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(n_frames, len(axes), 3)
    if directions is None:
        directions = np.full_like(normals, np.nan)
    else:
        directions = np.asarray(directions, dtype=float).reshape(normals.shape)
    costs = np.ones(len(axes), dtype=float) if costs is None else np.asarray(costs, dtype=float)
    if costs.shape != (len(axes),):
        raise ValueError("projection costs must match projection axes")
    return axes, normals, directions, costs


def evaluate_arm_trajectory(
        model, arm_qadr, ee_ref, arm_q, target_pos, target_rot, tracking_mask,
        orientation_axes=(True, True, True), projection_axes=None,
        projection_normals=None, projection_directions=None,
        projection_costs=None):
    """Measure FK position and enabled orientation errors per frame."""
    arm_q = np.asarray(arm_q, dtype=float)
    n_frames = len(arm_q)
    target_pos = np.asarray(target_pos, dtype=float).reshape(n_frames, 3)
    target_rot = np.asarray(target_rot, dtype=float).reshape(n_frames, 3, 3)
    tracking_mask = np.asarray(tracking_mask, dtype=bool).reshape(n_frames)
    orientation_axes = np.asarray(orientation_axes, dtype=bool).reshape(3)
    proj_axes, proj_normals, proj_directions, _ = _normalize_projection_inputs(
        projection_axes, projection_normals, projection_directions,
        projection_costs, n_frames)

    data = mujoco.MjData(model)
    pos_error = np.full(n_frames, np.nan, dtype=float)
    rot_error = np.full(n_frames, np.nan, dtype=float)
    for frame in np.flatnonzero(tracking_mask):
        _set_arm_configuration(model, data, arm_qadr, arm_q[frame])
        pos, rotation = _ee_pose(model, data, ee_ref)
        pos_error[frame] = np.linalg.norm(pos - target_pos[frame])
        enabled = _rotation_error(rotation, target_rot[frame])[orientation_axes]
        frame_rot = float(np.linalg.norm(enabled)) if len(enabled) else 0.0
        projection_error = 0.0
        if proj_axes is not None:
            errors = []
            for axis, normal, direction in zip(
                    proj_axes, proj_normals[frame], proj_directions[frame]):
                if not np.isfinite(normal).all():
                    continue
                world_axis = rotation @ axis
                normal_component = float(world_axis @ normal)
                if np.isfinite(direction).all():
                    direction_component = float(world_axis @ direction)
                    errors.append(abs(np.arctan2(
                        normal_component, max(direction_component, 1e-8))))
                else:
                    errors.append(abs(np.arcsin(np.clip(normal_component, -1.0, 1.0))))
            projection_error = max(errors, default=0.0)
        rot_error[frame] = max(frame_rot, projection_error)
    return pos_error, rot_error


def _joint_bounds_and_home(model, arm_ids, arm_qadr):
    lower = np.full(len(arm_ids), -np.inf, dtype=float)
    upper = np.full(len(arm_ids), np.inf, dtype=float)
    for index, joint_id in enumerate(arm_ids):
        if model.jnt_limited[joint_id]:
            lower[index], upper[index] = model.jnt_range[joint_id]
    home = np.asarray(model.qpos0, dtype=float)[np.asarray(arm_qadr, dtype=int)].copy()
    home = np.maximum(home, np.where(np.isfinite(lower), lower, home))
    home = np.minimum(home, np.where(np.isfinite(upper), upper, home))
    return lower, upper, home


def _trajectory_metrics(q, pos_error, lower, upper):
    valid = np.isfinite(pos_error)
    position = pos_error[valid]
    velocity = np.diff(q, axis=0)
    acceleration = np.diff(q, n=2, axis=0)
    margins = []
    for joint in range(q.shape[1]):
        if not (np.isfinite(lower[joint]) and np.isfinite(upper[joint])):
            continue
        span = upper[joint] - lower[joint]
        if span > 1e-9:
            margins.append(np.minimum(
                q[:, joint] - lower[joint], upper[joint] - q[:, joint]) / span)
    return {
        "position_mean": float(np.mean(position)) if len(position) else 0.0,
        "position_p95": float(np.quantile(position, 0.95)) if len(position) else 0.0,
        "position_max": float(np.max(position)) if len(position) else 0.0,
        "velocity_rms": float(np.sqrt(np.mean(velocity ** 2))) if velocity.size else 0.0,
        "acceleration_rms": (float(np.sqrt(np.mean(acceleration ** 2)))
                             if acceleration.size else 0.0),
        "joint_margin_min": (float(np.min(np.stack(margins)))
                             if margins else 1.0),
    }


def _jacobian_sparsity(n_frames, n_joints, tracked_frames, orientation_rows,
                       projection_rows):
    row_count = (len(tracked_frames) * (3 + orientation_rows + projection_rows) +
                 max(n_frames - 1, 0) * n_joints +
                 max(n_frames - 2, 0) * n_joints +
                 n_frames * n_joints + n_frames * n_joints * 2)
    sparsity = lil_matrix((row_count, n_frames * n_joints), dtype=np.int8)
    row = 0
    frame_rows = 3 + orientation_rows + projection_rows
    for frame in tracked_frames:
        sparsity[row:row + frame_rows,
                 frame * n_joints:(frame + 1) * n_joints] = 1
        row += frame_rows
    for frame in range(max(n_frames - 1, 0)):
        for joint in range(n_joints):
            sparsity[row, frame * n_joints + joint] = 1
            sparsity[row, (frame + 1) * n_joints + joint] = 1
            row += 1
    for frame in range(max(n_frames - 2, 0)):
        for joint in range(n_joints):
            for offset in range(3):
                sparsity[row, (frame + offset) * n_joints + joint] = 1
            row += 1
    # Home prior, lower soft margin, upper soft margin.
    for _term in range(3):
        for frame in range(n_frames):
            for joint in range(n_joints):
                sparsity[row, frame * n_joints + joint] = 1
                row += 1
    if row != row_count:
        raise AssertionError(f"sparsity rows {row} != residual rows {row_count}")
    return sparsity.tocsr()


def refine_arm_trajectory(
        model, arm_qadr, arm_ids, ee_ref, initial_arm_q, target_pos,
        target_rot, tracking_mask, config=None, orientation_cost=(1.0, 1.0, 1.0),
        projection_axes=None, projection_normals=None,
        projection_directions=None, projection_costs=None,
        return_functions=False):
    """Refine one arm over the complete episode and guard TCP regressions.

    When ``return_functions`` is set the residual/analytic-Jacobian closures and
    their metadata are returned instead of running the solve; this exists purely
    so unit tests can compare the analytic Jacobian against finite differences.
    """
    cfg = config or TrajectoryRefinementConfig()
    cfg.validate()
    q0 = np.asarray(initial_arm_q, dtype=float).copy()
    if q0.ndim != 2 or q0.shape[1] != len(arm_qadr):
        raise ValueError("initial_arm_q must have shape (frames, arm joints)")
    n_frames, n_joints = q0.shape
    mask = np.asarray(tracking_mask, dtype=bool).reshape(n_frames)
    target_pos = np.asarray(target_pos, dtype=float).reshape(n_frames, 3)
    target_rot = np.asarray(target_rot, dtype=float).reshape(n_frames, 3, 3)
    orientation_cost = np.asarray(orientation_cost, dtype=float).reshape(3)
    orientation_mask = orientation_cost > 0.0
    proj_axes, proj_normals, proj_directions, proj_costs = _normalize_projection_inputs(
        projection_axes, projection_normals, projection_directions,
        projection_costs, n_frames)
    tracked_frames = np.flatnonzero(mask)
    lower, upper, home = _joint_bounds_and_home(model, arm_ids, arm_qadr)
    q0 = np.clip(q0, lower, upper)

    before_pos, before_rot = evaluate_arm_trajectory(
        model, arm_qadr, ee_ref, q0, target_pos, target_rot, mask,
        orientation_axes=orientation_mask, projection_axes=proj_axes,
        projection_normals=proj_normals, projection_directions=proj_directions,
        projection_costs=proj_costs)
    metrics_before = _trajectory_metrics(q0, before_pos, lower, upper)
    data = mujoco.MjData(model)
    orientation_scale = np.where(orientation_mask,
                                 orientation_cost / max(orientation_cost.max(), 1e-12), 0.0)
    # DOF (velocity) addresses for the arm joints, matching the columns of the
    # MuJoCo Jacobian buffers. Arm joints are single-DOF hinge/slide joints.
    arm_dof = np.asarray(
        [int(model.jnt_dofadr[int(j)]) for j in arm_ids], dtype=int)

    def residual(flat):
        q = flat.reshape(n_frames, n_joints)
        values = []
        for frame in tracked_frames:
            _set_arm_configuration(model, data, arm_qadr, q[frame])
            position, rotation = _ee_pose(model, data, ee_ref)
            values.extend((np.sqrt(cfg.position_weight) *
                           (position - target_pos[frame]) / cfg.position_scale).tolist())
            if cfg.orientation_weight > 0.0 and np.any(orientation_mask):
                error = _rotation_error(rotation, target_rot[frame])
                values.extend((np.sqrt(cfg.orientation_weight) *
                               orientation_scale[orientation_mask] *
                               error[orientation_mask] /
                               cfg.orientation_scale).tolist())
            if proj_axes is not None:
                for axis, normal, direction, cost in zip(
                        proj_axes, proj_normals[frame], proj_directions[frame], proj_costs):
                    projection_error = 0.0
                    if np.isfinite(normal).all():
                        current_axis = rotation @ axis
                        normal_component = float(current_axis @ normal)
                        if np.isfinite(direction).all():
                            projection_error = np.arctan2(
                                normal_component, max(float(current_axis @ direction), 1e-8))
                        else:
                            projection_error = np.arcsin(
                                np.clip(normal_component, -1.0, 1.0))
                    values.append(np.sqrt(cfg.orientation_weight) * float(cost) *
                                  projection_error / cfg.orientation_scale)
        if n_frames > 1:
            values.extend((np.sqrt(cfg.velocity_weight) * np.diff(q, axis=0)).ravel())
        if n_frames > 2:
            values.extend((np.sqrt(cfg.acceleration_weight) *
                           np.diff(q, n=2, axis=0)).ravel())
        values.extend((np.sqrt(cfg.home_weight) * (q - home)).ravel())

        lower_margin = np.zeros_like(q)
        upper_margin = np.zeros_like(q)
        for joint in range(n_joints):
            if not (np.isfinite(lower[joint]) and np.isfinite(upper[joint])):
                continue
            span = upper[joint] - lower[joint]
            soft = cfg.joint_margin_fraction * span
            if soft <= 1e-12:
                continue
            lower_margin[:, joint] = np.maximum(
                0.0, lower[joint] + soft - q[:, joint]) / soft
            upper_margin[:, joint] = np.maximum(
                0.0, q[:, joint] - (upper[joint] - soft)) / soft
        values.extend((np.sqrt(cfg.joint_margin_weight) * lower_margin).ravel())
        values.extend((np.sqrt(cfg.joint_margin_weight) * upper_margin).ravel())
        return np.asarray(values, dtype=float)

    orientation_rows = (int(np.count_nonzero(orientation_mask))
                        if cfg.orientation_weight > 0.0 else 0)
    projection_rows = len(proj_axes) if proj_axes is not None else 0
    n_cols = n_frames * n_joints
    orientation_active = cfg.orientation_weight > 0.0 and bool(np.any(orientation_mask))
    row_count = (len(tracked_frames) * (3 + orientation_rows + projection_rows) +
                 max(n_frames - 1, 0) * n_joints +
                 max(n_frames - 2, 0) * n_joints +
                 3 * n_frames * n_joints)

    def _analytic_jacobian(flat):
        """Exact Gauss-Newton Jacobian aligned row-for-row with ``residual``.

        Orientation and projection blocks use a first-order linearization
        (right/left Jacobian approximated by identity), matching the mink
        ``FrameTask``/``AxisProjectionPlaneTask`` convention. The unit tests
        check every block against finite differences.
        """
        q = flat.reshape(n_frames, n_joints)
        jac = lil_matrix((row_count, n_cols), dtype=float)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        sqrt_wp = np.sqrt(cfg.position_weight)
        sqrt_wo = np.sqrt(cfg.orientation_weight)
        row = 0
        for frame in tracked_frames:
            _set_arm_configuration(model, data, arm_qadr, q[frame])
            _, rotation = _ee_pose_and_jacobian(model, data, ee_ref, jacp, jacr)
            col = frame * n_joints
            arm_jacp = jacp[:, arm_dof]
            arm_jacr = jacr[:, arm_dof]
            # Position block: d(pos)/dq.
            jac[row:row + 3, col:col + n_joints] = (
                sqrt_wp / cfg.position_scale) * arm_jacp
            row += 3
            # Orientation block: d(log(R_t^T R_c))/dq ~= R_t^T @ jacr.
            if orientation_active:
                target_rot_T = target_rot[frame].T
                ori_jac = target_rot_T @ arm_jacr
                factor = (sqrt_wo * orientation_scale[orientation_mask] /
                          cfg.orientation_scale)
                jac[row:row + orientation_rows, col:col + n_joints] = (
                    factor[:, None] * ori_jac[orientation_mask, :])
                row += orientation_rows
            # Projection block: one row per axis, matching residual ordering.
            if proj_axes is not None:
                for axis, normal, direction, cost in zip(
                        proj_axes, proj_normals[frame],
                        proj_directions[frame], proj_costs):
                    proj_row = np.zeros(n_joints)
                    if np.isfinite(normal).all():
                        current_axis = rotation @ axis
                        # d(current_axis . n)/dq = (current_axis x n) . jacr.
                        dn = (np.cross(current_axis, normal) @ arm_jacr)
                        n_comp = float(current_axis @ normal)
                        if np.isfinite(direction).all():
                            dd = (np.cross(current_axis, direction) @ arm_jacr)
                            d_raw = float(current_axis @ direction)
                            d_comp = max(d_raw, 1e-8)
                            # arctan2(n, d) with d clamped to 1e-8 in residual.
                            if d_raw <= 1e-8:
                                dd = np.zeros_like(dd)
                            denom = n_comp * n_comp + d_comp * d_comp
                            proj_row = (d_comp * dn - n_comp * dd) / denom
                        else:
                            # arcsin(clip(n, -1, 1)); derivative is 0 when clipped.
                            if -1.0 < n_comp < 1.0:
                                proj_row = dn / np.sqrt(1.0 - n_comp * n_comp)
                    jac[row, col:col + n_joints] = (
                        sqrt_wo * float(cost) / cfg.orientation_scale) * proj_row
                    row += 1
        # Velocity: sqrt(wv) * diff(q, axis=0), C-order (frame-major, joint).
        sqrt_wv = np.sqrt(cfg.velocity_weight)
        for frame in range(max(n_frames - 1, 0)):
            for joint in range(n_joints):
                jac[row, frame * n_joints + joint] = -sqrt_wv
                jac[row, (frame + 1) * n_joints + joint] = sqrt_wv
                row += 1
        # Acceleration: sqrt(wa) * diff(q, n=2, axis=0), coefficients [1,-2,1].
        sqrt_wa = np.sqrt(cfg.acceleration_weight)
        for frame in range(max(n_frames - 2, 0)):
            for joint in range(n_joints):
                jac[row, frame * n_joints + joint] = sqrt_wa
                jac[row, (frame + 1) * n_joints + joint] = -2.0 * sqrt_wa
                jac[row, (frame + 2) * n_joints + joint] = sqrt_wa
                row += 1
        # Home prior: sqrt(wh) * (q - home), diagonal.
        sqrt_wh = np.sqrt(cfg.home_weight)
        for frame in range(n_frames):
            for joint in range(n_joints):
                jac[row, frame * n_joints + joint] = sqrt_wh
                row += 1
        # Soft joint-limit margins. Sub-gradient is zero outside the band.
        sqrt_wm = np.sqrt(cfg.joint_margin_weight)
        soft = np.zeros(n_joints)
        for joint in range(n_joints):
            if np.isfinite(lower[joint]) and np.isfinite(upper[joint]):
                span = upper[joint] - lower[joint]
                candidate = cfg.joint_margin_fraction * span
                soft[joint] = candidate if candidate > 1e-12 else 0.0
        for frame in range(n_frames):
            for joint in range(n_joints):
                if soft[joint] > 0.0 and (
                        lower[joint] + soft[joint] - q[frame, joint]) > 0.0:
                    jac[row, frame * n_joints + joint] = -sqrt_wm / soft[joint]
                row += 1
        for frame in range(n_frames):
            for joint in range(n_joints):
                if soft[joint] > 0.0 and (
                        q[frame, joint] - (upper[joint] - soft[joint])) > 0.0:
                    jac[row, frame * n_joints + joint] = sqrt_wm / soft[joint]
                row += 1
        if row != row_count:
            raise AssertionError(
                f"analytic jacobian rows {row} != residual rows {row_count}")
        return csr_matrix(jac)

    sparsity = _jacobian_sparsity(
        n_frames, n_joints, tracked_frames, orientation_rows, projection_rows)
    if return_functions:
        return {
            "residual": residual,
            "jacobian": _analytic_jacobian,
            "sparsity": sparsity,
            "x0": q0.ravel(),
            "lower": lower,
            "upper": upper,
            "n_frames": n_frames,
            "n_joints": n_joints,
            "row_count": row_count,
        }
    initial_residual = residual(q0.ravel())
    if cfg.use_analytic_jacobian:
        result = least_squares(
            residual, q0.ravel(), bounds=(np.tile(lower, n_frames),
                                          np.tile(upper, n_frames)),
            jac=_analytic_jacobian, method="trf", x_scale="jac",
            max_nfev=int(cfg.max_nfev), ftol=1e-5, xtol=1e-5, gtol=1e-5,
            verbose=0)
    else:
        result = least_squares(
            residual, q0.ravel(), bounds=(np.tile(lower, n_frames),
                                          np.tile(upper, n_frames)),
            jac_sparsity=sparsity, method="trf", x_scale="jac",
            max_nfev=int(cfg.max_nfev), ftol=1e-5, xtol=1e-5, gtol=1e-5,
            verbose=0)
    refined = result.x.reshape(n_frames, n_joints)
    after_pos, after_rot = evaluate_arm_trajectory(
        model, arm_qadr, ee_ref, refined, target_pos, target_rot, mask,
        orientation_axes=orientation_mask, projection_axes=proj_axes,
        projection_normals=proj_normals, projection_directions=proj_directions,
        projection_costs=proj_costs)
    metrics_after = _trajectory_metrics(refined, after_pos, lower, upper)
    initial_cost = 0.5 * float(initial_residual @ initial_residual)
    final_residual = residual(refined.ravel())
    final_cost = 0.5 * float(final_residual @ final_residual)

    mean_regression = (cfg.primary_max_mean_position_regression
                       if cfg.primary else cfg.max_mean_position_regression)
    p95_regression = (cfg.primary_max_p95_position_regression
                      if cfg.primary else cfg.max_p95_position_regression)
    reason = ""
    if not np.isfinite(refined).all():
        reason = "non-finite refined trajectory"
    elif not cfg.primary and final_cost >= initial_cost - 1e-10:
        # In primary mode the optimizer owns the final trajectory; an initial
        # guess that is already optimal legitimately yields refined ~= init.
        reason = "objective did not improve"
    elif (metrics_after["position_mean"] >
          metrics_before["position_mean"] + mean_regression):
        reason = "mean TCP position error regressed"
    elif (metrics_after["position_p95"] >
          metrics_before["position_p95"] + p95_regression):
        reason = "p95 TCP position error regressed"
    accepted = not reason
    return TrajectoryRefinementResult(
        qpos=refined if accepted else q0, accepted=accepted,
        status=int(result.status), nfev=int(result.nfev),
        initial_cost=initial_cost, final_cost=final_cost,
        metrics_before=metrics_before, metrics_after=metrics_after,
        rejection_reason=reason)


def _dual_cross_arm_geom_pairs(model):
    """Return collidable left/right geom pairs in an attached dual model."""
    left = []
    right = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith("left_"):
            left.append(geom_id)
        elif body_name.startswith("right_"):
            right.append(geom_id)

    pairs = []
    for geom_left in left:
        for geom_right in right:
            can_collide = bool(
                (model.geom_contype[geom_left] &
                 model.geom_conaffinity[geom_right]) or
                (model.geom_contype[geom_right] &
                 model.geom_conaffinity[geom_left]))
            if can_collide:
                pairs.append((geom_left, geom_right))
    return tuple(pairs)


def _dual_collision_distances(model, data, geom_pairs, distmax):
    """Return signed pair distances, using ``distmax`` for separated pairs."""
    distances = np.full(len(geom_pairs), float(distmax), dtype=float)
    for index, (geom1, geom2) in enumerate(geom_pairs):
        distance = float(mujoco.mj_geomDistance(
            model, data, geom1, geom2, float(distmax), None))
        if distance < distmax:
            distances[index] = distance
    return distances


def _project_trajectory_positions(model, arm_qadr, arm_ids, ee_ref, arm_q,
                                  target_pos, tracking_mask, target_tolerance,
                                  target_margin=0.8, max_iterations=30,
                                  damping=1e-5, max_joint_step=0.08):
    """Project tracked frames into the TCP-position feasible set.

    This is a local DLS projection initialized from the existing continuous IK
    branch.  It changes only frames outside the inner margin, leaving enough
    room for the later orientation/collision optimizer to move without starting
    exactly on the hard boundary.
    """
    q = np.asarray(arm_q, dtype=float).copy()
    targets = np.asarray(target_pos, dtype=float).reshape(len(q), 3)
    tracked = np.asarray(tracking_mask, dtype=bool).reshape(len(q))
    _, _, _home = _joint_bounds_and_home(model, arm_ids, arm_qadr)
    del _home
    inner_tolerance = float(target_tolerance) * float(target_margin)
    data = mujoco.MjData(model)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    arm_dof = np.asarray(
        [int(model.jnt_dofadr[int(joint)]) for joint in arm_ids], dtype=int)
    projected = 0
    failed = []
    for frame in np.flatnonzero(tracked):
        _set_arm_configuration(model, data, arm_qadr, q[frame])
        position, _ = _ee_pose(model, data, ee_ref)
        initial_error = float(np.linalg.norm(targets[frame] - position))
        if initial_error <= inner_tolerance:
            continue
        for _ in range(int(max_iterations)):
            position, _ = _ee_pose_and_jacobian(
                model, data, ee_ref, jacp, jacr)
            error = targets[frame] - position
            current_error = float(np.linalg.norm(error))
            if current_error <= inner_tolerance:
                break
            J = jacp[:, arm_dof]
            dq = J.T @ np.linalg.solve(
                J @ J.T + float(damping) * np.eye(3), error)
            max_abs = float(np.max(np.abs(dq), initial=0.0))
            if max_abs > max_joint_step:
                dq *= float(max_joint_step) / max_abs
            previous_q = data.qpos.copy()
            accepted_step = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                data.qpos[:] = previous_q
                for index, (qpos_address, joint_id) in enumerate(
                        zip(arm_qadr, arm_ids)):
                    value = previous_q[qpos_address] + scale * dq[index]
                    if model.jnt_limited[joint_id]:
                        value = np.clip(value, *model.jnt_range[joint_id])
                    data.qpos[qpos_address] = value
                mujoco.mj_forward(model, data)
                trial_position, _ = _ee_pose(model, data, ee_ref)
                if np.linalg.norm(targets[frame] - trial_position) < current_error:
                    accepted_step = True
                    break
            if not accepted_step:
                data.qpos[:] = previous_q
                mujoco.mj_forward(model, data)
                break
        position, _ = _ee_pose(model, data, ee_ref)
        final_error = float(np.linalg.norm(targets[frame] - position))
        if final_error <= target_tolerance:
            q[frame] = data.qpos[np.asarray(arm_qadr, dtype=int)]
            projected += 1
        else:
            failed.append((int(frame), initial_error, final_error))
    return q, {
        "projected_frames": projected,
        "failed_frames": failed,
        "inner_tolerance": inner_tolerance,
    }


def _restore_position_feasibility(model, arm_qadr, arm_ids, ee_ref, candidate_q,
                                  fallback_q, target_pos, tracking_mask,
                                  tolerance):
    """Project violating candidate frames, then restore only failures."""
    repaired, stats = _project_trajectory_positions(
        model, arm_qadr, arm_ids, ee_ref, candidate_q, target_pos,
        tracking_mask, tolerance, target_margin=0.98, max_iterations=50)
    restored = []
    for frame, _before, _after in stats["failed_frames"]:
        repaired[frame] = fallback_q[frame]
        restored.append(frame)
    stats = {**stats, "restored_frames": restored}
    return repaired, stats


def refine_dual_arm_trajectory(
        single_model, dual_model, single_arm_qadr, single_arm_ids, ee_ref,
        dual_left_qadr, dual_right_qadr, initial_left_q, initial_right_q,
        left_target_pos, left_target_rot, right_target_pos, right_target_rot,
        left_tracking_mask, right_tracking_mask, config=None,
        orientation_cost=(1.0, 1.0, 1.0),
        left_projection_axes=None, left_projection_normals=None,
        left_projection_directions=None, left_projection_costs=None,
        right_projection_axes=None, right_projection_normals=None,
        right_projection_directions=None, right_projection_costs=None,
        baseline_qpos=None, return_functions=False):
    """Jointly refine two complete arm trajectories with collision avoidance.

    The per-arm tracking and regularization blocks are reused verbatim from
    :func:`refine_arm_trajectory`.  Collision residuals couple only the left and
    right variables at the same frame, preserving a block-sparse Jacobian.
    """
    cfg = config or TrajectoryRefinementConfig()
    cfg.validate()
    q_left = np.asarray(initial_left_q, dtype=float)
    q_right = np.asarray(initial_right_q, dtype=float)
    if q_left.shape != q_right.shape or q_left.ndim != 2:
        raise ValueError(
            "initial_left_q and initial_right_q must have matching 2D shapes")
    n_frames, n_joints = q_left.shape
    if n_joints != len(single_arm_qadr):
        raise ValueError("arm trajectory width does not match arm qpos addresses")
    q_left, left_projection = _project_trajectory_positions(
        single_model, single_arm_qadr, single_arm_ids, ee_ref, q_left,
        left_target_pos, left_tracking_mask, cfg.position_tolerance)
    q_right, right_projection = _project_trajectory_positions(
        single_model, single_arm_qadr, single_arm_ids, ee_ref, q_right,
        right_target_pos, right_tracking_mask, cfg.position_tolerance)
    projection_failures = (left_projection["failed_frames"] +
                           right_projection["failed_frames"])

    side_cfg = TrajectoryRefinementConfig(**{
        **cfg.__dict__, "collision_weight": 0.0,
        "use_analytic_jacobian": True,
    })
    left_funcs = refine_arm_trajectory(
        single_model, single_arm_qadr, single_arm_ids, ee_ref, q_left,
        left_target_pos, left_target_rot, left_tracking_mask, config=side_cfg,
        orientation_cost=orientation_cost, projection_axes=left_projection_axes,
        projection_normals=left_projection_normals,
        projection_directions=left_projection_directions,
        projection_costs=left_projection_costs, return_functions=True)
    right_funcs = refine_arm_trajectory(
        single_model, single_arm_qadr, single_arm_ids, ee_ref, q_right,
        right_target_pos, right_target_rot, right_tracking_mask, config=side_cfg,
        orientation_cost=orientation_cost, projection_axes=right_projection_axes,
        projection_normals=right_projection_normals,
        projection_directions=right_projection_directions,
        projection_costs=right_projection_costs, return_functions=True)
    orientation_rows = (int(np.count_nonzero(np.asarray(orientation_cost) > 0.0))
                        if cfg.orientation_weight > 0.0 else 0)
    left_frame_rows = (3 + orientation_rows +
                       (len(left_projection_axes)
                        if left_projection_axes is not None else 0))
    right_frame_rows = (3 + orientation_rows +
                        (len(right_projection_axes)
                         if right_projection_axes is not None else 0))
    left_position_rows = np.concatenate([
        np.arange(frame * left_frame_rows, frame * left_frame_rows + 3)
        for frame in range(int(np.count_nonzero(left_tracking_mask)))
    ])
    right_position_rows = np.concatenate([
        np.arange(frame * right_frame_rows, frame * right_frame_rows + 3)
        for frame in range(int(np.count_nonzero(right_tracking_mask)))
    ])

    left_cols = n_frames * n_joints
    x0 = np.concatenate([left_funcs["x0"], right_funcs["x0"]])
    lower = np.concatenate([
        np.tile(left_funcs["lower"], n_frames),
        np.tile(right_funcs["lower"], n_frames),
    ])
    upper = np.concatenate([
        np.tile(left_funcs["upper"], n_frames),
        np.tile(right_funcs["upper"], n_frames),
    ])
    geom_pairs = _dual_cross_arm_geom_pairs(dual_model)
    collision_rows = n_frames * len(geom_pairs) if cfg.collision_weight > 0.0 else 0
    base_qpos = (np.asarray(baseline_qpos, dtype=float).copy()
                 if baseline_qpos is not None else dual_model.qpos0.copy())
    if base_qpos.shape != (dual_model.nq,):
        raise ValueError("baseline_qpos must match dual_model.nq")
    dual_data = mujoco.MjData(dual_model)
    # One scalar minimum-distance residual per frame is enough to enforce the
    # cross-arm clearance. Keeping hundreds of inactive pair rows dilutes the
    # finite-difference trust-region step and lets the active pair change while
    # the optimizer is sampling a column.
    collision_rows = n_frames if collision_rows else 0
    sqrt_collision = np.sqrt(cfg.collision_weight)
    distmax = cfg.collision_safe_distance

    def collision_residual(flat):
        if not collision_rows:
            return np.empty(0, dtype=float)
        left_q = flat[:left_cols].reshape(n_frames, n_joints)
        right_q = flat[left_cols:].reshape(n_frames, n_joints)
        values = np.empty(n_frames, dtype=float)
        for frame in range(n_frames):
            dual_data.qpos[:] = base_qpos
            dual_data.qpos[np.asarray(dual_left_qadr, dtype=int)] = left_q[frame]
            dual_data.qpos[np.asarray(dual_right_qadr, dtype=int)] = right_q[frame]
            mujoco.mj_forward(dual_model, dual_data)
            distances = _dual_collision_distances(
                dual_model, dual_data, geom_pairs, distmax)
            minimum_distance = float(np.min(distances, initial=distmax))
            values[frame] = (sqrt_collision *
                             max(0.0, distmax - minimum_distance) /
                             cfg.collision_scale)
        return values

    def residual(flat):
        return np.concatenate([
            left_funcs["residual"](flat[:left_cols]),
            right_funcs["residual"](flat[left_cols:]),
            collision_residual(flat),
        ])

    left_rows = left_funcs["row_count"]
    right_rows = right_funcs["row_count"]
    row_count = left_rows + right_rows + collision_rows
    sparsity = lil_matrix((row_count, 2 * left_cols), dtype=np.int8)
    sparsity[:left_rows, :left_cols] = left_funcs["sparsity"]
    sparsity[left_rows:left_rows + right_rows, left_cols:] = right_funcs["sparsity"]
    collision_start = left_rows + right_rows
    for frame in range(n_frames):
        row0 = collision_start + frame
        row1 = row0 + 1
        left_col = frame * n_joints
        right_col = left_cols + frame * n_joints
        sparsity[row0:row1, left_col:left_col + n_joints] = 1
        sparsity[row0:row1, right_col:right_col + n_joints] = 1
    sparsity = sparsity.tocsr()

    if return_functions:
        return {
            "residual": residual, "sparsity": sparsity, "x0": x0,
            "lower": lower, "upper": upper, "row_count": row_count,
            "collision_rows": collision_rows, "geom_pairs": geom_pairs,
        }

    # Position is a feasibility constraint, not a zero-error tracking objective.
    # A high-gain exact penalty is zero throughout the feasible set and activates
    # only when a frame exceeds the configured TCP tolerance. This keeps the
    # sparse least-squares solve practical for thousand-variable trajectories
    # while making position unavailable as a tradeoff inside the 5 mm ball.
    def objective_residual(flat):
        left_values = left_funcs["residual"](flat[:left_cols])
        right_values = right_funcs["residual"](flat[left_cols:])
        left_keep = np.ones(len(left_values), dtype=bool)
        right_keep = np.ones(len(right_values), dtype=bool)
        left_keep[left_position_rows] = False
        right_keep[right_position_rows] = False
        return np.concatenate([
            left_values[left_keep], right_values[right_keep],
            collision_residual(flat),
        ])

    left_objective_rows = left_funcs["row_count"] - len(left_position_rows)
    right_objective_rows = right_funcs["row_count"] - len(right_position_rows)
    objective_row_count = (left_objective_rows + right_objective_rows +
                           collision_rows)
    objective_sparsity = lil_matrix(
        (objective_row_count, 2 * left_cols), dtype=np.int8)
    left_keep_rows = np.ones(left_funcs["row_count"], dtype=bool)
    right_keep_rows = np.ones(right_funcs["row_count"], dtype=bool)
    left_keep_rows[left_position_rows] = False
    right_keep_rows[right_position_rows] = False
    objective_sparsity[:left_objective_rows, :left_cols] = (
        left_funcs["sparsity"][left_keep_rows])
    objective_sparsity[left_objective_rows:
                       left_objective_rows + right_objective_rows,
                       left_cols:] = right_funcs["sparsity"][right_keep_rows]
    collision_start_objective = left_objective_rows + right_objective_rows
    if collision_rows:
        for frame in range(n_frames):
            row0 = collision_start_objective + frame
            left_col = frame * n_joints
            right_col = left_cols + frame * n_joints
            objective_sparsity[row0, left_col:left_col + n_joints] = 1
            objective_sparsity[row0, right_col:right_col + n_joints] = 1

    left_tracked = np.flatnonzero(np.asarray(left_tracking_mask, dtype=bool))
    right_tracked = np.flatnonzero(np.asarray(right_tracking_mask, dtype=bool))
    position_data = mujoco.MjData(single_model)

    def position_constraint(flat):
        left_q = flat[:left_cols].reshape(n_frames, n_joints)
        right_q = flat[left_cols:].reshape(n_frames, n_joints)
        errors = []
        for q, targets, frames in (
                (left_q, left_target_pos, left_tracked),
                (right_q, right_target_pos, right_tracked)):
            for frame in frames:
                _set_arm_configuration(
                    single_model, position_data, single_arm_qadr, q[frame])
                position, _ = _ee_pose(single_model, position_data, ee_ref)
                errors.append(float(np.linalg.norm(position - targets[frame])))
        return np.asarray(errors, dtype=float)

    def position_violation_residual(flat):
        errors = position_constraint(flat)
        return (np.sqrt(cfg.position_constraint_weight) *
                np.maximum(0.0, errors - cfg.position_tolerance) /
                cfg.position_tolerance)

    def constrained_residual(flat):
        return np.concatenate([
            objective_residual(flat), position_violation_residual(flat)])

    constraint_rows = len(left_tracked) + len(right_tracked)
    constrained_sparsity = lil_matrix(
        (objective_row_count + constraint_rows, 2 * left_cols), dtype=np.int8)
    constrained_sparsity[:objective_row_count] = objective_sparsity
    row = objective_row_count
    for frame in left_tracked:
        col = int(frame) * n_joints
        constrained_sparsity[row, col:col + n_joints] = 1
        row += 1
    for frame in right_tracked:
        col = left_cols + int(frame) * n_joints
        constrained_sparsity[row, col:col + n_joints] = 1
        row += 1
    constrained_sparsity = constrained_sparsity.tocsr()

    initial_residual = constrained_residual(x0)
    result = least_squares(
        constrained_residual, x0, bounds=(lower, upper),
        jac_sparsity=constrained_sparsity, method="trf", x_scale="jac",
        max_nfev=int(cfg.max_nfev), ftol=1e-5, xtol=1e-5, gtol=1e-5,
        verbose=0)
    refined = result.x
    final_residual = constrained_residual(refined)

    refined_left = refined[:left_cols].reshape(n_frames, n_joints)
    refined_right = refined[left_cols:].reshape(n_frames, n_joints)
    refined_left, left_restore = _restore_position_feasibility(
        single_model, single_arm_qadr, single_arm_ids, ee_ref, refined_left,
        q_left, left_target_pos, left_tracking_mask, cfg.position_tolerance)
    refined_right, right_restore = _restore_position_feasibility(
        single_model, single_arm_qadr, single_arm_ids, ee_ref, refined_right,
        q_right, right_target_pos, right_tracking_mask, cfg.position_tolerance)
    refined = np.concatenate([refined_left.ravel(), refined_right.ravel()])
    final_residual = constrained_residual(refined)
    before_left_pos, _ = evaluate_arm_trajectory(
        single_model, single_arm_qadr, ee_ref, q_left, left_target_pos,
        left_target_rot, left_tracking_mask,
        orientation_axes=np.asarray(orientation_cost) > 0.0,
        projection_axes=left_projection_axes,
        projection_normals=left_projection_normals,
        projection_directions=left_projection_directions,
        projection_costs=left_projection_costs)
    before_right_pos, _ = evaluate_arm_trajectory(
        single_model, single_arm_qadr, ee_ref, q_right, right_target_pos,
        right_target_rot, right_tracking_mask,
        orientation_axes=np.asarray(orientation_cost) > 0.0,
        projection_axes=right_projection_axes,
        projection_normals=right_projection_normals,
        projection_directions=right_projection_directions,
        projection_costs=right_projection_costs)
    after_left_pos, _ = evaluate_arm_trajectory(
        single_model, single_arm_qadr, ee_ref, refined_left, left_target_pos,
        left_target_rot, left_tracking_mask,
        orientation_axes=np.asarray(orientation_cost) > 0.0,
        projection_axes=left_projection_axes,
        projection_normals=left_projection_normals,
        projection_directions=left_projection_directions,
        projection_costs=left_projection_costs)
    after_right_pos, _ = evaluate_arm_trajectory(
        single_model, single_arm_qadr, ee_ref, refined_right, right_target_pos,
        right_target_rot, right_tracking_mask,
        orientation_axes=np.asarray(orientation_cost) > 0.0,
        projection_axes=right_projection_axes,
        projection_normals=right_projection_normals,
        projection_directions=right_projection_directions,
        projection_costs=right_projection_costs)

    def finite_stats(values):
        values = np.asarray(values)[np.isfinite(values)]
        return (float(np.mean(values)), float(np.quantile(values, 0.95)))

    before_stats = [finite_stats(before_left_pos), finite_stats(before_right_pos)]
    after_stats = [finite_stats(after_left_pos), finite_stats(after_right_pos)]
    max_position_error = float(np.max(position_constraint(refined), initial=0.0))
    accepted = (not projection_failures and np.isfinite(refined).all() and
                max_position_error <= cfg.position_tolerance + 1e-6)
    return {
        "left_qpos": refined_left if accepted else q_left.copy(),
        "right_qpos": refined_right if accepted else q_right.copy(),
        "accepted": accepted,
        "rejection_reason": ("" if accepted else
                             (f"position projection failed on "
                              f"{len(projection_failures)} frames"
                              if projection_failures else
                              f"TCP tolerance violated: {max_position_error*1000:.2f}mm")),
        "status": int(result.status), "nfev": int(result.nfev),
        "initial_cost": 0.5 * float(initial_residual @ initial_residual),
        "final_cost": 0.5 * float(final_residual @ final_residual),
        "initial_collision_cost": 0.5 * float(
            collision_residual(x0) @ collision_residual(x0)),
        "final_collision_cost": 0.5 * float(
            collision_residual(refined) @ collision_residual(refined)),
        "geom_pair_count": len(geom_pairs),
        "position_metrics_before": before_stats,
        "position_metrics_after": after_stats,
        "max_position_error": max_position_error,
        "position_tolerance": cfg.position_tolerance,
        "position_projection_left": left_projection,
        "position_projection_right": right_projection,
        "position_restore_left": left_restore,
        "position_restore_right": right_restore,
    }
