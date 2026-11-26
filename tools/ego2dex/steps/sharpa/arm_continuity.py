# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Whole-source-frame bounds for iterative Mink IK, including its fallback QP."""

import numpy as np
from scipy.spatial.transform import Rotation
from mink.limits import Constraint


class FrameConfigurationLimit:
    """Combine original joint limits with a budget fixed for an entire frame.

    Mink inequalities constrain delta-q, not velocity. Recomputing remaining
    slack on every iteration prevents repeated QP steps spending the budget twice.
    """

    def __init__(self, original, model, qadr):
        self.original = original
        self.qadr = np.asarray(qadr, dtype=int)
        joints = [int(np.flatnonzero(model.jnt_qposadr == a)[0]) for a in self.qadr]
        self.dof = np.asarray(model.jnt_dofadr[joints], dtype=int)
        self.G = np.eye(model.nv)[self.dof]
        self.lower = self.upper = None

    def clear(self):
        self.lower = self.upper = None

    def set_previous(self, previous, source_dt, max_speed=3.0):
        if not np.isfinite(source_dt) or source_dt <= 0:
            raise ValueError("Source timestamps must be finite and strictly increasing")
        previous = np.asarray(previous, dtype=float)
        if previous.shape != self.qadr.shape or not np.isfinite(previous).all():
            raise ValueError("Invalid previous arm configuration")
        budget = max(0.0, max_speed * source_dt - 1e-7)
        self.lower, self.upper = previous - budget, previous + budget

    def compute_qp_inequalities(self, configuration, dt):
        original = self.original.compute_qp_inequalities(configuration, dt)
        if self.lower is None:
            return original
        current = configuration.q[self.qadr]
        matrices = [self.G, -self.G]
        bounds = [self.upper - current, current - self.lower]
        if original.G is not None:
            matrices.insert(0, original.G)
            bounds.insert(0, original.h)
        return Constraint(G=np.vstack(matrices), h=np.concatenate(bounds))


def bounded_trajectory(solver, rows, times, target_fn, placement, home, arm, fingers, site_id, anchor_frame):
    """Solve from one offline anchor, then propagate in both time directions."""
    import mujoco

    limit = solver.config_limit
    assert isinstance(limit, FrameConfigurationLimit)
    times = np.asarray(times)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must be strictly increasing")
    limit.clear()
    solver.reset()
    targets = [target_fn(row, placement) for row in rows]
    p, r = targets[anchor_frame]
    seed, _, _, _ = solver.solve(p, r, rows[anchor_frame, 6:28], max_iters=250)
    result = np.tile(home, (len(rows), 1))
    result[:, fingers] = rows[:, 6:28]
    result[anchor_frame, arm] = seed
    for direction in [1, -1]:
        # Restore the same solved anchor before the backward sweep.
        solver.configuration.update(q=result[anchor_frame])
        stop = len(rows) if direction == 1 else -1
        for i in range(anchor_frame + direction, stop, direction):
            previous = i - direction
            limit.set_previous(result[previous, arm], abs(times[i] - times[previous]))
            p, r = targets[i]
            sol, _, _, _ = solver.solve(p, r, rows[i, 6:28], max_iters=64)
            result[i, arm] = sol
    limit.clear()
    probe = mujoco.MjData(solver.model)
    errors, rotations, clearance = [], [], 0.2
    pairs = solver.collision_limit.geom_id_pairs
    for i, row in enumerate(result):
        probe.qpos[:] = row
        mujoco.mj_forward(solver.model, probe)
        site = probe.site(site_id)
        p, r = targets[i]
        errors.append(float(np.linalg.norm(site.xpos - p)))
        desired = Rotation.from_quat(np.asarray(r)[[1, 2, 3, 0]])
        rotations.append(float((desired.inv() * Rotation.from_matrix(site.xmat.reshape(3, 3))).magnitude()))
        for pose in [row] + ([(row + result[i - 1]) / 2] if i else []):
            probe.qpos[:] = pose
            mujoco.mj_forward(solver.model, probe)
            for a, b in pairs:
                clearance = min(
                    clearance, float(mujoco.mj_geomDistance(solver.model, probe, int(a), int(b), 0.2, None))
                )
    speed = float(np.max(np.abs(np.diff(result[:, arm], axis=0) / np.diff(times)[:, None])))
    metrics = dict(
        workspace=list(placement),
        anchor_frame=int(anchor_frame),
        position_max_m=max(errors),
        rotation_max_rad=max(rotations),
        arm_speed_max_rad_s=speed,
        collision_min_clearance_m=clearance,
        collision_qp_fallbacks=int(solver.qp_failures),
    )
    # Rank by the same independent gates, never by coarse pose errors alone.
    ratios = [max(errors) / 0.005, max(rotations) / 0.1, speed / 3.0]
    metrics["score"] = max(ratios) + 1000 * max(0.0, -clearance)
    metrics["candidate_passed"] = bool(max(ratios) <= 1 and clearance >= 0)
    return result, targets, metrics
