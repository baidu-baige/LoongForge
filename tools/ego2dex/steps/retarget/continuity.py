# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Continuous, collision-aware reconstruction of an entire dual-arm IK track.

Every solve is bounded relative to its adjacent accepted configuration. An
episode uses one branch propagated in both directions from a seed, rather than
independently choosing a collision-free branch at each frame.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .base_search import _is_known_baseline_self_contact, _quality_collision_metrics


@dataclass(frozen=True)
class ContinuityConfig:
    max_velocity: float = 3.0
    position_tolerance: float = 0.005
    clearance: float = 0.003
    position_weight: float = 100.0
    orientation_weight: float = 0.5
    velocity_weight: float = 0.12
    acceleration_weight: float = 0.35
    collision_weight: float = 1000.0
    max_nfev: int = 60
    max_seeds: int = 3

    def validate(self, fps):
        for key, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if int(self.max_seeds) != self.max_seeds or int(self.max_nfev) != self.max_nfev:
            raise ValueError("max_seeds and max_nfev must be integers")


def _hat(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _log_jacobian(v):
    angle = np.linalg.norm(v)
    skew = _hat(v)
    coefficient = 1 / 12 if angle < 1e-5 else (1 - 0.5 * angle / np.tan(0.5 * angle)) / angle**2
    return np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)


class ContinuousProblem:
    def __init__(self, model, qadr, ee_refs, targets_p, targets_R, masks, fps, config=None):
        self.model = model
        self.qadr = np.asarray(qadr, dtype=int)
        self.refs = ee_refs
        self.targets_p = np.asarray(targets_p)
        self.targets_R = np.asarray(targets_R)
        self.masks = np.asarray(masks, dtype=bool)
        self.cfg = config or ContinuityConfig()
        self.cfg.validate(fps)
        n = len(self.targets_p)
        if self.targets_p.shape != (n, 2, 3) or self.targets_R.shape != (n, 2, 3, 3) or self.masks.shape != (n, 2):
            raise ValueError("Targets must have shapes (N,2,3), (N,2,3,3), and (N,2)")
        if (
            not self.masks.any()
            or not np.isfinite(self.targets_p[self.masks]).all()
            or not np.isfinite(self.targets_R[self.masks]).all()
        ):
            raise ValueError("Finite observed-hand targets are required")
        self.fps = float(fps)
        self.step = self.cfg.max_velocity / fps
        self.data = mujoco.MjData(model)
        self.jids = np.array([next(j for j in range(model.njnt) if model.jnt_qposadr[j] == q) for q in self.qadr])
        if not np.all(model.jnt_type[self.jids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Motion continuity currently supports hinge arm joints")
        self.vadr = model.jnt_dofadr[self.jids]
        self.lower = np.where(model.jnt_limited[self.jids], model.jnt_range[self.jids, 0] + 1e-7, -np.inf)
        self.upper = np.where(model.jnt_limited[self.jids], model.jnt_range[self.jids, 1] - 1e-7, np.inf)
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros_like(self.jp)
        self.jp2 = np.zeros_like(self.jp)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
        eligible = [
            g
            for g in range(model.ngeom)
            if names[model.geom_bodyid[g]].startswith(("left_", "right_"))
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        pairs = []
        for i, g1 in enumerate(eligible):
            for g2 in eligible[i + 1 :]:
                b1, b2 = model.geom_bodyid[[g1, g2]]
                if b1 == b2 or _is_known_baseline_self_contact(names[b1], names[b2]):
                    continue
                if (model.geom_contype[g1] & model.geom_conaffinity[g2]) or (
                    model.geom_contype[g2] & model.geom_conaffinity[g1]
                ):
                    pairs.append((g1, g2))
        self.pairs = {pair: i for i, pair in enumerate(pairs)}
        self.joint_count = len(qadr)
        self.rows = 12 + 2 * self.joint_count + len(pairs)
        self.cache_x = None
        self.collision_residual_clearance = self.cfg.clearance
        self.stable_box_clearance = False
        self.box_pair_clearance = {}
        self.box_pairs = {
            pair: index
            for pair, index in self.pairs.items()
            if all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in pair)
            and names[model.geom_bodyid[pair[0]]].startswith("left_")
            != names[model.geom_bodyid[pair[1]]].startswith("left_")
        }
        self.jr2 = np.zeros_like(self.jp)

    def frame(self, t, full_q, previous=None, previous2=None):
        self.t = t
        self.data.qpos[:] = full_q
        self.reference = full_q[self.qadr].copy() if previous is None else np.asarray(previous).copy()
        self.predicted = self.reference if previous2 is None else 2 * self.reference - np.asarray(previous2)
        self.cache_x = None

    def evaluate(self, q):
        if self.cache_x is not None and np.array_equal(q, self.cache_x):
            return self.cached
        cfg = self.cfg
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        r = np.zeros(self.rows)
        jac = np.zeros((self.rows, self.joint_count))
        for side, (kind, ref_id) in enumerate(self.refs):
            if not self.masks[self.t, side]:
                continue
            if kind == "site":
                p = self.data.site_xpos[ref_id]
                R = self.data.site_xmat[ref_id].reshape(3, 3)
                mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, ref_id)
            else:
                p = self.data.xpos[ref_id]
                R = self.data.xmat[ref_id].reshape(3, 3)
                mujoco.mj_jacBody(self.model, self.data, self.jp, self.jr, ref_id)
            start = side * 6
            target_R = self.targets_R[self.t, side]
            rotation = Rotation.from_matrix(target_R.T @ R).as_rotvec()
            r[start : start + 3] = cfg.position_weight * (p - self.targets_p[self.t, side])
            r[start + 3 : start + 6] = cfg.orientation_weight * rotation
            jac[start : start + 3] = cfg.position_weight * self.jp[:, self.vadr]
            jac[start + 3 : start + 6] = (
                cfg.orientation_weight * _log_jacobian(rotation) @ target_R.T @ self.jr[:, self.vadr]
            )
        j = self.joint_count
        # Scaled for frame rate: the same motion has comparable temporal costs.
        rate = self.fps / 30
        r[12 : 12 + j] = cfg.velocity_weight * rate * (q - self.reference)
        jac[12 : 12 + j] = np.eye(j) * cfg.velocity_weight * rate
        r[12 + j : 12 + 2 * j] = cfg.acceleration_weight * rate**2 * (q - self.predicted)
        jac[12 + j : 12 + 2 * j] = np.eye(j) * cfg.acceleration_weight * rate**2
        for c in self.data.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            if self.stable_box_clearance and pair in self.box_pairs:
                continue
            index = self.pairs.get(pair)
            if index is None or c.dist >= self.collision_residual_clearance:
                continue
            row = 12 + 2 * j + index
            depth = self.collision_residual_clearance - float(c.dist)
            slope = 1.0
            if self.collision_residual_clearance == 0.0:
                # C1 hinge at first contact; this only smooths the optimizer
                # residual. Final validation still rejects every penetration.
                transition = 1e-4
                slope = min(depth / transition, 1.0)
                depth = depth * depth / (2 * transition) if depth < transition else depth - transition / 2
            penalty = cfg.collision_weight * depth
            if penalty <= r[row]:
                continue
            normal = c.frame[:3]
            p1 = c.pos - 0.5 * c.dist * normal
            p2 = c.pos + 0.5 * c.dist * normal
            b1, b2 = self.model.geom_bodyid[[c.geom1, c.geom2]]
            mujoco.mj_jac(self.model, self.data, self.jp, None, p1, b1)
            mujoco.mj_jac(self.model, self.data, self.jp2, None, p2, b2)
            r[row] = penalty
            jac[row] = -cfg.collision_weight * slope * normal @ (self.jp2 - self.jp)[:, self.vadr]
        if self.stable_box_clearance:
            self._box_clearance(r, jac)
        self.cache_x = q.copy()
        self.cached = (r, jac)
        return self.cached

    def _box_clearance(self, residual, jacobian):
        from .box_clearance import box_gap_gradient

        for (g1, g2), index in self.box_pairs.items():
            clearance = self.box_pair_clearance.get((g1, g2), self.cfg.clearance)
            p1, p2 = self.data.geom_xpos[[g1, g2]]
            radius = self.model.geom_rbound[g1] + self.model.geom_rbound[g2] + clearance
            if np.linalg.norm(p2 - p1) > radius:
                continue
            b1, b2 = self.model.geom_bodyid[[g1, g2]]
            mujoco.mj_jac(self.model, self.data, self.jp, self.jr, p1, b1)
            mujoco.mj_jac(self.model, self.data, self.jp2, self.jr2, p2, b2)
            gap, gradient = box_gap_gradient(
                p1,
                self.data.geom_xmat[g1].reshape(3, 3),
                self.model.geom_size[g1],
                p2,
                self.data.geom_xmat[g2].reshape(3, 3),
                self.model.geom_size[g2],
                self.jp[:, self.vadr],
                self.jr[:, self.vadr],
                self.jp2[:, self.vadr],
                self.jr2[:, self.vadr],
            )
            if gap >= clearance:
                continue
            row = 12 + 2 * self.joint_count + index
            residual[row] = self.cfg.collision_weight * (clearance - gap)
            jacobian[row] = -self.cfg.collision_weight * gradient

    def solve(self, seed, bounded=True):
        lower, upper = self.lower.copy(), self.upper.copy()
        if bounded:
            lower = np.maximum(lower, self.reference - self.step)
            upper = np.minimum(upper, self.reference + self.step)
        x0 = np.clip(seed, lower + 1e-9, upper - 1e-9)
        return least_squares(
            lambda x: self.evaluate(x)[0],
            x0,
            jac=lambda x: self.evaluate(x)[1],
            bounds=(lower, upper),
            max_nfev=self.cfg.max_nfev,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        ).x

    def measure(self, track):
        position = np.full((len(track), 2), np.nan)
        rotation = position.copy()
        collisions = []
        for t, q in enumerate(track):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            collisions.append(_quality_collision_metrics(self.model, self.data))
            for s, (kind, idx) in enumerate(self.refs):
                if not self.masks[t, s]:
                    continue
                p = self.data.site_xpos[idx] if kind == "site" else self.data.xpos[idx]
                R = (self.data.site_xmat[idx] if kind == "site" else self.data.xmat[idx]).reshape(3, 3)
                position[t, s] = np.linalg.norm(p - self.targets_p[t, s])
                rotation[t, s] = Rotation.from_matrix(self.targets_R[t, s].T @ R).magnitude()
        delta = np.diff(track[:, self.qadr], axis=0)
        accel = np.diff(track[:, self.qadr], n=2, axis=0)
        midpoint_collisions = 0
        for t in range(1, len(track)):
            self.data.qpos[:] = 0.5 * (track[t - 1] + track[t])
            mujoco.mj_forward(self.model, self.data)
            c = _quality_collision_metrics(self.model, self.data)
            midpoint_collisions += bool(c["self_collision"] or c["cross_arm_contact_count"])
        bad_position = np.any(np.isfinite(position) & (position > self.cfg.position_tolerance), axis=1)

        def means(values, scale):
            return [
                float(np.mean(values[self.masks[:, s], s]) * scale) if self.masks[:, s].any() else None
                for s in range(2)
            ]

        report = {
            "position_mean_mm": means(position, 1000),
            "position_max_m": float(np.nanmax(position)),
            "position_failed_frames": int(bad_position.sum()),
            "orientation_mean_deg": means(rotation, 180 / np.pi),
            "max_joint_step_rad": float(np.max(abs(delta), initial=0)),
            "max_velocity_rad_s": float(np.max(abs(delta), initial=0) * self.fps),
            "velocity_rms_rad_s": float(np.sqrt(np.mean(delta**2)) * self.fps) if delta.size else 0.0,
            "acceleration_rms_rad_s2": float(np.sqrt(np.mean(accel**2)) * self.fps**2) if accel.size else 0.0,
            "max_acceleration_rad_s2": float(np.max(abs(accel), initial=0) * self.fps**2),
            "self_collision_frames": sum(c["self_collision"] for c in collisions),
            "cross_arm_collision_frames": sum(c["cross_arm_contact_count"] > 0 for c in collisions),
            "midpoint_collision_intervals": int(midpoint_collisions),
            "joint_limit_violation_frames": int(
                np.any(
                    (track[:, self.qadr] < self.lower - 1e-6) | (track[:, self.qadr] > self.upper + 1e-6), axis=1
                ).sum()
            ),
        }
        report["accepted"] = bool(
            not report["position_failed_frames"]
            and not report["self_collision_frames"]
            and not report["cross_arm_collision_frames"]
            and not midpoint_collisions
            and not report["joint_limit_violation_frames"]
            and report["max_joint_step_rad"] <= self.step + 1e-6
        )
        return report, position, rotation


def reconstruct_continuous_trajectory(model, initial, qadr, ee_refs, targets_p, targets_R, masks, fps, config=None):
    cfg = config or ContinuityConfig()
    problem = ContinuousProblem(model, qadr, ee_refs, targets_p, targets_R, masks, fps, cfg)
    initial = np.asarray(initial, dtype=float)
    if not len(initial) or initial.shape != (len(targets_p), model.nq) or not np.isfinite(initial).all():
        raise ValueError("A non-empty finite initial joint trajectory is required")
    original_margins = model.geom_margin.copy()
    candidates = []
    try:
        # Broadphase near contacts provide clearance gradients before penetration.
        model.geom_margin[:] = np.maximum(original_margins, cfg.clearance)
        anchors = list(dict.fromkeys([len(initial) // 2, len(initial) // 4, 3 * len(initial) // 4]))
        for anchor in anchors[: cfg.max_seeds]:
            track = initial.copy()
            problem.frame(anchor, track[anchor])
            track[anchor, problem.qadr] = problem.solve(track[anchor, problem.qadr], bounded=False)
            for direction in (-1, 1):
                for t in range(anchor + direction, len(track) if direction > 0 else -1, direction):
                    previous = track[t - direction, problem.qadr]
                    previous2 = track[t - 2 * direction, problem.qadr] if abs(t - anchor) > 1 else None
                    problem.frame(t, track[t], previous, previous2)
                    track[t, problem.qadr] = problem.solve(previous)
            report, pos, rot = problem.measure(track)
            report["anchor_frame"] = anchor
            print(f"    continuity seed {anchor}: {report}", flush=True)
            candidates.append((track, report, pos, rot))
            if report["accepted"]:
                break
        selected = min(
            candidates,
            key=lambda c: (
                not c[1]["accepted"],
                c[1]["self_collision_frames"] + c[1]["cross_arm_collision_frames"],
                c[1]["position_failed_frames"],
                c[1]["position_max_m"],
                c[1]["acceleration_rms_rad_s2"],
            ),
        )
        track, report, pos, rot = selected
        return (
            track,
            {**report, "seed_attempts": [c[1] for c in candidates], "config": vars(cfg), "fps": float(fps)},
            pos,
            rot,
        )
    finally:
        model.geom_margin[:] = original_margins
