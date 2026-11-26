# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Joint optimization of pose and temporal residuals over a complete trajectory."""

from dataclasses import replace
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from .continuity import ContinuousProblem, ContinuityConfig


class TrajectoryProblem:
    def __init__(
        self,
        model,
        initial,
        qadr,
        refs,
        p,
        R,
        masks,
        fps,
        orientation_weight=2.0,
        acceleration_weight=2.0,
        config=None,
        angular_guard=None,
        midpoint_collisions=False,
    ):
        self.initial = np.array(initial, copy=True)
        self.problem = ContinuousProblem(
            model,
            qadr,
            refs,
            p,
            R,
            masks,
            fps,
            replace(config or ContinuityConfig(), position_weight=300.0, orientation_weight=orientation_weight),
        )
        self.n, self.j = len(initial), len(qadr)
        if self.n < 3:
            raise ValueError("Global posture optimization requires at least three frames")
        self.keep = np.r_[np.arange(12), np.arange(12 + 2 * self.j, self.problem.rows)]
        d1 = sparse.diags([-np.ones(self.n - 1), np.ones(self.n - 1)], [0, 1], shape=(self.n - 1, self.n), format="csr")
        d2 = sparse.diags(
            [np.ones(self.n - 2), -2 * np.ones(self.n - 2), np.ones(self.n - 2)],
            [0, 1, 2],
            shape=(self.n - 2, self.n),
            format="csr",
        )
        self.D = sparse.kron(d1, sparse.eye(self.j), format="csr")
        self.A = sparse.kron(d2, sparse.eye(self.j), format="csr")
        self.velocity_weight = 0.12 * fps / 30
        self.acceleration_weight = acceleration_weight * (fps / 30) ** 2
        self.cache = None
        self.calls = 0
        self.midpoint_collisions = midpoint_collisions
        mid = sparse.diags(
            [np.full(self.n - 1, 0.5), np.full(self.n - 1, 0.5)], [0, 1], shape=(self.n - 1, self.n), format="csr"
        )
        self.M = sparse.kron(mid, sparse.eye(self.j), format="csr")
        self.angular_guard = None
        if angular_guard is not None:
            from .angular_guard import AngularGuard

            self.angular_guard = AngularGuard(self.problem, **angular_guard)

    def evaluate(self, x):
        if self.cache is not None and np.array_equal(x, self.cache):
            return self.values
        q = x.reshape(self.n, self.j)
        rs = []
        js = []
        angular_values = []
        angular_jacs = []
        for t in range(self.n):
            self.problem.frame(t, self.initial[t])
            r, j = self.problem.evaluate(q[t])
            rs.append(r[self.keep])
            js.append(sparse.csr_matrix(j[self.keep]))
            if self.angular_guard is not None:
                av, aj = self.angular_guard.frame(t)
                angular_values.append(av)
                angular_jacs.append(aj)
        delta = self.D @ x
        # A small interior buffer leaves room for finite solver tolerance.
        buffer = min(1e-3, self.problem.step * 0.01)
        excess = np.maximum(abs(delta) - (self.problem.step - buffer), 0)
        speed_jac = sparse.diags((excess > 0).astype(float) * 1000) @ self.D
        r = np.r_[
            np.concatenate(rs),
            self.velocity_weight * delta,
            self.acceleration_weight * (self.A @ x),
            1000 * np.sign(delta) * excess,
        ]
        jac = sparse.vstack(
            [
                sparse.block_diag(js, format="csr"),
                self.velocity_weight * self.D,
                self.acceleration_weight * self.A,
                speed_jac,
            ],
            format="csr",
        )
        if self.angular_guard is not None:
            ar, aj = self.angular_guard.aggregate(np.asarray(angular_values), np.asarray(angular_jacs))
            r = np.r_[r, ar]
            jac = sparse.vstack([jac, aj], format="csr")
        if self.midpoint_collisions:
            mr = []
            mj = []
            start = 12 + 2 * self.j
            for t in range(self.n - 1):
                self.problem.frame(t, 0.5 * (self.initial[t] + self.initial[t + 1]))
                rv, jv = self.problem.evaluate(0.5 * (q[t] + q[t + 1]))
                mr.append(rv[start:])
                mj.append(sparse.csr_matrix(jv[start:]))
            r = np.r_[r, np.concatenate(mr)]
            jac = sparse.vstack([jac, sparse.block_diag(mj, format="csr") @ self.M], format="csr")
        self.cache = x.copy()
        self.values = (r, jac)
        self.calls += 1
        if self.calls % 10 == 0:
            print(f"trajectory evaluations={self.calls} cost={r @ r / 2:.6f}", flush=True)
        return self.values


def refine_posture(
    model,
    initial,
    qadr,
    refs,
    p,
    R,
    masks,
    fps,
    max_nfev=50,
    config=None,
    candidate_callback=None,
    allow_infeasible=False,
    angular_guard=None,
    midpoint_collisions=None,
    margin_safe=True,
):
    original = model.geom_margin.copy()
    try:
        model.geom_margin[:] = np.maximum(original, (config or ContinuityConfig()).clearance)
        problem = TrajectoryProblem(
            model,
            initial,
            qadr,
            refs,
            p,
            R,
            masks,
            fps,
            config=config,
            orientation_weight=(config or ContinuityConfig()).orientation_weight if allow_infeasible else 2.0,
            angular_guard=angular_guard,
            midpoint_collisions=allow_infeasible if midpoint_collisions is None else midpoint_collisions,
        )
        before, _, old_angles = problem.problem.measure(initial)
        before["orientation_p95_deg"] = float(np.nanpercentile(old_angles, 95) * 180 / np.pi)
        if not before["accepted"] and not allow_infeasible:
            raise ValueError("Posture refinement requires a valid continuous input")
        if allow_infeasible:
            # Positive contact margins can generate discontinuous box contact
            # manifolds. Optimize actual penetration with zero margins; restore
            # the original validation margins before measuring any candidate.
            validation_margin = model.geom_margin.copy()
            model.geom_margin[:] = 0.0
            problem.problem.collision_residual_clearance = 0.0
            problem.problem.stable_box_clearance = True
            problem.problem.box_pair_clearance = (
                {
                    pair: max(
                        problem.problem.cfg.clearance, float(validation_margin[pair[0]] + validation_margin[pair[1]])
                    )
                    for pair in problem.problem.box_pairs
                }
                if margin_safe
                else {}
            )
        x = np.asarray(initial)[:, qadr].ravel()
        initial_residual = problem.evaluate(x)[0]
        initial_cost = float(initial_residual @ initial_residual / 2)
        result = least_squares(
            lambda x: problem.evaluate(x)[0],
            x,
            jac=lambda x: problem.evaluate(x)[1],
            bounds=(np.tile(problem.problem.lower, problem.n), np.tile(problem.problem.upper, problem.n)),
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-7,
            x_scale=1.0 if allow_infeasible else "jac",
            tr_solver="lsmr",
            tr_options={"maxiter": 300, "atol": 1e-8, "btol": 1e-8},
        )
        candidate = np.array(initial, copy=True)
        candidate[:, qadr] = result.x.reshape(problem.n, problem.j)
        # Remove residual numerical speed violations before strict FK validation.
        for t in range(1, len(candidate)):
            candidate[t, qadr] = np.clip(
                candidate[t, qadr],
                candidate[t - 1, qadr] - problem.problem.step,
                candidate[t - 1, qadr] + problem.problem.step,
            )
        attempts = []
        model.geom_margin[:] = np.maximum(original, (config or ContinuityConfig()).clearance)
        selected = (np.array(initial, copy=True), before)
        before_angles = np.asarray(before["orientation_mean_deg"], dtype=float)
        observed = np.isfinite(before_angles)
        for fraction in (1.0, 0.5, 0.25, 0.125):
            track = np.asarray(initial) + fraction * (candidate - initial)
            report, _, angles = problem.problem.measure(track)
            report["orientation_p95_deg"] = float(np.nanpercentile(angles, 95) * 180 / np.pi)
            if candidate_callback is not None:
                candidate_callback(track.copy(), dict(report))
            allowed = report["accepted"] and (
                not before["accepted"]
                or (
                    report["velocity_rms_rad_s"] <= before["velocity_rms_rad_s"] * 1.05 + 1e-6
                    and report["acceleration_rms_rad_s2"] <= before["acceleration_rms_rad_s2"] * 1.05 + 1e-6
                    and report["orientation_p95_deg"] <= float(np.nanpercentile(old_angles, 95) * 180 / np.pi) + 1e-6
                    and np.all(
                        np.asarray(report["orientation_mean_deg"], dtype=float)[observed]
                        <= before_angles[observed] + 0.1
                    )
                    and np.nanmean(np.asarray(report["orientation_mean_deg"], dtype=float))
                    < np.nanmean(np.asarray(selected[1]["orientation_mean_deg"], dtype=float))
                )
            )
            if allowed and selected[1]["accepted"] and not before["accepted"]:
                # Once recovery finds a feasible track, retain the smoother
                # accepted fraction instead of comparing against a bad seed.
                allowed = report["acceleration_rms_rad_s2"] < selected[1]["acceleration_rms_rad_s2"]
            attempts.append(dict(fraction=fraction, selected_eligible=bool(allowed), **report))
            if allowed:
                selected = (track, report)
        q, report = selected
        _, position, rotation = problem.problem.measure(q)
        return (
            q,
            dict(
                before=before,
                after=report,
                attempts=attempts,
                changed=bool(not np.array_equal(q, initial)),
                nfev=result.nfev,
                initial_cost=initial_cost,
                final_cost=float(result.cost),
                optimality=float(result.optimality),
                step_norm=float(np.linalg.norm(result.x - x)),
                optimizer_success=bool(result.success),
                optimizer_message=result.message,
            ),
            position,
            rotation,
        )
    finally:
        model.geom_margin[:] = original


def refine_recovery_stages(
    model, initial, qadr, refs, p, R, masks, fps, max_nfev=160, config=None, candidate_callback=None, angular_guard=None
):
    """Continuation from frame constraints to margin-safe midpoint constraints.

    Warm starts never redefine the angular baseline supplied by the caller.
    Both phases retain the independent final physical validation.
    """
    first = []
    _, phase1, _, _ = refine_posture(
        model,
        initial,
        qadr,
        refs,
        p,
        R,
        masks,
        fps,
        max_nfev=max_nfev,
        config=config,
        allow_infeasible=True,
        angular_guard=angular_guard,
        midpoint_collisions=False,
        margin_safe=False,
        candidate_callback=lambda q, r: first.append((q, r)),
    )
    # Full optimized track is a numerical seed, even when its midpoint check fails.
    warm = first[0][0]
    q, phase2, position, rotation = refine_posture(
        model,
        warm,
        qadr,
        refs,
        p,
        R,
        masks,
        fps,
        max_nfev=min(80, max_nfev),
        config=config,
        allow_infeasible=True,
        angular_guard=angular_guard,
        candidate_callback=candidate_callback,
        midpoint_collisions=True,
        margin_safe=True,
    )
    phase2["frame_phase"] = phase1
    phase2["total_nfev"] = phase1["nfev"] + phase2["nfev"]
    phase2["method"] = "frame_then_margin_safe_midpoint"
    return q, phase2, position, rotation
