# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Angular acceptance metrics and piecewise analytic trajectory derivatives.

Uses a stricter, unconditional baseline +0.20 degree optimization target;
the independent +0.25 degree final gate remains authoritative.
"""

import numpy as np
import mujoco
from scipy import sparse
from scipy.spatial.transform import Rotation


class AngularGuard:
    def __init__(self, problem, targets, tcp, baseline, weight=1000.0):
        self.problem = problem
        self.targets = np.asarray(targets)
        self.tcp = np.asarray(tcp)
        self.weight = weight
        keys = ("symmetry_orientation_mean_deg", "approach_mean_deg", "opening_line_mean_deg")
        self.limits = np.r_[
            np.asarray([baseline[k] for k in keys], dtype=float).ravel(), baseline["symmetry_orientation_p95_deg"]
        ]
        self.limits = np.deg2rad(self.limits + 0.20)

    def frame(self, t):
        p = self.problem
        values = np.full((3, 2), np.nan)
        gradients = np.zeros((3, 2, len(p.qadr)))
        for side, (kind, idx) in enumerate(p.refs):
            if not p.masks[t, side]:
                continue
            if kind == "site":
                actual = p.data.site_xmat[idx].reshape(3, 3) @ self.tcp[side]
                mujoco.mj_jacSite(p.model, p.data, p.jp, p.jr, idx)
            else:
                actual = p.data.xmat[idx].reshape(3, 3) @ self.tcp[side]
                mujoco.mj_jacBody(p.model, p.data, p.jp, p.jr, idx)
            target = self.targets[t, side] @ self.tcp[side]
            alternatives = (target, target @ np.diag([-1.0, -1.0, 1.0]))
            vectors = [Rotation.from_matrix(a.T @ actual).as_rotvec() for a in alternatives]
            branch = int(np.argmin([np.linalg.norm(v) for v in vectors]))
            v = vectors[branch]
            angle = np.linalg.norm(v)
            values[0, side] = angle
            if angle > 1e-10:
                gradients[0, side] = (alternatives[branch] @ (v / angle)) @ p.jr[:, p.vadr]
            for metric, axis, symmetric in ((1, 2, False), (2, 1, True)):
                a, b = target[:, axis], actual[:, axis]
                dot = float(np.clip(a @ b, -1.0, 1.0))
                sign = (1.0 if dot >= 0 else -1.0) if symmetric else 1.0
                values[metric, side] = np.arccos(sign * dot)
                sine = np.sqrt(max(0.0, 1.0 - dot * dot))
                if sine > 1e-10:
                    gradients[metric, side] = sign * np.cross(a, b) @ p.jr[:, p.vadr] / sine
        return values, gradients

    def aggregate(self, values, gradients):
        n, _, _, j = gradients.shape
        metrics = np.zeros(7)
        jac = np.zeros((7, n, j))
        for metric in range(3):
            for side in range(2):
                row = metric * 2 + side
                valid = np.isfinite(values[:, metric, side])
                if not valid.any():
                    continue
                metrics[row] = values[valid, metric, side].mean()
                jac[row, valid] = gradients[valid, metric, side] / valid.sum()
        # Match NumPy's default linearly interpolated 95th percentile exactly.
        pairs = np.argwhere(np.isfinite(values[:, 0, :]))
        order = np.argsort(values[pairs[:, 0], 0, pairs[:, 1]], kind="stable")
        rank = 0.95 * (len(order) - 1)
        lo, hi = int(np.floor(rank)), int(np.ceil(rank))
        for index, weight in ((lo, 1.0 - (rank - lo)), (hi, rank - lo)):
            t, side = pairs[order[index]]
            metrics[6] += weight * values[t, 0, side]
            jac[6, t] += weight * gradients[t, 0, side]
        active = np.isfinite(self.limits) & (metrics > self.limits)
        residual = np.where(active, metrics - self.limits, 0.0) * self.weight
        jac *= (active * self.weight)[:, None, None]
        return residual, sparse.csr_matrix(jac.reshape(7, n * j))
