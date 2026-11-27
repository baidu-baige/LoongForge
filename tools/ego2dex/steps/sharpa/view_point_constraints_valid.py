# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Camera and task-workspace avoidance for proximal arm placement.

This is an explicit conservative workspace envelope, not reconstructed cloth
or scene geometry. It changes physical arm/base poses, never hides geometry.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class ViewConstraints:
    def __init__(self, head, K, shape, goals, masks):
        assert head is not None and K is not None and shape is not None
        self.head = np.asarray(head)
        self.R = Rotation.from_quat(self.head[:, [4, 5, 6, 3]]).as_matrix()
        self.K = np.asarray(K)
        self.shape = shape
        self.goals = goals
        self.masks = masks
        self.cp = np.einsum("nji,nsj->nsi", self.R, goals - self.head[:, None, :3])
        self.xy = self.cp[:, :, :2] / np.maximum(self.cp[:, :, 2:], 0.1)

    def terms(self, m, d, t, va):
        R = self.R[t]
        cam = self.head[t, :3]
        jp = np.zeros((3, m.nv))
        rows = []
        if not self.masks[t].any():
            return rows
        for side, s in enumerate(["left", "right"]):
            for k in range(0, 6):
                bid = m.body(s + "_link" + str(k)).id
                world = d.xpos[bid]
                p = R.T @ (world - cam)
                mujoco.mj_jacBody(m, d, jp, None, bid)
                J = R.T @ jp[:, va]
                # A center/radius conservative envelope keeps the base and
                # upper arm to either image side or below the working hands.
                radius = [0.09, 0.09, 0.08, 0.08, 0.075, 0.055][k]
                # Camera visibility concerns rendered geometry. The hidden
                # static model root still receives the3Dworkspace constraint.
                if p[2] > 0.01 and k > 0:
                    active = self.xy[t, self.masks[t]]
                    edgeL = active[:, 0].min() - 0.12
                    edgeR = active[:, 0].max() + 0.12
                    low = active[:, 1].max() + (0.22 if k < 3 else 0.10 if k < 5 else 0.025)
                    H, W = self.shape
                    fx, fy = self.K[0, 0], self.K[1, 1]
                    cx, cy = self.K[0, 2], self.K[1, 2]
                    outside = (
                        p[0] + radius < (0 - cx) / fx * p[2]
                        or p[0] - radius > (W - cx) / fx * p[2]
                        or p[1] + radius < (0 - cy) / fy * p[2]
                        or p[1] - radius > (H - cy) / fy * p[2]
                    )
                    if not outside:
                        axes = [np.array([1.0, 0.0, -edgeL]), np.array([-1.0, 0.0, edgeR]), np.array([0.0, -1.0, low])]
                        axis = min(axes, key=lambda a: float(a @ p))
                        g = float(axis @ p + radius)
                        rows.append((max(g, 0.0), axis @ J, k, "view"))
                if k < 5:
                    validP = self.goals[t, self.masks[t]]
                    a, b = validP[0], validP[-1]
                    ab = b - a
                    fraction = np.clip((world - a) @ ab / max(float(ab @ ab), 1e-10), 0, 1)
                    offset = world - (a + fraction * ab)
                    distance = float(np.linalg.norm(offset))
                    safe = radius + 0.10
                    if distance < safe and distance > 1e-9:
                        rows.append((safe - distance, -(offset / distance) @ jp[:, va], k, "workspace"))
        return rows

    def residuals(self, m, d, t, va):
        for value, J, k, kind in self.terms(m, d, t, va):
            if value <= 0:
                continue
            weight = (350.0 if k < 3 else 250.0 if k < 5 else 60.0) if kind == "view" else 500.0
            yield np.array([weight * value]), weight * J

    def audit(self, m, q):
        d = mujoco.MjData(m)
        maximum = 0.0
        violations = []
        workmax = 0.0
        for t, row in enumerate(q):
            d.qpos[:] = row
            mujoco.mj_forward(m, d)
            for g, _, k, kind in self.terms(m, d, t, []):
                if k >= 5:
                    continue
                maximum = max(maximum, g)
                if kind == "workspace":
                    workmax = max(workmax, g)
                if g > 0.045:
                    violations.append(dict(frame=t, link=k, kind=kind, violation_m=g))
        return dict(
            proximal_layout_max_violation_m=maximum,
            workspace_envelope_max_violation_m=workmax,
            failed_samples=len(violations),
            worst_samples=sorted(violations, key=lambda r: r["violation_m"], reverse=True)[:12],
            reconstructed_scene_collision_validated=False,
            geometry_hidden_to_improve_view=False,
        )
