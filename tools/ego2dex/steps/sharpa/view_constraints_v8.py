# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Visibility envelopes of actual rendered proximal meshes, not joint centers."""

import itertools
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .view_point_constraints_valid import ViewConstraints as PointConstraints


class ViewConstraints(PointConstraints):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xy = self.cp[:, :, :2] / np.maximum(self.cp[:, :, 2:], 1e-4)

    def meshes(self, m):
        if hasattr(self, "_mesh_cache") and self._mesh_cache[0] is m:
            return self._mesh_cache[1]
        rows = []
        for g in range(m.ngeom):
            body = m.body(int(m.geom_bodyid[g])).name or ""
            parts = body.split("_")
            if (
                len(parts) != 2
                or parts[0] not in ["left", "right"]
                or parts[1] not in ["link" + str(k) for k in range(1, 6)]
            ):
                continue
            if m.geom_group[g] == 3 or m.geom_rgba[g, 3] <= 0:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(m.geom_dataid[g])
                start = int(m.mesh_vertadr[mesh])
                count = int(m.mesh_vertnum[mesh])
                v = m.mesh_vert[start : start + count]
                lo, hi = v.min(axis=0), v.max(axis=0)
            else:
                radius = float(m.geom_rbound[g])
                lo = np.full(3, -radius)
                hi = np.full(3, radius)
            corners = np.array(list(itertools.product(*zip(lo, hi))))
            rows.append((g, int(m.geom_bodyid[g]), int(parts[1][4:]), corners))
        self._mesh_cache = (m, rows)
        return rows

    def terms(self, m, d, t, va):
        rows = [row for row in super().terms(m, d, t, va) if row[3] == "workspace"]
        R = self.R[t]
        cam = self.head[t, :3]
        K = self.K
        H, W = self.shape
        L = (0 - K[0, 2]) / K[0, 0]
        Right = (W - K[0, 2]) / K[0, 0]
        U = (0 - K[1, 2]) / K[1, 1]
        D = (H - K[1, 2]) / K[1, 1]
        visible = (
            self.masks[t]
            & (self.cp[t, :, 2] > 0.05)
            & (self.xy[t, :, 0] >= L)
            & (self.xy[t, :, 0] <= Right)
            & (self.xy[t, :, 1] >= U)
            & (self.xy[t, :, 1] <= D)
        )
        active = self.xy[t, visible]
        if not hasattr(self, "_camera_right"):
            self._camera_right = Rotation.from_matrix(self.R).mean().as_matrix()[:, 0]
            self._corridor_sign = {}
        for g, bid, k, corners in self.meshes(m):
            world = corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pc = (world - cam) @ R
            centre = world.mean(axis=0)
            jp_c = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp_c, None, centre, bid)
            radius = float(m.geom_rbound[g])
            body = m.body(bid).name
            side = body.split("_")[0]
            if side not in self._corridor_sign:
                lateral = float(self._camera_right @ (centre - np.median(self.head[:, :3], axis=0)))
                self._corridor_sign[side] = np.sign(lateral) if abs(lateral) > 0.08 else (-1 if side == "left" else 1)
            offset = centre - cam
            distance = max(float(np.linalg.norm(offset)), 1e-9)
            deficit = radius + (0.18 if k < 5 else 0.10) - distance
            Jcamera = -(offset / distance) @ jp_c[:, va]
            if deficit > 0:
                rows.append((deficit, Jcamera, k, "camera"))
            if np.max(pc[:, 2]) <= 0.01:
                continue
            frustum = [
                np.array([1.0, 0.0, -L]),
                np.array([-1.0, 0.0, Right]),
                np.array([0.0, 1.0, -U]),
                np.array([0.0, -1.0, D]),
            ]
            if any(np.max(pc @ axis) < 0 for axis in frustum):
                continue
            # Shoulders should enter from beyond the viewport. The elbow and
            # proximal forearm must clear the observed hand-workspace hull.
            if not len(active):
                continue
            left, right, down = active[:, 0].min() - 0.10, active[:, 0].max() + 0.10, active[:, 1].max() + 0.075
            axes = [np.array([1.0, 0.0, -left]), np.array([-1.0, 0.0, right]), np.array([0.0, -1.0, down])]
            scores = [float(np.max(pc @ axis)) for axis in axes]
            which = int(np.argmin(scores))
            axis = axes[which]
            gvalue = scores[which] + 0.015
            if gvalue <= 0:
                continue
            index = int(np.argmax(pc @ axis))
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, world[index], bid)
            rows.append((gvalue, axis @ R.T @ jp[:, va], k, "view"))
        return rows

    def residuals(self, m, d, t, va):
        for value, J, k, kind in self.terms(m, d, t, va):
            if value <= 0:
                continue
            weight = 500.0 if kind == "workspace" else 350.0
            yield np.array([weight * value]), weight * J

    def audit(self, m, q):
        d = mujoco.MjData(m)
        worst = []
        maximum = 0.0
        work = 0.0
        for t, row in enumerate(q):
            d.qpos[:] = row
            mujoco.mj_forward(m, d)
            for value, _, k, kind in self.terms(m, d, t, []):
                maximum = max(maximum, value)
                if kind == "workspace":
                    work = max(work, value)
                if value > 0.045:
                    worst.append(dict(frame=t, link=k, kind=kind, violation_m=value))
        return dict(
            proximal_layout_max_violation_m=maximum,
            workspace_envelope_max_violation_m=work,
            failed_samples=len(worst),
            worst_samples=sorted(worst, key=lambda r: r["violation_m"], reverse=True)[:12],
            actual_visual_mesh_envelopes=True,
            camera_keepout_enabled=True,
            camera_keepout_rule="Euclidean clearance from the actual source camera path",
            reconstructed_scene_collision_validated=False,
            geometry_hidden_to_improve_view=False,
        )
