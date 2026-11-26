# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
from .acceptance import gate as _acceptance_gate
import itertools
import os
import numpy as np
import mujoco
from .registry import selected
from steps.sharpa.view_constraints_v8 import ViewConstraints as Parent


def housing_body_names(m):
    # A palm geometry may be carried by a fixed ancestor of the semantic
    # finger frame (Ability's `base` does this). Do not classify a real palm
    # as an arm merely because it precedes that reference frame.
    names = set()
    for side in ["left", "right"]:
        bodies = ["forearm", "wrist"] if selected() == "shadow" else ["base_link"] if selected() == "ability" else []
        for body in bodies:
            name = side + "_hand_" + body
            assert m.body(name).id > 0
            names.add(name)
    return names


def proximal_housing_names(m):
    # Genuine forearm housing joins the proximal arm group. Short distal
    # wrists remain counted in total arm pixels, like Panda link6 and link7.
    return {s + "_hand_forearm" for s in ["left", "right"]} if selected() == "shadow" else set()


class ViewConstraints(Parent):
    def __init__(self, *args, workspace_points=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace_points = workspace_points
        if workspace_points is not None:
            self.workspace_cp = np.einsum(
                "nji,nsaj->nsai", self.R, np.asarray(workspace_points) - self.head[:, None, None, :3]
            )
            self.workspace_xy = self.workspace_cp[:, :, :, :2] / np.maximum(self.workspace_cp[:, :, :, 2:], 1e-4)

    def _workspace_rows(self, m, d, t, va, need):
        # Exact copy of the 'workspace' rows of PointConstraints.terms without
        # its discarded 'view' rows; Jacobians only when the caller needs them.
        rows = []
        if not self.masks[t].any():
            return rows
        if not hasattr(self, "_link_ids") or self._link_ids[0] is not m:
            self._link_ids = (m, {(s, k): m.body(s + "_link" + str(k)).id for s in ["left", "right"] for k in range(5)})
        jp = np.zeros((3, m.nv))
        for s in ["left", "right"]:
            for k in range(5):
                bid = self._link_ids[1][(s, k)]
                world = d.xpos[bid]
                radius = [0.09, 0.09, 0.08, 0.08, 0.075, 0.055][k]
                validP = self.goals[t, self.masks[t]]
                a, b = validP[0], validP[-1]
                ab = b - a
                fraction = np.clip((world - a) @ ab / max(float(ab @ ab), 1e-10), 0, 1)
                offset = world - (a + fraction * ab)
                distance = float(np.linalg.norm(offset))
                safe = radius + 0.10
                if distance < safe and distance > 1e-9:
                    if need:
                        mujoco.mj_jacBody(m, d, jp, None, bid)
                    rows.append((safe - distance, -(offset / distance) @ jp[:, va] if need else None, k, "workspace"))
        return rows

    def terms(self, m, d, t, va):
        need = getattr(self, "need_jac", True)
        rows = self._workspace_rows(m, d, t, va, need)
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
        if self.workspace_points is not None:
            cp = self.workspace_cp[t].reshape(-1, 3)
            xy = self.workspace_xy[t].reshape(-1, 2)
            valid = (
                np.repeat(self.masks[t], self.workspace_cp.shape[2])
                & (cp[:, 2] > 0.05)
                & (xy[:, 0] >= L)
                & (xy[:, 0] <= Right)
                & (xy[:, 1] >= U)
                & (xy[:, 1] <= D)
            )
            active = xy[valid]
        frustum = [
            np.array([1.0, 0.0, -L]),
            np.array([-1.0, 0.0, Right]),
            np.array([0.0, 1.0, -U]),
            np.array([0.0, -1.0, D]),
        ]
        for g, bid, k, vertices in self.meshes(m):
            world = vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pc = (world - cam) @ R
            # geom_rbound is centered at geom_xpos, not at an arbitrary mean
            # of convex-hull vertices. Preserve the true sphere keepout.
            centre = d.geom_xpos[g]
            radius = float(m.geom_rbound[g])
            offset = centre - cam
            distance = max(float(np.linalg.norm(offset)), 1e-9)
            deficit = radius + (0.18 if k < 5 else 0.10) - distance
            if deficit > 0:
                if need:
                    jp_c = np.zeros((3, m.nv))
                    mujoco.mj_jac(m, d, jp_c, None, centre, bid)
                rows.append((deficit, -(offset / distance) @ jp_c[:, va] if need else None, k, "camera"))
            if np.max(pc[:, 2]) <= 0.01:
                continue
            if any(np.max(pc @ axis) < 0 for axis in frustum) or not len(active):
                continue
            left, right, down = active[:, 0].min() - 0.10, active[:, 0].max() + 0.10, active[:, 1].max() + 0.075
            axes = [np.array([1.0, 0.0, -left]), np.array([-1.0, 0.0, right]), np.array([0.0, -1.0, down])]
            scores = [float(np.max(pc @ axis)) for axis in axes]
            which = int(np.argmin(scores))
            axis = axes[which]
            gvalue = scores[which] + 0.015
            if gvalue <= 0:
                continue
            if not need:
                rows.append((gvalue, None, k, "view"))
                continue
            index = int(np.argmax(pc @ axis))
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, world[index], bid)
            rows.append((gvalue, axis @ R.T @ jp[:, va], k, "view"))
        if os.environ.get("EGO2ROBOT_ACTUAL_HAND_VIEW_SEARCH") == "1":
            point_scratch = np.zeros((3, m.nv))
            for hand_side, s in enumerate(["left", "right"]):
                if not self.masks[t, hand_side]:
                    continue
                hand_points = []
                point_bodies = []
                if not hasattr(self, "_hand_visual_hulls") or self._hand_visual_hulls[0] is not m:
                    from scipy.spatial import ConvexHull

                    hulls = []
                    housing = housing_body_names(m)
                    for hg in range(m.ngeom):
                        hb = int(m.geom_bodyid[hg])
                        body = m.body(hb).name or ""
                        if "_hand_" not in body or body in housing or m.geom_group[hg] == 3 or m.geom_rgba[hg, 3] <= 0:
                            continue
                        if m.geom_type[hg] == mujoco.mjtGeom.mjGEOM_MESH:
                            mesh = int(m.geom_dataid[hg])
                            start = int(m.mesh_vertadr[mesh])
                            count = int(m.mesh_vertnum[mesh])
                            v = m.mesh_vert[start : start + count]
                            vertices = v[ConvexHull(v).vertices].copy()
                        else:
                            radius = float(m.geom_rbound[hg])
                            vertices = np.array(list(itertools.product([-radius, radius], repeat=3)))
                        hulls.append((hg, hb, body.split("_")[0], vertices))
                    self._hand_visual_hulls = (m, hulls)
                for hg, hb, side, hull in self._hand_visual_hulls[1]:
                    if side != s:
                        continue
                    world = hull @ d.geom_xmat[hg].reshape(3, 3).T + d.geom_xpos[hg]
                    hand_points.extend(world)
                    point_bodies.extend([hb] * len(world))
                hand_points = np.asarray(hand_points)
                hand_pc = (hand_points - cam) @ R
                if not len(hand_points) or np.min(hand_pc[:, 2]) <= 0.05:
                    continue
                hand_xy = hand_pc[:, :2] / hand_pc[:, 2:]

                def projected_jac(pc, J):
                    x, y, z = pc
                    return np.array([[1 / z, 0, -x / z**2], [0, 1 / z, -y / z**2]]) @ R.T @ J

                hj = {}

                def hand_projected_jac(index):
                    if index not in hj:
                        mujoco.mj_jac(m, d, point_scratch, None, hand_points[index], point_bodies[index])
                        hj[index] = projected_jac(hand_pc[index], point_scratch[:, va].copy())
                    return hj[index]

                for geom, bid, k, vertices in self.meshes(m):
                    world = vertices @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
                    pc = (world - cam) @ R
                    if np.min(pc[:, 2]) <= 0.02 or np.min(pc[:, 2]) > np.max(hand_pc[:, 2]) + 0.015:
                        continue
                    xy = pc[:, :2] / pc[:, 2:]
                    choices = []
                    for axis in [0, 1]:
                        arm_hi = int(np.argmax(xy[:, axis]))
                        arm_lo = int(np.argmin(xy[:, axis]))
                        hand_hi = int(np.argmax(hand_xy[:, axis]))
                        hand_lo = int(np.argmin(hand_xy[:, axis]))
                        choices.append(
                            (float(xy[arm_hi, axis] - hand_xy[hand_lo, axis] + 0.03), axis, 1, arm_hi, hand_lo)
                        )
                        choices.append(
                            (float(hand_xy[hand_hi, axis] - xy[arm_lo, axis] + 0.03), axis, -1, arm_lo, hand_hi)
                        )
                    value, axis, sign, ai, hi = min(choices, key=lambda row: row[0])
                    if value <= 0:
                        continue
                    mujoco.mj_jac(m, d, point_scratch, None, world[ai], bid)
                    aj = projected_jac(pc[ai], point_scratch[:, va])
                    jac = sign * (aj[axis] - hand_projected_jac(hi)[axis])
                    rows.append(
                        (_acceptance_gate(0.0137, "layout_m", -0.0003) + 0.4 * value, 0.4 * jac, k, "hand_view")
                    )
        return rows

    def meshes(self, m):
        if hasattr(self, "_exact_mesh_cache") and self._exact_mesh_cache[0] is m:
            base = self._exact_mesh_cache[1]
        else:
            from scipy.spatial import ConvexHull

            base = []
            for g, bid, k, corners in super().meshes(m):
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = int(m.geom_dataid[g])
                    a = int(m.mesh_vertadr[mesh])
                    n = int(m.mesh_vertnum[mesh])
                    v = m.mesh_vert[a : a + n]
                    corners = v[ConvexHull(v).vertices].copy()
                base.append((g, bid, k, corners))
            self._exact_mesh_cache = (m, base)
        if hasattr(self, "_housing_cache") and self._housing_cache[0] is m:
            return base + self._housing_cache[1]
        names = proximal_housing_names(m)
        extra = []
        for g in range(m.ngeom):
            bid = int(m.geom_bodyid[g])
            if m.body(bid).name not in names or m.geom_group[g] == 3 or m.geom_rgba[g, 3] <= 0:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                from scipy.spatial import ConvexHull

                mesh = int(m.geom_dataid[g])
                a = int(m.mesh_vertadr[mesh])
                n = int(m.mesh_vertnum[mesh])
                v = m.mesh_vert[a : a + n]
                # Linear support maxima are exact on the actual mesh convex
                # hull. Empty corners of its rotated AABB are not robot parts.
                hull = ConvexHull(v)
                vertices = v[hull.vertices].copy()
            else:
                lo = -np.full(3, m.geom_rbound[g])
                hi = -lo
                vertices = np.array(list(itertools.product(*zip(lo, hi))))
            extra.append((g, bid, 5, vertices))
        self._housing_cache = (m, extra)
        return base + extra

    def audit(self, m, q):
        d = mujoco.MjData(m)
        maximum = 0.0
        camera = 0.0
        workspace = 0.0
        projection = 0.0
        worst = []
        for t, state in enumerate(q):
            d.qpos[:] = state
            mujoco.mj_forward(m, d)
            for value, _, k, kind in self.terms(m, d, t, []):
                maximum = max(maximum, value)
                if kind == "camera":
                    camera = max(camera, value)
                elif kind == "workspace":
                    workspace = max(workspace, value)
                elif kind in ["view", "hand_view"]:
                    projection = max(projection, value)
                if value > 0.045:
                    worst.append(dict(frame=t, link=k, kind=kind, violation_m=value))
        return dict(
            proximal_layout_max_violation_m=maximum,
            camera_keepout_max_violation_m=camera,
            workspace_envelope_max_violation_m=workspace,
            projected_mesh_proxy_max_violation_m=projection,
            legacy_search_proxy_passed=bool(maximum <= _acceptance_gate(0.014, "layout_m")),
            actual_pixel_verification_required=True,
            failed_samples=len(worst),
            worst_samples=sorted(worst, key=lambda x: x["violation_m"], reverse=True)[:12],
            actual_visual_mesh_envelopes=True,
            camera_keepout_enabled=True,
            camera_keepout_rule="Euclidean clearance from the actual source camera path",
            reconstructed_scene_collision_validated=False,
            geometry_hidden_to_improve_view=False,
        )
