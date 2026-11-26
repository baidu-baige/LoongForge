# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Hard conic projection of arm joints and static bases near acceptance.

Small joint corrections use real independent coordinates and preserve every
coupling and the rigid assembly. Original task
balls, original angular gate, native speeds, contact and pixel checks remain.
"""

from .acceptance import gate as _acceptance_gate
import json
import os
from pathlib import Path
import clarabel
import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from steps.sharpa.task_tcp import pose_and_jac, track, angular_metrics
from steps.sharpa.rigid_retarget import arm_contacts
from steps.retarget.continuity import _log_jacobian
from steps.retarget.joint_pose import angular_eligible
from .coupled_place import virtual_model, allowed_contact
from .panda_scene import build, indices
from .registry import speed_limits, coupling_error, whole_affine
from .mount_contract import audit_connection
from .view_constraints import ViewConstraints
from .acceptance import current


def refine(result, times, P, R, masks, head, K, shape, baseline, output):
    acceptance = current()
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    history = []
    n = len(times)
    q = result["qpos"].copy()
    model = result["model"]
    record = result["record"]
    if record["passed"]:
        return result
    measured_view = os.environ.get("EGO2ROBOT_VIEW_PROXY_MODE") == "measured_pixels"

    def rank(record):
        layout = record["view_layout"]
        hard_view = (
            max(
                layout.get("camera_keepout_max_violation_m", 0.0), layout.get("workspace_envelope_max_violation_m", 0.0)
            )
            if measured_view
            else layout["proximal_layout_max_violation_m"]
        )
        return (
            not record["passed"],
            max(
                record["eef_position_max_m"] / acceptance.position_m,
                record["eef_rotation_max_rad"] / acceptance.rotation_rad,
                hard_view / _acceptance_gate(0.014, "layout_m"),
                record.get("speed_to_native_cap_ratio", 0.0),
            ),
        )

    best = result
    for iteration in range(6):
        data = mujoco.MjData(model)
        data.qpos[:] = q[0]
        mujoco.mj_forward(model, data)
        bases = {
            s: dict(
                pos=data.xpos[model.body(s + "_link0").id].copy(), quat=data.xquat[model.body(s + "_link0").id].copy()
            )
            for s in ["left", "right"]
        }
        vm = virtual_model(bases, wide=True)
        vm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        vm.geom_margin[:] = np.maximum(vm.geom_margin, 0.003)
        names = [model.joint(i).name for i in range(model.njnt)]
        vq = np.tile(vm.qpos0, (n, 1))
        for i, name in enumerate(names):
            vq[:, int(vm.joint(name).qposadr[0])] = q[:, int(model.jnt_qposadr[i])]
        arm = np.r_[indices(model, "left")["arm_qadr"], indices(model, "right")["arm_qadr"]]
        control = np.arange(model.nq)
        offset, M, motor_names = whole_affine(model, control)
        dcount = M.shape[1]
        pcount = model.nq
        basis = sparse.block_diag((sparse.csc_matrix(M), sparse.eye(6)), format="csr").toarray()
        va_control = np.array([int(vm.joint(model.joint(int(i)).name).dofadr[0]) for i in control])
        va_base = np.array([int(vm.joint(f"{s}_static_t{a}").dofadr[0]) for s in ["left", "right"] for a in "xyz"])
        va = np.r_[va_control, va_base]
        N = n * dcount + 6
        d = mujoco.MjData(vm)
        view = ViewConstraints(head, K, shape, P, masks)
        linear = []
        lh = []
        soc = []
        sh = []
        cones = []
        jp = np.zeros((3, vm.nv))
        jp2 = np.zeros_like(jp)

        def mapped(J, t, half=False):
            J = np.atleast_2d(J) @ basis
            rr, cc = np.nonzero(J)
            vv = J[rr, cc]
            dyn = cc < dcount
            rows = []
            cols = []
            vals = []
            for frame, weight in [(t, 0.5), (t + 1, 0.5)] if half else [(t, 1.0)]:
                rows.extend(rr[dyn])
                cols.extend(frame * dcount + cc[dyn])
                vals.extend(weight * vv[dyn])
            rows.extend(rr[~dyn])
            cols.extend(n * dcount + cc[~dyn] - dcount)
            vals.extend(vv[~dyn])
            return sparse.csr_matrix((vals, (rows, cols)), shape=(J.shape[0], N))

        def line(J, h, t, half=False):
            linear.append(mapped(J, t, half))
            lh.extend(np.atleast_1d(h))

        def ball(error, J, t, radius):
            soc.append(sparse.vstack([sparse.csr_matrix((1, N)), -mapped(J, t)], format="csr"))
            sh.extend(np.r_[radius, error])
            cones.append(clarabel.SecondOrderConeT(4))

        from steps.sharpa.search_masks import configure

        configure(vm, vq[0], output)
        for sample in range(2 * n - 1):
            half = bool(sample % 2)
            t = sample // 2
            d.qpos[:] = 0.5 * (vq[t] + vq[t + 1]) if half else vq[t]
            mujoco.mj_forward(vm, d)
            if not half:
                for side, s in enumerate(["left", "right"]):
                    if not masks[t, side]:
                        continue
                    cp, cR, Jp, Jr = pose_and_jac(vm, d, s, va)
                    rot = Rotation.from_matrix(R[t, side].T @ cR).as_rotvec()
                    Jr = _log_jacobian(rot) @ R[t, side].T @ Jr
                    ball(cp - P[t, side], Jp, t, acceptance.position_finish_m)
                    ball(rot, Jr, t, acceptance.rotation_finish_rad)
                for value, J, k, kind in view.terms(vm, d, t, va):
                    if measured_view and kind in ["view", "hand_view"]:
                        continue
                    line(J, 0.01398 - value, t)
            for c in d.contact:
                if c.dist >= 0.002 or not allowed_contact(vm, c):
                    continue
                normal = c.frame[:3]
                p1 = c.pos - 0.5 * c.dist * normal
                p2 = c.pos + 0.5 * c.dist * normal
                mujoco.mj_jac(vm, d, jp, None, p1, int(vm.geom_bodyid[c.geom1]))
                mujoco.mj_jac(vm, d, jp2, None, p2, int(vm.geom_bodyid[c.geom2]))
                line(-normal @ (jp2 - jp)[:, va], float(c.dist) - 0.0001, t, half)
        C = sparse.hstack(
            [sparse.kron(sparse.eye(n), sparse.csc_matrix(M)), sparse.csr_matrix((n * pcount, 6))], format="csr"
        )
        difference = sparse.kron(
            sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)), sparse.eye(pcount), format="csr"
        )
        D = difference @ C
        accel = sparse.kron(
            sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)),
            sparse.eye(pcount),
            format="csr",
        )
        A = accel @ C
        dt = (np.diff(times)[:, None] * speed_limits(model, control)[None, :] - 1e-6).ravel()
        previous = np.diff(q[:, control], axis=0).ravel()
        linear.extend([D, -D])
        lh.extend(np.r_[dt - previous, dt + previous])
        limits = model.jnt_range[control]
        lower = np.r_[np.maximum(limits[:, 0][None, :] - q[:, control] + 1e-7, -0.04).ravel(), np.full(6, -0.003)]
        upper = np.r_[np.minimum(limits[:, 1][None, :] - q[:, control] - 1e-7, 0.04).ravel(), np.full(6, 0.003)]
        bounds_map = sparse.vstack(
            [C, sparse.hstack([sparse.csr_matrix((6, n * dcount)), sparse.eye(6)])], format="csr"
        )
        linear.extend([bounds_map, -bounds_map])
        lh.extend(np.r_[upper, -lower])
        G = sparse.vstack([*linear, *soc], format="csc")
        h = np.r_[lh, sh]
        allcones = [clarabel.NonnegativeConeT(len(lh)), *cones]
        zero_bad = np.flatnonzero((np.asarray(G[: len(lh)].getnnz(axis=1)).ravel() == 0) & (np.asarray(lh) < 0))
        if len(zero_bad):
            print("CONIC_IMMOVABLE_GUARDS", [(int(i), float(h[i])) for i in zero_bad[:12]], flush=True)
        H = sparse.diags(np.r_[np.ones(n * dcount), np.full(6, 1000.0)], format="csc") + 2 * (A.T @ A)
        gradient = np.asarray(2 * A.T @ (accel @ q[:, control].ravel())).ravel()
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 120
        settings.tol_feas = 1e-9
        settings.tol_gap_abs = 1e-8
        solution = clarabel.DefaultSolver(sparse.triu(H, format="csc"), gradient, G, h, allcones, settings).solve()
        row = dict(
            iteration=iteration,
            status=str(solution.status),
            acceptance_gates_unchanged=True,
            acceptance_profile=acceptance.report(),
        )
        history.append(row)
        (output / "optimizer.json").write_text(json.dumps(history, indent=2))
        print("CONIC_FINISH_QP", row, flush=True)
        if str(solution.status) not in ["Solved", "AlmostSolved"]:
            break
        step = np.asarray(solution.x)
        candidate = q.copy()
        candidate[:, control] += step[: n * dcount].reshape(n, dcount) @ M.T
        newbases = {}
        for i, s in enumerate(["left", "right"]):
            # Virtual translation axes belong to the rotated static base
            # wrapper. Rebuild the physical base with the WORLD displacement.
            baseR = Rotation.from_quat(np.asarray(bases[s]["quat"])[[1, 2, 3, 0]]).as_matrix()
            delta_world = baseR @ step[n * dcount + 3 * i : n * dcount + 3 * i + 3]
            newbases[s] = dict(pos=(bases[s]["pos"] + delta_world).tolist(), quat=bases[s]["quat"].tolist())
        spec = build(base_poses=newbases, include_floor=False)
        physical = spec.compile()
        flags = int(physical.opt.disableflags)
        physical.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        cp, cR = track(physical, candidate)
        pe = float(np.max(np.linalg.norm(cp - P, axis=-1)[masks]))
        re = float(
            np.max(
                Rotation.from_matrix((R.transpose(0, 1, 3, 2) @ cR).reshape(-1, 3, 3)).magnitude().reshape(n, 2)[masks]
            )
        )
        layout = ViewConstraints(head, K, shape, P, masks).audit(physical, candidate)
        connection = audit_connection(physical, candidate)
        physical.opt.disableflags = flags
        limits_error = float(
            max(0.0, np.max(physical.jnt_range[:, 0] - candidate), np.max(candidate - physical.jnt_range[:, 1]))
        )
        speed = float(np.max(abs(np.diff(candidate, axis=0) / np.diff(times)[:, None]) / speed_limits(physical)))
        am = angular_metrics(cR, R, masks)
        angular = angular_eligible(am, baseline)
        hard_view = (
            max(layout["camera_keepout_max_violation_m"], layout["workspace_envelope_max_violation_m"])
            if measured_view
            else layout["proximal_layout_max_violation_m"]
        )
        acceptable = (
            pe <= acceptance.position_m
            and re <= acceptance.rotation_rad
            and hard_view <= _acceptance_gate(0.014, "layout_m")
            and limits_error <= 1e-5
            and speed <= 1 + 1e-5
            and angular is not False
        )
        contacts = arm_contacts(physical, candidate) if acceptable else dict(arm_collision_samples=None)
        passed = bool(acceptable and contacts["arm_collision_samples"] == 0)
        row.update(
            passed=passed,
            position_max_m=pe,
            rotation_max_rad=re,
            view_max_m=layout["proximal_layout_max_violation_m"],
            native_collision_samples=contacts["arm_collision_samples"],
        )
        (output / "optimizer.json").write_text(json.dumps(history, indent=2))
        np.savez_compressed(output / f"iteration_{iteration}.npz", qpos=candidate, step=step)
        print("CONIC_FINISH_PHYSICAL", row, flush=True)
        record = dict(
            record,
            acceptance_profile=acceptance.report(),
            label="conic_arm_and_static_base_finish",
            passed=passed,
            eef_position_max_m=pe,
            eef_rotation_max_rad=re,
            task_tcp_position_max_m=pe,
            task_tcp_rotation_max_rad=re,
            view_layout=layout,
            angular_metrics=am,
            original_orientation_angular_gate_passed=angular,
            arm_collision_samples=contacts["arm_collision_samples"],
            speed_to_native_cap_ratio=speed,
            conic_finish=True,
            coupling_residual_max_rad=coupling_error(physical, candidate),
            all_joint_speed_max_rad_s=float(np.max(abs(np.diff(candidate, axis=0) / np.diff(times)[:, None]))),
            optimized_final_coordinates_per_frame=dcount,
            physical_coordinate_count=pcount,
            rigid_mount_unchanged=True,
        )
        result = dict(
            result,
            spec=spec,
            model=physical,
            qpos=candidate,
            bases=newbases,
            record=record,
            connection=connection,
            collisions=contacts,
        )
        if rank(result["record"]) < rank(best["record"]):
            best = result
        q = candidate
        model = physical
        if passed:
            break
    return best
