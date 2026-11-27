# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Sequential conic restoration in real motor coordinates.

Elastic variables exist only in numerical search. Acceptance recomputes the
original task, native limits/speeds/contacts and view criteria without slack.
"""

from .acceptance import gate as _acceptance_gate
import json
import time
import os
from pathlib import Path
import mujoco
import clarabel
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from .panda_scene import build, standalone
from .registry import whole_affine, project_seed, speed_limits, coupling_error
from .coupled_place import virtual_model, allowed_contact
from .mount_contract import audit_connection
from .acceptance import current
from .view_constraints import ViewConstraints
from steps.sharpa.task_tcp import pose_and_jac, track, angular_metrics, opening_and_jac, tcp_ids
from steps.sharpa.rigid_retarget import arm_contacts
from steps.sharpa.search_masks import configure
from steps.retarget.continuity import _log_jacobian
from steps.retarget.joint_pose import angular_eligible

_ELASTIC = None


def _elastic_worker(args):
    return _ELASTIC(*args)


def remap_arms(
    old_scene,
    old_track,
    times,
    output,
    target_positions=None,
    target_rotations=None,
    target_masks=None,
    angular_baseline=None,
    head_poses=None,
    camera_K=None,
    image_shape=None,
    finger_targets=None,
    palm_reference_positions=None,
    palm_reference_rotations=None,
    hand_workspace_points=None,
    trajectory_reference=None,
    optimize_fingers=True,
    finger_target_weight=3.0,
    orientation_body=None,
    fixed_bases=False,
    max_iterations=45,
    fixed_frames=None,
    candidate_view_score=None,
    min_iterations=0,
    view_weight=25.0,
    **unused,
):
    acceptance = current()
    start = time.monotonic()
    out = Path(output)
    out.mkdir(exist_ok=True, parents=True)
    n = len(times)
    nq = old_scene.nq
    geometric = os.environ.get("EGO2ROBOT_TASK_FRAME_SOLVE") == "geometric"
    measured_view = os.environ.get("EGO2ROBOT_VIEW_PROXY_MODE") == "measured_pixels"
    temporal_weight = float(os.environ.get("EGO2ROBOT_TEMPORAL_WEIGHT", "8"))
    jerk_weight = float(os.environ.get("EGO2ROBOT_JERK_WEIGHT", "0"))
    locked = list(fixed_frames or [])
    q = project_seed(old_scene, old_track, times, fixed_rows=locked)
    d = mujoco.MjData(old_scene)
    d.qpos[:] = q[0]
    mujoco.mj_forward(old_scene, d)
    bases = {
        s: dict(
            pos=d.xpos[old_scene.body(s + "_link0").id].copy(), quat=d.xquat[old_scene.body(s + "_link0").id].copy()
        )
        for s in ["left", "right"]
    }
    vm = virtual_model(bases, wide=True)
    vm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    if os.environ.get("EGO2ROBOT_NATIVE_SEARCH") == "1":
        vm.opt.disableflags &= ~(int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD))
    vm.geom_margin[:] = np.maximum(vm.geom_margin, 0.003)
    vd = mujoco.MjData(vm)
    names = [old_scene.joint(i).name for i in range(old_scene.njnt)]
    control = np.array([int(vm.joint(name).qposadr[0]) for name in names])
    base = np.array(
        [int(vm.joint(f"{s}_static_{k}{a}").qposadr[0]) for s in ["left", "right"] for k in ["t", "r"] for a in "xyz"]
    )
    posture_reference = None if trajectory_reference is None else np.asarray(trajectory_reference)
    arm_names = {f"{s}_joint{j}" for s in ["left", "right"] for j in range(1, 8)}
    posture_weights = np.tile([0.5 if name in arm_names else 25.0 for name in names], n)
    if posture_reference is not None:
        assert posture_reference.shape == (n, nq)
    offset, M, motor_names = whole_affine(old_scene)
    total_motor_count = M.shape[1]
    if not optimize_fingers:
        keep = [i for i, name in enumerate(motor_names) if name in arm_names]
        M = M[:, keep]
        motor_names = [motor_names[i] for i in keep]
        arm_positions = [int(old_scene.joint(name).qposadr[0]) for name in motor_names]
        offset = q.copy()
        offset[:, arm_positions] = 0.0
    mcount = M.shape[1]
    bdim = 0 if fixed_bases else 12
    N = n * mcount + bdim
    T = N + 4
    motors = np.linalg.lstsq(M, (q - offset).T, rcond=None)[0].T
    initial = motors.copy()
    x = np.r_[motors.ravel(), np.zeros(bdim)]
    va = vm.jnt_dofadr[
        np.r_[
            [vm.joint(name).id for name in names],
            [vm.joint(f"{s}_static_{k}{a}").id for s in ["left", "right"] for k in ["t", "r"] for a in "xyz"],
        ]
    ]
    local = sparse.block_diag((sparse.csc_matrix(M), sparse.eye(12)), format="csc").toarray()
    vq = np.tile(vm.qpos0, (n, 1))
    jp = np.zeros((3, vm.nv))
    jp2 = np.zeros_like(jp)
    P = np.asarray(target_positions)
    R = np.asarray(target_rotations)
    masks = np.asarray(target_masks, bool)
    view = ViewConstraints(head_poses, camera_K, image_shape, P, masks, workspace_points=hand_workspace_points)
    speed = speed_limits(old_scene)
    ranges = old_scene.jnt_range.copy()
    B = sparse.hstack(
        [sparse.kron(sparse.eye(n), sparse.csc_matrix(M)), sparse.csr_matrix((n * nq, bdim))], format="csr"
    )
    D0 = sparse.kron(
        sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)), sparse.eye(nq), format="csr"
    )
    D = D0 @ B
    A0 = sparse.kron(
        sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)),
        sparse.eye(nq),
        format="csr",
    )
    A = A0 @ B
    J0 = sparse.kron(
        sparse.diags(
            [-np.ones(n - 3), 3 * np.ones(n - 3), -3 * np.ones(n - 3), np.ones(n - 3)], [0, 1, 2, 3], shape=(n - 3, n)
        ),
        sparse.eye(nq),
        format="csr",
    )
    Jerk = J0 @ B
    contact_clearance = float(os.environ.get("EGO2ROBOT_SEARCH_CONTACT_CLEARANCE", ".0005"))
    caps = (np.diff(times)[:, None] * speed[None, :] - 1e-6).ravel()
    history = []
    best = None
    radius = 0.16

    def expand(v):
        physical = offset + v[: n * mcount].reshape(n, mcount) @ M.T
        vq[:, control] = physical
        vq[:, base] = v[-12:] if bdim else 0
        return vq

    configure(vm, expand(x)[0], out)
    sites = {}
    bodies = {}
    tcp = {}

    def site_of(name):
        if name not in sites:
            sites[name] = vm.site(name).id
        return sites[name]

    def body_id(name):
        if name not in bodies:
            bodies[name] = vm.body(name).id
        return bodies[name]

    def tcp_of(s):
        if s not in tcp:
            tcp[s] = tcp_ids(vm, s)
        return tcp[s]

    def mapped(J, t, half=False, extra=None):
        # extra=(column,value) fills that column on every row (former lil edit).
        J = np.atleast_2d(J) @ local
        rr, cc = np.nonzero(J)
        vals = J[rr, cc]
        dynamic = cc < mcount
        rd, cd, dv = rr[dynamic], cc[dynamic], vals[dynamic]
        rows = []
        cols = []
        values = []
        for frame, weight in [(t, 0.5), (t + 1, 0.5)] if half else [(t, 1.0)]:
            rows.append(rd)
            cols.append(frame * mcount + cd)
            values.append(weight * dv)
        if bdim:
            rows.append(rr[~dynamic])
            cols.append(n * mcount + cc[~dynamic] - mcount)
            values.append(vals[~dynamic])
        if extra is not None:
            rows.append(np.arange(J.shape[0]))
            cols.append(np.full(J.shape[0], extra[0]))
            values.append(np.full(J.shape[0], float(extra[1])))
        return sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(J.shape[0], T)
        )

    def evaluate_raw(v, assemble, midpoints, samples):
        rows = expand(v)
        maxima = np.zeros(4)
        lin = []
        lh = []
        soc = []
        sh = []
        cones = []
        soft = []
        soft_r = []
        minimum_width = 1.0
        costs = []
        view.need_jac = assemble

        def line(J, h, t, kind, scale, half=False):
            if assemble:
                lin.append(mapped(J, t, half, (N + kind, -scale)))
                lh.extend(np.atleast_1d(h))

        def ball(err, J, t, tolerance, kind, scale):
            maxima[kind] = max(maxima[kind], (float(np.linalg.norm(err)) - tolerance) / scale)
            if assemble:
                soc.append(
                    sparse.vstack(
                        [sparse.csr_matrix(([-scale], ([0], [N + kind])), shape=(1, T)), -mapped(J, t)], format="csr"
                    )
                )
                sh.extend(np.r_[tolerance, err])
                cones.append(4)

        def direction(vector, J, direction, perpendicular, t, scale, minimum):
            along = float(direction @ vector)
            perp = perpendicular @ vector
            tangent = 0.008
            maxima[1] = max(
                maxima[1], (float(np.linalg.norm(perp)) - tangent * along) / scale, (minimum - along) / scale
            )
            if assemble:
                top = (-tangent * mapped(direction @ J, t)).tolil()
                top[0, N + 1] = -scale
                soc.append(sparse.vstack([top.tocsr(), -mapped(perpendicular @ J, t)], format="csr"))
                sh.extend(np.r_[tangent * along, perp])
                cones.append(3)
                line(-direction @ J, along - minimum, t, 1, scale)
                soft.append(3 / scale * mapped(perpendicular @ J, t))
                soft_r.extend(3 / scale * perp)
            costs.append(float((3 / scale * perp) @ (3 / scale * perp)))

        for sample in samples:
            half = bool(sample % 2) if midpoints else False
            t = sample // 2 if midpoints else sample
            vd.qpos[:] = 0.5 * (rows[t] + rows[t + 1]) if half else rows[t]
            mujoco.mj_forward(vm, vd)
            if not half:
                for side, s in enumerate(["left", "right"]):
                    if not masks[t, side]:
                        continue
                    p, r, Jp, Jr = pose_and_jac(vm, vd, s, va, ids=tcp_of(s), jacobian=assemble)
                    if orientation_body == "palm":
                        bid = body_id(f"{s}_hand_{s}_hand_C_MC")
                        r = vd.xmat[bid].reshape(3, 3)
                        if assemble:
                            mujoco.mj_jacBody(vm, vd, None, jp2, bid)
                            Jr = jp2[:, va].copy()
                    rot = Rotation.from_matrix(R[t, side].T @ r).as_rotvec()
                    if assemble:
                        Jr = _log_jacobian(rot) @ R[t, side].T @ Jr
                    ball(
                        p - P[t, side],
                        Jp,
                        t,
                        float(os.environ.get("EGO2ROBOT_SEARCH_POSITION_RADIUS", str(acceptance.position_search_m))),
                        0,
                        acceptance.position_m,
                    )
                    if geometric and orientation_body is None:
                        pts = []
                        jacs = []
                        for digit in ["thumb", "index", "middle"]:
                            sid = vm.site(f"{s}_hand_{s}_{digit}_tip").id
                            mujoco.mj_jacSite(vm, vd, jp, None, sid)
                            pts.append(vd.site_xpos[sid].copy())
                            jacs.append(jp[:, va].copy())
                        root = vm.body(f"{s}_hand_{s}_hand_C_MC").id
                        mujoco.mj_jacBody(vm, vd, jp, None, root)
                        virtual = 0.7 * pts[1] + 0.3 * pts[2]
                        Jvirtual = 0.7 * jacs[1] + 0.3 * jacs[2]
                        sign = 1 if s == "right" else -1
                        opening = sign * (pts[0] - virtual)
                        Jopening = sign * (jacs[0] - Jvirtual)
                        approach = virtual - vd.xpos[root]
                        Japproach = Jvirtual - jp[:, va]
                        normal = np.cross(opening, approach)
                        Jnormal = np.cross(Jopening.T, approach).T + np.cross(opening, Japproach.T).T
                        target = R[t, side]
                        direction(opening, Jopening, target[:, 1], target[:, [2, 0]].T, t, 0.03, 0.006)
                        direction(normal, Jnormal, target[:, 0], target[:, [2, 1]].T, t, 0.0005, 0.00005)
                        residual = 80 * (p - P[t, side])
                        Jtask = 80 * mapped(Jp, t) if assemble else None
                    else:
                        ball(
                            rot,
                            Jr,
                            t,
                            float(
                                os.environ.get("EGO2ROBOT_SEARCH_ROTATION_RADIUS", str(acceptance.rotation_search_rad))
                            ),
                            1,
                            acceptance.rotation_rad,
                        )
                        residual = np.r_[80 * (p - P[t, side]), 15 * rot]
                        Jtask = sparse.vstack([80 * mapped(Jp, t), 15 * mapped(Jr, t)]) if assemble else None
                    costs.append(float(residual @ residual))
                    if assemble:
                        soft.append(Jtask)
                        soft_r.extend(residual)
                    if palm_reference_positions is not None:
                        bid = body_id(f"{s}_hand_{s}_hand_C_MC")
                        rotp = Rotation.from_matrix(
                            palm_reference_rotations[t, side].T @ vd.xmat[bid].reshape(3, 3)
                        ).as_rotvec()
                        palm_res = np.r_[20 * (vd.xpos[bid] - palm_reference_positions[t, side]), 2 * rotp]
                        costs.append(float(palm_res @ palm_res))
                        if assemble:
                            mujoco.mj_jacBody(vm, vd, jp, jp2, bid)
                            Jrot = _log_jacobian(rotp) @ palm_reference_rotations[t, side].T @ jp2[:, va]
                            soft.append(sparse.vstack([20 * mapped(jp[:, va], t), 2 * mapped(Jrot, t)]))
                            soft_r.extend(palm_res)
                    width, Jw = opening_and_jac(vm, vd, s, va, ids=tcp_of(s)[0], jacobian=assemble)
                    minimum_width = min(minimum_width, width)
                    minimum_opening = float(os.environ.get("EGO2ROBOT_MINIMUM_SEARCH_OPENING", ".002"))
                    maxima[0] = max(maxima[0], (minimum_opening - width) / 0.005)
                    line(-Jw if assemble else None, width - minimum_opening, t, 0, 0.005)
                    if finger_targets is not None:
                        for digit, target in zip(
                            ["thumb", "index", "middle", "ring", "pinky"], finger_targets[t, side]
                        ):
                            if not np.isfinite(target).all():
                                continue
                            site = site_of(f"{s}_hand_{s}_{digit}_tip")
                            tip_res = finger_target_weight * (vd.site_xpos[site] - target)
                            costs.append(float(tip_res @ tip_res))
                            if assemble:
                                mujoco.mj_jacSite(vm, vd, jp, None, site)
                                soft.append(finger_target_weight * mapped(jp[:, va], t))
                                soft_r.extend(tip_res)
                for value, J, k, kind in view.terms(vm, vd, t, va):
                    if measured_view and kind in ["view", "hand_view"]:
                        excess = max(value - _acceptance_gate(0.0137, "layout_m", -0.0003), 0.0)
                        costs.append((view_weight * excess) ** 2)
                        if assemble and excess > 0:
                            soft.append(view_weight * mapped(J, t))
                            soft_r.append(view_weight * excess)
                    else:
                        maxima[2] = max(
                            maxima[2],
                            (value - _acceptance_gate(0.0137, "layout_m", -0.0003))
                            / _acceptance_gate(0.014, "layout_m"),
                        )
                        line(
                            J,
                            _acceptance_gate(0.0137, "layout_m", -0.0003) - value,
                            t,
                            2,
                            _acceptance_gate(0.014, "layout_m"),
                        )
            for c in vd.contact:
                if c.dist >= 0.002 or not allowed_contact(vm, c):
                    continue
                maxima[3] = max(maxima[3], (contact_clearance - float(c.dist)) / 0.003)
                if assemble:
                    normal = c.frame[:3]
                    p1 = c.pos - 0.5 * c.dist * normal
                    p2 = c.pos + 0.5 * c.dist * normal
                    mujoco.mj_jac(vm, vd, jp, None, p1, int(vm.geom_bodyid[c.geom1]))
                    mujoco.mj_jac(vm, vd, jp2, None, p2, int(vm.geom_bodyid[c.geom2]))
                    line(-normal @ (jp2 - jp)[:, va], float(c.dist) - contact_clearance, t, 3, 0.003, half)
        return maxima, lin, lh, soc, sh, cones, soft, soft_r, minimum_width, costs

    def evaluate_chunk(v, assemble, midpoints, samples):
        # Worker side: stack per-chunk blocks to keep pickling cheap.
        maxima, lin, lh, soc, sh, cones, soft, soft_r, minimum_width, costs = evaluate_raw(
            v, assemble, midpoints, samples
        )
        stack = lambda blocks: [sparse.vstack(blocks, format="csr")] if blocks else []
        return maxima, stack(lin), lh, stack(soc), sh, cones, stack(soft), soft_r, minimum_width, costs

    pool = None
    workers = max(1, int(os.environ.get("EGO2ROBOT_ELASTIC_WORKERS", "1")))

    def evaluate(v, assemble=False, midpoints=False):
        # Frames are split into contiguous chunks; recombination keeps the serial
        # order of rows and of the soft-cost summation, so results are bit-identical.
        total = 2 * n - 1 if midpoints else n
        if pool is None:
            chunks = [evaluate_raw(v, assemble, midpoints, range(total))]
        else:
            bounds = np.linspace(0, total, workers + 1).round().astype(int)
            chunks = pool.map(
                _elastic_worker,
                [(v, assemble, midpoints, range(a, b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a],
                chunksize=1,
            )
        maxima = np.zeros(4)
        lin = []
        lh = []
        soc = []
        sh = []
        cones = []
        soft = []
        soft_r = []
        minimum_width = 1.0
        soft_cost = 0.0
        for cmax, clin, clh, csoc, csh, ccones, csoft, csoft_r, cwidth, ccosts in chunks:
            for k in range(4):
                maxima[k] = max(maxima[k], cmax[k])
            lin += clin
            lh += clh
            soc += csoc
            sh += csh
            cones += ccones
            soft += csoft
            soft_r += list(csoft_r)
            minimum_width = min(minimum_width, cwidth)
            for c in ccosts:
                soft_cost += c
        return np.maximum(maxima, 0), dict(
            lin=lin,
            lh=lh,
            soc=soc,
            sh=sh,
            cones=[clarabel.SecondOrderConeT(k) for k in cones],
            soft=soft,
            soft_r=np.asarray(soft_r),
            minimum_width=minimum_width,
            soft_cost=soft_cost,
        )

    def physical(v, label):
        nonlocal best
        rows = expand(v).copy()
        vd.qpos[:] = rows[0]
        mujoco.mj_forward(vm, vd)
        actual_bases = {
            s: dict(pos=vd.xpos[vm.body(s + "_link0").id].copy(), quat=vd.xquat[vm.body(s + "_link0").id].copy())
            for s in ["left", "right"]
        }
        spec = build(base_poses=actual_bases, include_floor=False)
        m = spec.compile()
        qq = rows[:, control].copy()
        flags = int(m.opt.disableflags)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        try:
            cp, cr = track(m, qq)
            if orientation_body == "palm":
                bodydata = mujoco.MjData(m)
                body_ids = [m.body(f"{s}_hand_{s}_hand_C_MC").id for s in ["left", "right"]]
                rotations = []
                for state in qq:
                    bodydata.qpos[:] = state
                    mujoco.mj_forward(m, bodydata)
                    rotations.append(bodydata.xmat[body_ids].reshape(2, 3, 3).copy())
                cr = np.asarray(rotations)
            pe = float(np.max(np.linalg.norm(cp - P, axis=-1)[masks]))
            re = float(
                np.max(
                    Rotation.from_matrix((R.transpose(0, 1, 3, 2) @ cr).reshape(-1, 3, 3))
                    .magnitude()
                    .reshape(n, 2)[masks]
                )
            )
            am = angular_metrics(cr, R, masks)
            angular = angular_eligible(am, angular_baseline) if angular_baseline is not None else None
            layout = view.audit(m, qq)
            connection = audit_connection(m, qq)
        finally:
            m.opt.disableflags = flags
        ratio = float(np.max(abs(np.diff(qq, axis=0) / np.diff(times)[:, None]) / speed))
        limits = float(max(0.0, np.max(ranges[:, 0] - qq), np.max(qq - ranges[:, 1])))
        coupling = coupling_error(m, qq)
        layout_ok = (
            max(layout["camera_keepout_max_violation_m"], layout["workspace_envelope_max_violation_m"])
            <= _acceptance_gate(0.014, "layout_m")
            if measured_view
            else layout["proximal_layout_max_violation_m"] <= _acceptance_gate(0.014, "layout_m")
        )
        candidate = (
            pe <= acceptance.position_m
            and re <= acceptance.rotation_rad
            and angular is not False
            and layout_ok
            and ratio <= 1 + 1e-5
            and limits <= 1e-5
            and coupling < 1e-8
        )
        collisions = arm_contacts(m, qq) if candidate else dict(arm_collision_samples=None)
        passed = bool(candidate and collisions["arm_collision_samples"] == 0)
        record = dict(
            label=label,
            passed=passed,
            eef_position_max_m=pe,
            eef_rotation_max_rad=re,
            task_tcp_position_max_m=pe,
            task_tcp_rotation_max_rad=re,
            angular_metrics=am,
            original_orientation_angular_gate_passed=angular,
            view_layout=layout,
            speed_to_native_cap_ratio=ratio,
            all_joint_speed_max_rad_s=float(np.max(abs(np.diff(qq, axis=0) / np.diff(times)[:, None]))),
            arm_collision_samples=collisions["arm_collision_samples"],
            coupling_residual_max_rad=coupling,
            seconds=time.monotonic() - start,
            optimized_joint_count=nq,
            independent_motor_count=total_motor_count,
            optimized_motor_count=mcount,
            finger_joints_optimized=bool(optimize_fingers),
            goal_profile="frozen_orientation_eq12_articulated_registered_hand_tcp",
            elastic_slacks_used_for_acceptance=False,
        )
        record["acceptance_profile"] = acceptance.report()
        record["task_frame_search_parameterization"] = (
            "unnormalized geometric directions" if geometric else "rotation log"
        )
        if orientation_body == "palm":
            record.update(
                goal_profile="anatomical_palm_orientation_initialization",
                requires_final_articulated_task_verification=True,
            )
        record["view_search_proxy_policy"] = (
            "soft silhouette proxy; original all-frame pixel gates remain mandatory"
            if measured_view
            else "legacy hard silhouette proxy"
        )
        palm_cost = 0.0
        if palm_reference_positions is not None:
            flags = int(m.opt.disableflags)
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            pd = mujoco.MjData(m)
            cost = []
            for t, state in enumerate(qq):
                pd.qpos[:] = state
                mujoco.mj_forward(m, pd)
                for side, s in enumerate(["left", "right"]):
                    if not masks[t, side]:
                        continue
                    bid = m.body(f"{s}_hand_{s}_hand_C_MC").id
                    diff = pd.xpos[bid] - palm_reference_positions[t, side]
                    rotp = Rotation.from_matrix(
                        palm_reference_rotations[t, side].T @ pd.xmat[bid].reshape(3, 3)
                    ).magnitude()
                    cost.append(400 * float(diff @ diff) + 4 * float(rotp**2))
            m.opt.disableflags = flags
            palm_cost = float(np.mean(cost)) if cost else 0.0
        roughness = float(np.mean((A0 @ qq.ravel()) ** 2))
        jerk_roughness = float(np.mean((J0 @ qq.ravel()) ** 2))
        record.update(
            soft_palm_fit_cost=palm_cost,
            temporal_roughness=roughness,
            jerk_roughness=jerk_roughness,
            temporal_weight=temporal_weight,
            jerk_weight=jerk_weight,
        )
        posture_cost = (
            0.0
            if posture_reference is None
            else float(np.mean(posture_weights * (qq - posture_reference).ravel() ** 2))
        )
        record["posture_reference_cost"] = posture_cost
        hard_view = (
            max(layout["camera_keepout_max_violation_m"], layout["workspace_envelope_max_violation_m"])
            if measured_view
            else layout["proximal_layout_max_violation_m"]
        )
        gate_ratio = max(
            pe / acceptance.position_m,
            re / acceptance.rotation_rad,
            hard_view / _acceptance_gate(0.014, "layout_m"),
            ratio,
            limits / 1e-5,
            coupling / 1e-8,
        )
        collision_count = collisions["arm_collision_samples"]
        if collision_count:
            gate_ratio = max(gate_ratio, 1.0 + min(collision_count, 100) / 100.0)
        visual_score = candidate_view_score(spec, qq, label) if passed and candidate_view_score is not None else None
        if visual_score is not None:
            record["search_only_sampled_pixel_score"] = visual_score
        rank = (
            not passed,
            visual_score
            if visual_score is not None
            else palm_cost + temporal_weight * roughness + jerk_weight * jerk_roughness + posture_cost
            if passed
            else gate_ratio,
            float("inf") if collision_count is None else collision_count,
        )
        result = dict(
            spec=spec,
            model=m,
            qpos=qq,
            bases=actual_bases,
            record=record,
            connection=connection,
            collisions=collisions,
            goal_positions=P,
            goal_rotations=R,
            target_masks=masks,
        )
        if best is None or rank < best[0]:
            best = (rank, result)
            import xml.etree.ElementTree as ET

            (out / "best_record.json").write_text(json.dumps(record, indent=2))
            (out / "best_scene.xml").write_text(ET.tostring(standalone(spec), encoding="unicode"))
            np.savez_compressed(out / "best_track.npz", qpos=qq, timestamp=times)
        np.savez_compressed(out / (label + ".npz"), qpos=qq, x=v)
        (out / "latest_record.json").write_text(json.dumps(record, indent=2))
        import xml.etree.ElementTree as ET

        (out / "latest_scene.xml").write_text(ET.tostring(standalone(spec), encoding="unicode"))
        np.savez_compressed(out / "latest_track.npz", qpos=qq, timestamp=times)
        print("ELASTIC_PHYSICAL", record, flush=True)
        return passed

    physical(x, "initial")
    slackweights = np.array([1.0, 1.0, 1.0, 2.0])
    diag = np.r_[np.ones(n * mcount) * 0.05, np.tile([0.1, 0.1, 0.1, 0.01, 0.01, 0.01], 2)[:bdim], slackweights * 1e5]
    smooth = sparse.hstack([A, sparse.csr_matrix((A.shape[0], 4))], format="csr")
    jerk = sparse.hstack([Jerk, sparse.csr_matrix((Jerk.shape[0], 4))], format="csr")
    posture_map = sparse.hstack([B, sparse.csr_matrix((B.shape[0], 4))], format="csc")
    posture_hessian = (
        posture_map.T @ sparse.diags(posture_weights) @ posture_map if posture_reference is not None else None
    )
    if workers > 1:
        global _ELASTIC
        import multiprocessing

        _ELASTIC = evaluate_chunk
        pool = multiprocessing.get_context("fork").Pool(workers)
    try:
        for iteration in range(max_iterations):
            mid = iteration >= 8
            maximum, parts = evaluate(x, True, mid)
            merit = float(np.dot(maximum * slackweights, maximum))
            qphysical = offset + x[: n * mcount].reshape(n, mcount) @ M.T
            score = (
                merit * 1e5
                + parts["soft_cost"]
                + temporal_weight * float(np.sum((A0 @ qphysical.ravel()) ** 2))
                + jerk_weight * float(np.sum((J0 @ qphysical.ravel()) ** 2))
            )
            if posture_reference is not None:
                score += float(np.sum(posture_weights * (qphysical - posture_reference).ravel() ** 2))
            velocity = np.diff(qphysical, axis=0).ravel()
            margin = np.full(nq, 1e-8)
            if not optimize_fingers:
                margin[np.setdiff1d(np.arange(nq), arm_positions)] = -1e-7
            lower = (ranges[:, 0][None, :] - qphysical + margin[None, :]).ravel()
            upper = (ranges[:, 1][None, :] - qphysical - margin[None, :]).ravel()
            lin = list(parts["lin"])
            lh = list(parts["lh"])
            Dfull = sparse.hstack([D, sparse.csr_matrix((D.shape[0], 4))])
            Bfull = sparse.hstack([B, sparse.csr_matrix((B.shape[0], 4))])
            lin.extend([Dfull, -Dfull, Bfull, -Bfull])
            lh.extend(np.r_[caps - velocity, caps + velocity, upper, -lower])
            trust = np.r_[
                np.full(n * mcount, radius),
                np.tile([0.025, 0.025, 0.025, 0.10, 0.10, 0.10], 2)[:bdim] * (radius / 0.16),
            ]
            eye_n = sparse.hstack([sparse.eye(N), sparse.csr_matrix((N, 4))], format="csr")
            nonnegative = sparse.hstack([sparse.csr_matrix((4, N)), -sparse.eye(4)])
            lin.extend([eye_n, -eye_n, nonnegative])
            lh.extend(np.r_[trust, trust, np.zeros(4)])
            if locked:
                trust[: n * mcount].reshape(n, mcount)[locked] = 0.0
                lh[-(2 * N + 4) :] = np.r_[trust, trust, np.zeros(4)].tolist()
            H = sparse.diags(diag, format="csc") + temporal_weight * (smooth.T @ smooth) + jerk_weight * (jerk.T @ jerk)
            gradient = np.asarray(
                temporal_weight * smooth.T @ (A0 @ qphysical.ravel()) + jerk_weight * jerk.T @ (J0 @ qphysical.ravel())
            ).ravel()
            if posture_reference is not None:
                H = H + posture_hessian
                gradient += np.asarray(
                    posture_map.T @ (posture_weights * (qphysical - posture_reference).ravel())
                ).ravel()
            if parts["soft"]:
                Js = sparse.vstack(parts["soft"], format="csc")
                H = H + Js.T @ Js
                gradient += np.asarray(Js.T @ parts["soft_r"]).ravel()
            G = sparse.vstack([*lin, *parts["soc"]], format="csc")
            h = np.r_[lh, parts["sh"]]
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            settings.max_iter = 120
            settings.tol_feas = 1e-8
            settings.tol_gap_abs = 1e-7
            solution = clarabel.DefaultSolver(
                sparse.triu(H, format="csc"),
                gradient,
                G,
                h,
                [clarabel.NonnegativeConeT(len(lh)), *parts["cones"]],
                settings,
            ).solve()
            record = dict(
                iteration=iteration, qp=str(solution.status), merit=merit, violations=maximum.tolist(), radius=radius
            )
            if str(solution.status) not in ["Solved", "AlmostSolved"]:
                record["failed_step"] = True
                radius *= 0.5
            else:
                step = np.asarray(solution.x)[:N]
                accepted = False
                for factor in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                    trial = x + factor * step
                    try:
                        values, trial_parts = evaluate(trial, False, mid)
                    except ValueError:
                        continue
                    new = float(np.dot(values * slackweights, values))
                    trialq = offset + trial[: n * mcount].reshape(n, mcount) @ M.T
                    newscore = (
                        new * 1e5
                        + trial_parts["soft_cost"]
                        + temporal_weight * float(np.sum((A0 @ trialq.ravel()) ** 2))
                        + jerk_weight * float(np.sum((J0 @ trialq.ravel()) ** 2))
                    )
                    if posture_reference is not None:
                        newscore += float(np.sum(posture_weights * (trialq - posture_reference).ravel() ** 2))
                    if newscore < score - max(1e-8, score * 1e-6):
                        x = trial
                        record.update(merit_after=new, fraction=factor)
                        accepted = True
                        break
                if not accepted:
                    radius *= 0.5
                elif record["fraction"] == 1:
                    radius = min(0.25, radius * 1.15)
            history.append(record)
            (out / "optimizer.json").write_text(json.dumps(history, indent=2))
            np.savez_compressed(out / "checkpoint.npz", x=x)
            print("ELASTIC_ITER", record, flush=True)
            if iteration % 3 == 2 or radius < 0.0001:
                if (
                    physical(x, f"iter_{iteration}")
                    and iteration + 1 >= min_iterations
                    and (
                        candidate_view_score is None
                        or best[1]["record"].get("search_only_sampled_pixel_score", 2.0) <= 0.95
                    )
                ):
                    return best[1]
            if radius < 1e-5:
                break
        physical(x, "end")
        return best[1]
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
