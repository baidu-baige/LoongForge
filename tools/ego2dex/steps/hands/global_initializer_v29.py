# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Joint temporal least-squares initialization; never certifies acceptance."""

import json
import os
from pathlib import Path
import mujoco
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from steps.retarget.continuity import _log_jacobian
from steps.sharpa.task_tcp import pose_and_jac, opening_and_jac, tcp_ids  # noqa: F401
from .coupled_place import virtual_model, allowed_contact  # noqa: F401
from .panda_scene import indices, build, standalone  # noqa: F401
from .registry import affine, project_seed, speed_limits  # noqa: F401

_VM = None
_DATA = None
_ALLOWED = None


def _collision_block(vm, data, va, local, rows, start, n, mcount, allowed_contact):  # noqa: F811
    # Contiguous collision samples [start,start+len(rows)); same arithmetic and order as the serial loop.
    res = np.zeros((len(rows), 16))
    vr = []
    vc = []
    values = []
    jp1 = np.zeros((3, vm.nv))
    jp2 = np.zeros_like(jp1)
    for i, row in enumerate(rows):
        sample = start + i
        t = sample // 2
        half = bool(sample % 2)
        data.qpos[:] = row
        mujoco.mj_kinematics(vm, data)
        mujoco.mj_comPos(vm, data)
        mujoco.mj_collision(vm, data)
        contacts = sorted(
            [c for c in data.contact if c.dist < 0.0005 and allowed_contact(vm, c)], key=lambda c: float(c.dist)
        )[:16]
        for k, c in enumerate(contacts):
            normal = c.frame[:3]
            p1 = c.pos - 0.5 * c.dist * normal
            p2 = c.pos + 0.5 * c.dist * normal
            mujoco.mj_jac(vm, data, jp1, None, p1, int(vm.geom_bodyid[c.geom1]))
            mujoco.mj_jac(vm, data, jp2, None, p2, int(vm.geom_bodyid[c.geom2]))
            jac = -10000 * normal @ (jp2 - jp1)[:, va] @ local
            res[i, k] = 10000 * (0.0005 - float(c.dist))
            cols = np.flatnonzero(jac)
            dynamic = cols < mcount
            for frame, weight in [(t, 0.5), (t + 1, 0.5)] if half else [(t, 1.0)]:
                vr.extend([sample * 16 + k] * int(dynamic.sum()))
                vc.extend(frame * mcount + cols[dynamic])
                values.extend(weight * jac[cols[dynamic]])
            vr.extend([sample * 16 + k] * int((~dynamic).sum()))
            vc.extend(n * mcount + cols[~dynamic] - mcount)
            values.extend(jac[cols[~dynamic]])
    return res, vr, vc, values


def _collision_worker(args):
    global _DATA
    if _DATA is None:
        _DATA = mujoco.MjData(_VM)
    res, vr, vc, values = _collision_block(_VM, _DATA, *args, _ALLOWED)
    return res, np.asarray(vr, dtype=np.int64), np.asarray(vc, dtype=np.int64), np.asarray(values, dtype=float)


def initialize(model, q, times, P, R, masks, output, max_nfev=55, *, backend=None):
    pools = []
    try:
        return _initialize(model, q, times, P, R, masks, output, max_nfev, pools, backend)
    finally:
        for pool in pools:
            pool.terminate()
            pool.join()


def _initialize(model, q, times, P, R, masks, output, max_nfev, pools, backend):
    # Preserve public defaults for existing probes; backend overrides stay per call.
    virtual_model = globals()["virtual_model"] if backend is None else backend.virtual_model  # noqa: F811
    indices = globals()["indices"] if backend is None else backend.indices  # noqa: F811
    build = globals()["build"] if backend is None else backend.build  # noqa: F811
    standalone = globals()["standalone"] if backend is None else backend.standalone  # noqa: F811
    affine = globals()["affine"] if backend is None else backend.affine  # noqa: F811
    project_seed = globals()["project_seed"] if backend is None else backend.project_seed  # noqa: F811
    speed_limits = globals()["speed_limits"] if backend is None else backend.speed_limits  # noqa: F811
    pose_and_jac = globals()["pose_and_jac"] if backend is None else backend.pose_and_jac  # noqa: F811
    opening_and_jac = globals()["opening_and_jac"] if backend is None else backend.opening_and_jac  # noqa: F811
    allowed_contact = globals()["allowed_contact"] if backend is None else backend.allowed_contact  # noqa: F811
    if backend is not None:
        model = backend.prepare_model(model)
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    q = project_seed(model, q, times)
    n = len(q)
    d = mujoco.MjData(model)
    flags = int(model.opt.disableflags)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d.qpos[:] = q[0]
    mujoco.mj_forward(model, d)
    bases = {
        s: dict(pos=d.xpos[model.body(s + "_link0").id].copy(), quat=d.xquat[model.body(s + "_link0").id].copy())
        for s in ["left", "right"]
    }
    model.opt.disableflags = flags
    vm = virtual_model(bases, wide=True)
    vm.geom_margin[:] = np.maximum(vm.geom_margin, 0.003)
    data = mujoco.MjData(vm)
    vq = np.tile(vm.qpos0, (n, 1))
    for j in range(model.njnt):
        vq[:, int(vm.joint(model.joint(j).name).qposadr[0])] = q[:, int(model.jnt_qposadr[j])]
    reports = []
    workers = int(os.environ.get("EGO2ROBOT_ELASTIC_WORKERS", "1"))
    pool = None
    if workers > 1:
        global _VM, _ALLOWED
        import multiprocessing

        _VM = vm
        _ALLOWED = allowed_contact
        pool = multiprocessing.get_context("fork").Pool(workers)
        pools.append(pool)
    for side, s in enumerate(["left", "right"]):
        tips = tcp_ids(vm, s)
        ids = indices(model, s)
        finger, bf, Mf = affine(model, s)
        adr = np.r_[ids["arm_qadr"], finger]
        M = block_diag(np.eye(7), Mf)
        offset = np.r_[np.zeros(7), bf]
        pcount = len(adr)
        mcount = M.shape[1]
        vlo = []
        vhi = []
        lower = np.full(mcount, -np.inf)
        upper = np.full(mcount, np.inf)
        for j, row in enumerate(M):
            col = np.flatnonzero(row)
            assert len(col) == 1
            k = col[0]
            lo, hi = sorted((model.jnt_range[adr[j]] - offset[j]) / row[k])
            lower[k] = max(lower[k], lo)
            upper[k] = min(upper[k], hi)
        control = np.array([int(vm.joint(model.joint(int(j)).name).qposadr[0]) for j in adr])
        base = np.array([int(vm.joint(f"{s}_static_{k}{a}").qposadr[0]) for k in ["t", "r"] for a in "xyz"])
        va = np.array(
            [int(vm.joint(model.joint(int(j)).name).dofadr[0]) for j in adr]
            + [int(vm.joint(f"{s}_static_{k}{a}").dofadr[0]) for k in ["t", "r"] for a in "xyz"]
        )
        local = block_diag(M, np.eye(6))
        motors = np.linalg.lstsq(M, (q[:, adr] - offset).T, rcond=None)[0].T
        x = np.r_[motors.ravel(), np.zeros(6)]
        N = len(x)
        xlo = np.r_[np.tile(lower + 1e-8, n), [-0.3] * 3 + [-0.8] * 3]
        xhi = np.r_[np.tile(upper - 1e-8, n), [0.3] * 3 + [0.8] * 3]
        C = sparse.hstack(
            [sparse.kron(sparse.eye(n), sparse.csc_matrix(M)), sparse.csc_matrix((n * pcount, 6))], format="csr"
        )
        D0 = sparse.kron(
            sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)), sparse.eye(pcount), format="csr"
        )
        D = D0 @ C
        A0 = sparse.kron(
            sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)),
            sparse.eye(pcount),
            format="csr",
        )
        A = A0 @ C
        caps = (np.diff(times)[:, None] * speed_limits(model, adr)[None, :] - 1e-7).ravel()
        posture = sparse.hstack([sparse.eye(n * mcount), sparse.csr_matrix((n * mcount, 6))], format="csr")
        base_map = sparse.hstack([sparse.csr_matrix((6, n * mcount)), sparse.eye(6)], format="csr")
        evaluations = 0
        cache = []
        for velocity_weight in [3000.0, 30000.0, 300000.0]:

            def evaluate(v):
                nonlocal evaluations
                if cache and np.array_equal(cache[0], v):
                    return cache[1], cache[2]
                physical = offset + v[: n * mcount].reshape(n, mcount) @ M.T
                vq[:, control] = physical
                vq[:, base] = v[-6:]
                residual = np.zeros((n, 14))
                rr = []
                cc = []
                vv = []
                for t in range(n):
                    if not masks[t, side]:
                        continue
                    data.qpos[:] = vq[t]
                    mujoco.mj_kinematics(vm, data)
                    mujoco.mj_comPos(vm, data)
                    cp, cr, jp, jr = pose_and_jac(vm, data, s, va, ids=tips)
                    rotation = Rotation.from_matrix(R[t, side].T @ cr).as_rotvec()
                    position = cp - P[t, side]

                    def outside_ball(error, radius):
                        length = np.linalg.norm(error)
                        if length <= radius:
                            return np.zeros(3), np.zeros((3, 3))
                        return (1 - radius / length) * error, (1 - radius / length) * np.eye(
                            3
                        ) + radius / length**3 * np.outer(error, error)

                    ep, dp = outside_ball(position, 0.0045)
                    er, dr = outside_ball(rotation, 0.12)
                    log_jac = _log_jacobian(rotation) @ R[t, side].T @ jr
                    residual[t, :3] = 1000 * ep
                    residual[t, 3:6] = 30 * er
                    residual[t, 8:11] = position
                    residual[t, 11:14] = 0.05 * rotation
                    width, jw = opening_and_jac(vm, data, s, va, ids=tips[0])
                    residual[t, 6] = 5000 * min(width - 0.006, 0.0)
                    pts = []
                    jacs = []
                    scratch = np.zeros((3, vm.nv))
                    for site in tips[0]:
                        pts.append(data.site_xpos[site].copy())
                        mujoco.mj_jacSite(vm, data, scratch, None, site)
                        jacs.append(scratch[:, va].copy())
                    palm = tips[1]
                    mujoco.mj_jacBody(vm, data, scratch, None, palm)
                    virtual = 0.7 * pts[1] + 0.3 * pts[2]
                    jvirtual = 0.7 * jacs[1] + 0.3 * jacs[2]
                    opening = pts[0] - virtual
                    jopening = jacs[0] - jvirtual
                    approach = virtual - data.xpos[palm]
                    japproach = jvirtual - scratch[:, va]
                    normal = np.cross(opening, approach)
                    area = np.linalg.norm(normal)
                    jnormal = np.cross(jopening.T, approach).T + np.cross(opening, japproach.T).T
                    jarea = (normal / max(area, 1e-12)) @ jnormal
                    residual[t, 7] = 1000000 * min(area - 0.00005, 0.0)
                    jac = (
                        np.vstack(
                            [
                                1000 * dp @ jp,
                                30 * dr @ log_jac,
                                5000 * jw if width < 0.006 else np.zeros_like(jw),
                                1000000 * jarea if area < 0.00005 else np.zeros_like(jarea),
                                jp,
                                0.05 * log_jac,
                            ]
                        )
                        @ local
                    )
                    row, col = np.nonzero(jac)
                    dynamic = col < mcount
                    rr.extend(t * 14 + row)
                    cc.extend(np.where(dynamic, t * mcount + col, n * mcount + col - mcount))
                    vv.extend(jac[row, col])
                Jtask = sparse.csr_matrix((vv, (rr, cc)), shape=(n * 14, N))
                rows = np.stack(
                    [
                        0.5 * (vq[t] + vq[t + 1]) if sample % 2 else vq[t]
                        for sample in range(2 * n - 1)
                        for t in [sample // 2]
                    ]
                )
                if pool is None:
                    collision_res, vr, vc, values = _collision_block(
                        vm, data, va, local, rows, 0, n, mcount, allowed_contact
                    )
                else:
                    bounds = np.linspace(0, len(rows), workers + 1).astype(int)
                    parts = pool.map(
                        _collision_worker,
                        [(va, local, rows[a:b], int(a), n, mcount) for a, b in zip(bounds[:-1], bounds[1:])],
                    )
                    collision_res = np.vstack([p[0] for p in parts])
                    vr, vc, values = (np.concatenate([p[i] for p in parts]) for i in (1, 2, 3))
                Jcollision = sparse.csr_matrix((values, (vr, vc)), shape=((2 * n - 1) * 16, N))
                velocity = D0 @ physical.ravel()
                excess = np.maximum(abs(velocity) - caps, 0)
                active = (excess > 0).astype(float) * np.sign(velocity)
                full_res = np.r_[
                    residual.ravel(),
                    collision_res.ravel(),
                    velocity_weight * excess,
                    0.5 * (A0 @ physical.ravel()),
                    0.5 * (v[: n * mcount] - motors.ravel()),
                    0.01 * v[-6:],
                ]
                J = sparse.vstack(
                    [
                        Jtask,
                        Jcollision,
                        velocity_weight * sparse.diags(active) @ D,
                        0.5 * A,
                        0.5 * posture,
                        0.01 * base_map,
                    ],
                    format="csr",
                )
                cache[:] = [v.copy(), full_res, J]
                evaluations += 1
                if evaluations % 5 == 0:
                    print(
                        "GLOBAL_INIT",
                        s,
                        evaluations,
                        "task_rms",
                        float(np.sqrt(np.mean(residual**2))),
                        "speed_excess",
                        float(excess.max()),
                        flush=True,
                    )
                return full_res, J

            cache = []
            fit = least_squares(
                lambda v: evaluate(v)[0],
                np.clip(x, xlo + 1e-10, xhi - 1e-10),
                jac=lambda v: evaluate(v)[1],
                bounds=(xlo, xhi),
                method="trf",
                tr_solver="lsmr",
                x_scale="jac",
                max_nfev=max_nfev,
                ftol=1e-7,
                xtol=1e-8,
                gtol=1e-7,
                tr_options=dict(maxiter=120, atol=1e-6, btol=1e-6),
            )
            x = fit.x
            reports.append(
                dict(
                    side=s,
                    velocity_weight=velocity_weight,
                    evaluations=fit.nfev,
                    cost=float(fit.cost),
                    optimality=float(fit.optimality),
                    message=fit.message,
                )
            )
            (output / "optimizer.json").write_text(json.dumps(reports, indent=2))
        vq[:, control] = offset + x[: n * mcount].reshape(n, mcount) @ M.T
        vq[:, base] = x[-6:]
    data.qpos[:] = vq[0]
    mujoco.mj_forward(vm, data)
    bases = {
        s: dict(pos=data.xpos[vm.body(s + "_link0").id].copy(), quat=data.xquat[vm.body(s + "_link0").id].copy())
        for s in ["left", "right"]
    }
    spec = build(base_poses=bases, include_floor=False)
    physical = spec.compile()
    q = np.stack([vq[:, int(vm.joint(physical.joint(j).name).qposadr[0])] for j in range(physical.njnt)], axis=1)
    q = project_seed(physical, q, times)
    import xml.etree.ElementTree as ET

    (output / "seed_scene.xml").write_text(ET.tostring(standalone(spec), encoding="unicode"))
    np.savez_compressed(output / "seed_track.npz", qpos=q, timestamp=times)
    return spec, physical, q
