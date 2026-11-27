# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Joint temporal least-squares initialization; never certifies acceptance."""

import json
from pathlib import Path
import mujoco
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from steps.retarget.continuity import _log_jacobian  # noqa: F401
from steps.sharpa.task_tcp import pose_and_jac, opening_and_jac, tcp_ids  # noqa: F401
from .coupled_place import virtual_model  # noqa: F401
from .panda_scene import indices, build, standalone  # noqa: F401
from .registry import affine, project_seed, speed_limits  # noqa: F401


def initialize(model, q, times, P, R, masks, output, max_nfev=80, *, backend=None):
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
    vm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    data = mujoco.MjData(vm)
    vq = np.tile(vm.qpos0, (n, 1))
    for j in range(model.njnt):
        vq[:, int(vm.joint(model.joint(j).name).qposadr[0])] = q[:, int(model.jnt_qposadr[j])]
    reports = []
    for side, s in enumerate(["left", "right"]):
        tips = tcp_ids(vm, s)
        ids = indices(model, s)
        finger, bf, Mf = affine(model, s)
        adr = np.r_[ids["arm_qadr"], finger]
        M = np.vstack([np.eye(7), np.zeros((len(finger), 7))])
        offset = q[:, adr].copy()
        offset[:, :7] = 0.0
        pcount = len(adr)
        mcount = M.shape[1]
        vlo = []
        vhi = []
        lower = np.full(mcount, -np.inf)
        upper = np.full(mcount, np.inf)
        for j, row in enumerate(M):
            col = np.flatnonzero(row)
            if not len(col):
                continue
            assert len(col) == 1
            k = col[0]
            lo, hi = sorted((model.jnt_range[adr[j]]) / row[k])
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
        for velocity_weight in [300.0, 3000.0, 300000.0]:

            def evaluate(v):
                nonlocal evaluations
                if cache and np.array_equal(cache[0], v):
                    return cache[1], cache[2]
                physical = offset + v[: n * mcount].reshape(n, mcount) @ M.T
                vq[:, control] = physical
                vq[:, base] = v[-6:]
                residual = np.zeros((n, 9))
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
                    sign = 1.0 if s == "right" else -1.0
                    opening = sign * (pts[0] - virtual)
                    jopening = sign * (jacs[0] - jvirtual)
                    approach = virtual - data.xpos[palm]
                    japproach = jvirtual - scratch[:, va]
                    normal = np.cross(opening, approach)
                    jnormal = np.cross(jopening.T, approach).T + np.cross(opening, japproach.T).T
                    target = R[t, side]
                    op = target[:, [0, 2]].T
                    np_ = target[:, [1, 2]].T
                    along_open = target[:, 1] @ opening
                    along_normal = target[:, 0] @ normal
                    residual[t] = np.r_[
                        1000 * (cp - P[t, side]),
                        700 * (op @ opening),
                        20000 * (np_ @ normal),
                        5000 * min(along_open - 0.006, 0.0),
                        100000 * min(along_normal - 0.00005, 0.0),
                    ]
                    jac = (
                        np.vstack(
                            [
                                1000 * jp,
                                700 * op @ jopening,
                                20000 * np_ @ jnormal,
                                5000 * target[:, 1] @ jopening if along_open < 0.006 else np.zeros_like(jp[0]),
                                100000 * target[:, 0] @ jnormal if along_normal < 0.00005 else np.zeros_like(jp[0]),
                            ]
                        )
                        @ local
                    )
                    row, col = np.nonzero(jac)
                    dynamic = col < mcount
                    rr.extend(t * 9 + row)
                    cc.extend(np.where(dynamic, t * mcount + col, n * mcount + col - mcount))
                    vv.extend(jac[row, col])
                Jtask = sparse.csr_matrix((vv, (rr, cc)), shape=(n * 9, N))
                velocity = D0 @ physical.ravel()
                excess = np.maximum(abs(velocity) - caps, 0)
                active = (excess > 0).astype(float) * np.sign(velocity)
                full_res = np.r_[
                    residual.ravel(),
                    velocity_weight * excess,
                    0.5 * (A0 @ physical.ravel()),
                    0.01 * (v[: n * mcount] - motors.ravel()),
                    0.01 * v[-6:],
                ]
                J = sparse.vstack(
                    [Jtask, velocity_weight * sparse.diags(active) @ D, 0.5 * A, 0.01 * posture, 0.01 * base_map],
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
