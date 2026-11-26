# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Framewise IK in independent motor coordinates; never certifies outputs."""

import json
import os
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from steps.sharpa.task_tcp import pose_and_jac, opening_and_jac
from steps.retarget.continuity import _log_jacobian
from .panda_scene import indices
from .registry import affine, project_seed


def initialize(
    model, q, P, R, masks, tips, wristP, wristR, times, output, frame_filter=None, neighbor_speed_bounds=False
):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    q = q.copy()
    d = mujoco.MjData(model)
    flags = int(model.opt.disableflags)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    history = []
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    try:
        for side, s in enumerate(["left", "right"]):
            ids = indices(model, s)
            fadr, bf, Mf = affine(model, s)
            adr = np.r_[ids["arm_qadr"], fadr]
            M = block_diag(np.eye(7), Mf)
            b = np.r_[np.zeros(7), bf]
            va = np.array([int(model.jnt_dofadr[model.joint(model.joint(i).name).id]) for i in adr])
            physical_lo, physical_hi = model.jnt_range[adr].T
            lo = np.full(M.shape[1], -np.inf)
            hi = -lo
            for i, row in enumerate(M):
                nz = np.flatnonzero(row)
                assert len(nz) == 1
                j = nz[0]
                bounds = sorted([(physical_lo[i] + 1e-6 - b[i]) / row[j], (physical_hi[i] - 1e-6 - b[i]) / row[j]])
                lo[j] = max(lo[j], bounds[0])
                hi[j] = min(hi[j], bounds[1])
            global_lo = lo.copy()
            global_hi = hi.copy()
            previous = np.linalg.lstsq(M, q[0, adr] - b, rcond=None)[0]
            sites = [
                model.site(f"{s}_hand_{s}_{digit}_tip").id for digit in ["thumb", "index", "middle", "ring", "pinky"]
            ]
            bid = model.body(f"{s}_hand_{s}_hand_C_MC").id
            for t in range(len(q)):
                if frame_filter is not None and (t, side) not in frame_filter:
                    continue
                if not masks[t, side]:
                    q[t, adr] = b + M @ previous
                    continue
                lo = global_lo.copy()
                hi = global_hi.copy()
                if neighbor_speed_bounds:
                    from .registry import speed_limits

                    speed = speed_limits(model, adr)
                    physical_lower = physical_lo.copy() + 1e-7
                    physical_upper = physical_hi.copy() - 1e-7
                    for neighbor in (
                        [t - 1] if os.environ.get("EGO2ROBOT_CAUSAL_INITIALIZER") == "1" else [t - 1, t + 1]
                    ):
                        if not 0 <= neighbor < len(q):
                            continue
                        margin = speed * abs(times[t] - times[neighbor]) - 1e-7
                        physical_lower = np.maximum(physical_lower, q[neighbor, adr] - margin)
                        physical_upper = np.minimum(physical_upper, q[neighbor, adr] + margin)
                    for i, row in enumerate(M):
                        j = np.flatnonzero(row)[0]
                        bounds = sorted([(physical_lower[i] - b[i]) / row[j], (physical_upper[i] - b[i]) / row[j]])
                        lo[j] = max(lo[j], bounds[0])
                        hi[j] = min(hi[j], bounds[1])
                    if np.any(hi - lo < 1e-9):
                        history.append(dict(frame=t, side=s, bounded_interval_empty=True))
                        continue
                original = np.linalg.lstsq(M, q[t, adr] - b, rcond=None)[0]
                initial = np.clip(0.3 * original + 0.7 * previous, lo + 1e-9, hi - 1e-9)
                cache = None
                if frame_filter is not None:
                    initial = np.clip(
                        previous if os.environ.get("EGO2ROBOT_CAUSAL_INITIALIZER") == "1" and t else original,
                        lo + 1e-10,
                        hi - 1e-10,
                    )

                def evaluate(z):
                    nonlocal cache
                    if cache is not None and np.array_equal(cache[0], z):
                        return cache[1:]
                    d.qpos[:] = q[t]
                    d.qpos[adr] = b + M @ z
                    mujoco.mj_forward(model, d)
                    cp, cR, Jp, Jr = pose_and_jac(model, d, s, va)
                    rot = Rotation.from_matrix(R[t, side].T @ cR).as_rotvec()
                    residual = [400 * (cp - P[t, side]), 40 * rot]
                    jac = [400 * Jp @ M, 40 * _log_jacobian(rot) @ R[t, side].T @ Jr @ M]
                    for sid, target in zip(sites, tips[t, side]):
                        if not np.isfinite(target).all():
                            continue
                        mujoco.mj_jacSite(model, d, jp, None, sid)
                        residual.append(40 * (d.site_xpos[sid] - target))
                        jac.append(40 * jp[:, va] @ M)
                    width, Jw = opening_and_jac(model, d, s, va)
                    residual.append(np.array([10000 * min(width - 0.003, 0)]))
                    jac.append((10000 * Jw @ M if width < 0.003 else np.zeros(len(z)))[None, :])
                    mujoco.mj_jacBody(model, d, jp, jr, bid)
                    palm_rot = Rotation.from_matrix(wristR[t, side].T @ d.xmat[bid].reshape(3, 3)).as_rotvec()
                    residual.extend([3 * (d.xpos[bid] - wristP[t, side]), palm_rot])
                    jac.extend([3 * jp[:, va] @ M, _log_jacobian(palm_rot) @ wristR[t, side].T @ jr[:, va] @ M])
                    residual.extend([0.03 * (z - original), 0.06 * (z - previous)])
                    jac.extend([0.03 * np.eye(len(z)), 0.06 * np.eye(len(z))])
                    cache = (z.copy(), np.concatenate(residual), np.vstack(jac))
                    return cache[1:]

                fit = least_squares(
                    lambda z: evaluate(z)[0],
                    initial,
                    jac=lambda z: evaluate(z)[1],
                    bounds=(lo, hi),
                    max_nfev=120 if frame_filter is not None else 65,
                    ftol=1e-8,
                    xtol=1e-9,
                    gtol=1e-7,
                )
                previous = fit.x.copy()
                q[t, adr] = b + M @ fit.x
                history.append(dict(side=s, frame=t, cost=float(fit.cost), evaluations=fit.nfev))
                if t % 40 == 0:
                    print("MOTOR_IK_INITIALIZER", s, t, len(q), fit.cost, flush=True)
        q = project_seed(model, q, times)
        np.savez_compressed(output / "trajectory.npz", qpos=q, timestamp=times)
        (output / "report.json").write_text(json.dumps(history, indent=2))
        return q
    finally:
        model.opt.disableflags = flags
