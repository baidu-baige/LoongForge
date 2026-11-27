# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Framewise initialization only; final whole-track gates stay unchanged."""

import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .panda_scene import indices
from .task_tcp import pose_and_jac, opening_and_jac
from steps.retarget.continuity import _log_jacobian


def initialize(model, q, P, R, masks, finger_targets, wristP, wristR, output):
    output = Path(output)
    output.mkdir(exist_ok=True)
    q = q.copy()
    d = mujoco.MjData(model)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    history = []
    for side, s in enumerate(["left", "right"]):
        ids = indices(model, s)
        adr = np.r_[ids["arm_qadr"], ids["finger_qadr"]]
        jids = np.array([next(j for j in range(model.njnt) if model.jnt_qposadr[j] == a) for a in adr])
        va = model.jnt_dofadr[jids]
        lo, hi = model.jnt_range[jids].T
        lo = lo + 1e-6
        hi = hi - 1e-6
        bid = model.body(f"{s}_hand_{s}_hand_C_MC").id
        previous = q[0, adr].copy()
        sites = [model.site(f"{s}_hand_{s}_{digit}_tip").id for digit in ["thumb", "index", "middle", "ring", "pinky"]]
        for t in range(len(q)):
            if not masks[t, side]:
                continue
            original = q[t, adr].copy()
            initial = np.clip(0.5 * original + 0.5 * previous, lo, hi)
            cache = None
            goalwidth = float(
                np.linalg.norm(
                    finger_targets[t, side, 0] - 0.7 * finger_targets[t, side, 1] - 0.3 * finger_targets[t, side, 2]
                )
            )

            def evaluate(x):
                nonlocal cache
                if cache is not None and np.array_equal(cache[0], x):
                    return cache[1:]
                d.qpos[:] = q[t]
                d.qpos[adr] = x
                mujoco.mj_forward(model, d)
                cp, cR, Jp, Jr = pose_and_jac(model, d, s, va)
                rot = Rotation.from_matrix(R[t, side].T @ cR).as_rotvec()
                residual = [300 * (cp - P[t, side]), 30 * rot]
                jac = [300 * Jp, 30 * _log_jacobian(rot) @ R[t, side].T @ Jr]
                for sid, target in zip(sites, finger_targets[t, side]):
                    if not np.isfinite(target).all():
                        continue
                    mujoco.mj_jacSite(model, d, jp, None, sid)
                    residual.append(100 * (d.site_xpos[sid] - target))
                    jac.append(100 * jp[:, va])
                width, Jwidth = opening_and_jac(model, d, s, va)
                if np.isfinite(goalwidth):
                    residual.append(np.array([100 * (width - goalwidth)]))
                    jac.append(100 * Jwidth[None])
                residual.append(np.array([10000 * min(width - 0.003, 0.0)]))
                jac.append((10000 * Jwidth if width < 0.003 else np.zeros_like(Jwidth))[None])
                mujoco.mj_jacBody(model, d, jp, jr, bid)
                rootrot = Rotation.from_matrix(wristR[t, side].T @ d.xmat[bid].reshape(3, 3)).as_rotvec()
                residual.extend(
                    [3 * (d.xpos[bid] - wristP[t, side]), rootrot, 0.15 * (x - original), 0.25 * (x - previous)]
                )
                jac.extend(
                    [
                        3 * jp[:, va],
                        _log_jacobian(rootrot) @ wristR[t, side].T @ jr[:, va],
                        0.15 * np.eye(len(x)),
                        0.25 * np.eye(len(x)),
                    ]
                )
                rr = np.concatenate(residual)
                jj = np.vstack(jac)
                cache = (x.copy(), rr, jj)
                return rr, jj

            try:
                fit = least_squares(
                    lambda x: evaluate(x)[0],
                    initial,
                    jac=lambda x: evaluate(x)[1],
                    bounds=(lo, hi),
                    max_nfev=45,
                    ftol=1e-6,
                    xtol=1e-7,
                    gtol=1e-6,
                )
            except ValueError as error:
                if "Degenerate zero-width" not in str(error):
                    raise
                history.append(dict(side=s, frame=t, degenerate_seed=True))
                continue
            q[t, adr] = fit.x
            previous = fit.x.copy()
            history.append(dict(side=s, frame=t, cost=float(fit.cost), evaluations=fit.nfev))
            if t % 80 == 0:
                print("TASK_FRAME_PRESEED", s, t, len(q), fit.cost, flush=True)
        np.savez_compressed(output / "trajectory.npz", qpos=q)
    (output / "report.json").write_text(json.dumps(history, indent=2))
    return q
