# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Whole-workspace arm/base initialization, followed by full-frame refinement.

Sparse times are used only for numerical initialization, never for output or
acceptance. All frame times and targets remain in the final problem.
"""

import json
import numpy as np
import mujoco
from scipy import sparse
from scipy.spatial.transform import Rotation
from steps.retarget.continuity import _log_jacobian
from steps.recovery17.bounded_step_lowrank import bounded_step


def initialize(vm, initial, arm, base, P, R, times, output):
    n = len(initial)
    frames = np.unique(np.linspace(0, n - 1, min(n, 20)).round().astype(int))
    data = mujoco.MjData(vm)
    flags = int(vm.opt.disableflags)
    vm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    result = np.r_[initial[:, arm].ravel(), np.zeros(12)]
    report = []
    try:
        for side, side_name in enumerate(["left", "right"]):
            aa = arm[side * 7 : side * 7 + 7]
            bb = base[side * 6 : side * 6 + 6]
            qa = np.r_[aa, bb]
            body = vm.body(side_name + "_hand_" + side_name + "_hand_C_MC").id
            jids = np.array([next(j for j in range(vm.njnt) if vm.jnt_qposadr[j] == a) for a in qa])
            va = vm.jnt_dofadr[jids]
            lo = np.r_[np.tile(vm.jnt_range[jids[:7], 0] + 1e-7, len(frames)), vm.jnt_range[jids[7:], 0] + 1e-7]
            hi = np.r_[np.tile(vm.jnt_range[jids[:7], 1] - 1e-7, len(frames)), vm.jnt_range[jids[7:], 1] - 1e-7]
            x = np.r_[initial[frames][:, aa].ravel(), np.zeros(6)]
            prior = x.copy()
            jp = np.zeros((3, vm.nv))
            jr = np.zeros_like(jp)
            N = len(x)

            def evaluate(v):
                residual = []
                rows = []
                cols = []
                vals = []
                ridx = 0
                for k, t in enumerate(frames):
                    data.qpos[:] = initial[t]
                    data.qpos[aa] = v[k * 7 : k * 7 + 7]
                    data.qpos[bb] = v[-6:]
                    mujoco.mj_forward(vm, data)
                    rot = Rotation.from_matrix(R[t, side].T @ data.xmat[body].reshape(3, 3)).as_rotvec()
                    mujoco.mj_jacBody(vm, data, jp, jr, body)
                    J = np.vstack([400 * jp[:, va], 40 * _log_jacobian(rot) @ R[t, side].T @ jr[:, va]])
                    residual.extend(np.r_[400 * (data.xpos[body] - P[t, side]), 40 * rot])
                    rr, cc = np.nonzero(J)
                    rows.extend(rr + ridx)
                    cols.extend(np.where(cc < 7, k * 7 + cc, N - 6 + cc - 7))
                    vals.extend(J[rr, cc])
                    ridx += 6
                J = sparse.csr_matrix((vals, (rows, cols)), shape=(ridx, N))
                weights = np.r_[np.full(N - 6, 0.15), np.full(6, 0.05)]
                return np.r_[residual, weights * (v - prior)], sparse.vstack([J, sparse.diags(weights)], format="csr")

            k = len(frames)
            difference = sparse.diags([-np.ones(k - 1), np.ones(k - 1)], [0, 1], shape=(k - 1, k))
            D = sparse.hstack(
                [sparse.kron(difference, sparse.eye(7)), sparse.csr_matrix(((k - 1) * 7, 6))], format="csr"
            )
            second = sparse.diags([np.ones(k - 2), -2 * np.ones(k - 2), np.ones(k - 2)], [0, 1, 2], shape=(k - 2, k))
            A = sparse.hstack([sparse.kron(second, sparse.eye(7)), sparse.csr_matrix(((k - 2) * 7, 6))], format="csr")
            step_limit = np.repeat(3 * np.diff(times[frames]) - 1e-5, 7)
            evaluations = 0

            def complete(v):
                nonlocal evaluations
                r, J = evaluate(v)
                evaluations += 1
                return np.r_[r, 0.2 * (D @ v), 4 * (A @ v)], sparse.vstack([J, 0.2 * D, 4 * A], format="csr")

            x = np.clip(x, lo + 1e-10, hi - 1e-10)
            trace = []
            for it in range(100):
                rr, J = complete(x)
                cost = float(rr @ rr) / 2
                caps = np.r_[np.full(k * 7, 0.2), [0.03, 0.03, 0.03, 0.07, 0.07, 0.07]]
                delta, diag = bounded_step(J, rr, x, lo, hi, D, step_limit, caps, damping=10.0)
                if delta is None:
                    break
                found = False
                for fraction in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.015625]:
                    trial = x + fraction * delta
                    rnew, _ = complete(trial)
                    new = float(rnew @ rnew) / 2
                    if new < cost - 1e-8:
                        x = trial
                        found = True
                        break
                trace.append(dict(iteration=it, cost=cost, qp=diag, accepted=found))
                if not found or np.max(abs(delta)) < 1e-6:
                    break
            armtrack = np.vstack([np.interp(times, times[frames], x[:-6].reshape(k, 7)[:, j]) for j in range(7)]).T
            assert np.max(abs(np.diff(armtrack, axis=0) / np.diff(times)[:, None])) <= 3 + 1e-5
            result[:-12].reshape(n, 14)[:, side * 7 : side * 7 + 7] = armtrack
            result[-12 + side * 6 : -12 + side * 6 + 6 if side == 0 else None] = x[-6:]
            row = dict(
                side=side_name,
                optimization_frames=frames.tolist(),
                nfev=evaluations,
                cost=cost,
                base_parameters=x[-6:].tolist(),
                original_frames=n,
                hard_speed_constraints=True,
                iterations=trace,
            )
            report.append(row)
            print("COARSE_WORKSPACE_SEED", {k: v for k, v in row.items() if k != "iterations"}, flush=True)
    finally:
        vm.opt.disableflags = flags
    (output / "coarse_initialization.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / "coarse_seed.npz", x=result)
    return result
