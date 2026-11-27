# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Correct a legacy off-axis assembly by solving the arm for a rigid palm.

The desired palm motion comes from the previously delivered, smoothed hand
trajectory in the original camera world. Finger coordinates are carried over.
Only arm joints and, in search alternatives, one static base per arm can move.
"""

import json
import time
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .mount_contract import audit_connection
from .panda_scene import build, indices
from steps.retarget.continuity import _log_jacobian


def root_name(side):
    return side + "_hand_" + side + "_hand_C_MC"


def palm_track(model, q):
    data = mujoco.MjData(model)
    P = []
    R = []
    ids = [model.body(root_name(s)).id for s in ["left", "right"]]
    for row in q:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        P.append(data.xpos[ids].copy())
        R.append(data.xmat[ids].reshape(2, 3, 3).copy())
    return np.asarray(P), np.asarray(R)


def pose_errors(model, q, P, R):
    p, r = palm_track(model, q)
    angle = Rotation.from_matrix((R.transpose(0, 1, 3, 2) @ r).reshape(-1, 3, 3)).magnitude().reshape(len(q), 2)
    return np.linalg.norm(p - P, axis=-1), angle


def arm_contacts(model, q):
    """Audit native contacts involving an arm; record internal fingers separately."""
    from steps.retarget.base_search import _is_known_baseline_self_contact

    data = mujoco.MjData(model)
    bad = []
    finger_frames = 0
    for index in range(2 * len(q) - 1):
        midpoint = bool(index % 2)
        t = index // 2
        row = 0.5 * (q[t] + q[t + 1]) if midpoint else q[t]
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        pairs = []
        finger = False
        for c in data.contact:
            if c.dist >= 0:
                continue
            names = [model.body(int(model.geom_bodyid[g])).name or "" for g in [c.geom1, c.geom2]]
            if _is_known_baseline_self_contact(*names):
                continue
            if not all(n.startswith(("left_", "right_")) for n in names):
                continue
            hand = ["_hand_" in n for n in names]
            same_side = names[0].split("_")[0] == names[1].split("_")[0]
            if all(hand) and same_side:
                finger = True
                continue
            pairs.append(dict(bodies=names, penetration_m=-float(c.dist)))
        if pairs:
            bad.append(dict(frame=t, midpoint=midpoint, pairs=pairs))
        if finger and not midpoint:
            finger_frames += 1
    return dict(
        arm_collision_samples=len(bad),
        arm_collisions=bad,
        internal_hand_collision_frames=finger_frames,
        hand_contact_quality_passed=False,
        frame_and_midpoint_samples=2 * len(q) - 1,
    )


def _arm_fit(model, seed, P, R, times, side, anchor, budget=45):
    adr = indices(model, side)["arm_qadr"]
    bid = model.body(root_name(side)).id
    data = mujoco.MjData(model)
    jids = np.array([next(j for j in range(model.njnt) if model.jnt_qposadr[j] == a) for a in adr])
    dofs = model.jnt_dofadr[jids]
    lo = model.jnt_range[jids, 0] + 1e-7
    hi = model.jnt_range[jids, 1] - 1e-7
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    out = seed.copy()
    rot_weight = 5.0

    def solve(t, initial, lower, upper):
        data.qpos[:] = out[t]
        cache = None

        def ev(x):
            nonlocal cache
            if cache is not None and np.array_equal(x, cache[0]):
                return cache[1:]
            data.qpos[adr] = x
            mujoco.mj_forward(model, data)
            actual = data.xmat[bid].reshape(3, 3)
            vector = Rotation.from_matrix(R[t].T @ actual).as_rotvec()
            mujoco.mj_jacBody(model, data, jp, jr, bid)
            residual = np.r_[150 * (data.xpos[bid] - P[t]), rot_weight * vector, 0.002 * (x - initial)]
            J = np.vstack(
                [150 * jp[:, dofs], rot_weight * _log_jacobian(vector) @ R[t].T @ jr[:, dofs], 0.002 * np.eye(7)]
            )
            cache = (x.copy(), residual, J)
            return residual, J

        fitted = least_squares(
            lambda x: ev(x)[0],
            np.clip(initial, lower + 1e-10, upper - 1e-10),
            jac=lambda x: ev(x)[1],
            bounds=(lower, upper),
            max_nfev=budget,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-8,
        )
        return fitted.x

    out[anchor, adr] = solve(anchor, out[anchor, adr], lo, hi)
    for direction in [1, -1]:
        for t in range(anchor + direction, len(out) if direction == 1 else -1, direction):
            prev = t - direction
            dt = abs(times[t] - times[prev])
            previous = out[prev, adr]
            lower = np.maximum(lo, previous - 3 * dt + 1e-7)
            upper = np.minimum(hi, previous + 3 * dt - 1e-7)
            out[t, adr] = solve(t, previous, lower, upper)
    return out


def base_candidates(oldmodel, seed, P, R):
    d = mujoco.MjData(oldmodel)
    d.qpos[:] = seed[0]
    mujoco.mj_forward(oldmodel, d)
    bases = {
        s: dict(pos=d.xpos[oldmodel.body(s + "_link0").id].copy(), quat=d.xquat[oldmodel.body(s + "_link0").id].copy())
        for s in ["left", "right"]
    }
    yield "original_bases", bases
    canonical = build(base_poses=bases, include_floor=False).compile()
    names = [canonical.joint(j).name for j in range(canonical.njnt)]
    oldnames = [oldmodel.joint(j).name for j in range(oldmodel.njnt)]
    track = np.array(seed[:, [oldnames.index(n) for n in names]])
    predicted_p, predicted_R = palm_track(canonical, track)
    for label, t in [("middle", len(seed) // 2), ("first", 0), ("last", len(seed) - 1)]:
        new = {}
        for sideidx, s in enumerate(["left", "right"]):
            delta_R = R[t, sideidx] @ predicted_R[t, sideidx].T
            delta_p = P[t, sideidx] - delta_R @ predicted_p[t, sideidx]
            qb = np.asarray(bases[s]["quat"])
            rb = Rotation.from_quat(qb[[1, 2, 3, 0]]).as_matrix()
            new[s] = dict(
                pos=(delta_R @ bases[s]["pos"] + delta_p).tolist(),
                quat=Rotation.from_matrix(delta_R @ rb).as_quat()[[3, 0, 1, 2]].tolist(),
            )
        yield "aligned_" + label, new


def remap_arms(old_scene, old_track, times, output, max_base_candidates=4):
    """Try deterministic full-frame candidates; never accept a disconnected palm."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    P, R = palm_track(old_scene, old_track)
    old_names = [old_scene.joint(j).name for j in range(old_scene.njnt)]
    history = []
    best = None
    start = time.monotonic()
    for base_label, bases in list(base_candidates(old_scene, old_track, P, R))[:max_base_candidates]:
        spec = build(base_poses=bases, include_floor=False)
        model = spec.compile()
        names = [model.joint(j).name for j in range(model.njnt)]
        q = old_track[:, [old_names.index(n) for n in names]].copy()
        # FK search needs no contacts; native collision inspection uses a
        # separately compiled physical model below, with all original masks.
        search = spec.compile()
        search.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        for sidx, side in enumerate(["left", "right"]):
            choices = []
            for anchor in [len(q) // 2, 0, len(q) - 1]:
                proposal = _arm_fit(search, q, P[:, sidx], R[:, sidx], times, side, anchor)
                errors, angles = pose_errors(model, proposal, P, R)
                rank = max(errors[:, sidx].max() / 0.005, angles[:, sidx].max() / 0.15)
                choices.append((rank, proposal, anchor))
                if rank <= 1:
                    break
            rank, q, anchor = min(choices, key=lambda v: v[0])
            print("RIGID_ARM_SIDE", base_label, side, rank, anchor, flush=True)
        err, ang = pose_errors(model, q, P, R)
        collision = arm_contacts(model, q)
        arms = np.r_[indices(model, "left")["arm_qadr"], indices(model, "right")["arm_qadr"]]
        speed = float(np.max(abs(np.diff(q[:, arms], axis=0) / np.diff(times)[:, None])))
        record = dict(
            bases=base_label,
            palm_position_max_m=float(err.max()),
            palm_rotation_max_rad=float(ang.max()),
            arm_speed_max_rad_s=speed,
            arm_collision_samples=collision["arm_collision_samples"],
            seconds=time.monotonic() - start,
        )
        record["passed"] = bool(
            err.max() <= 0.005 and ang.max() <= 0.15 and speed <= 3.0 + 1e-5 and not collision["arm_collision_samples"]
        )
        history.append(record)
        (output / "search.json").write_text(json.dumps(history, indent=2))
        score = (not record["passed"], record["arm_collision_samples"], max(err.max() / 0.005, ang.max() / 0.15))
        if best is None or score < best[0]:
            best = (score, spec, model, q.copy(), record, collision, bases)
        print("RIGID_ARM_CANDIDATE", record, flush=True)
        if record["passed"]:
            break
    _, spec, model, q, record, collision, bases = best
    audit = audit_connection(model, q)
    return dict(
        spec=spec,
        model=model,
        qpos=q,
        bases=bases,
        record=record,
        collisions=collision,
        connection=audit,
        goal_positions=P,
        goal_rotations=R,
    )
