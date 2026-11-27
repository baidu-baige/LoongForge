# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Whole-track Panda IK for a physically fixed Sharpa attachment.

Arm joints and one static base pose per arm are optimized together. Finger
coordinates, camera, source clock, and palm world goals remain unchanged.
The exported model has the ordinary 58 joints, with no virtual wrist/base DOFs.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from .panda_scene import build, indices
from .mount_contract import audit_connection
from .rigid_retarget import palm_track, pose_errors, arm_contacts, root_name
from .hand_swap import _standalone_xml
from .backend import upstream_root
from steps import config
from steps.retarget.continuity import _log_jacobian
from steps.retarget.base_search import _is_known_baseline_self_contact
from steps.recovery17.bounded_step_lowrank import bounded_step


def virtual_model(bases):
    spec = build(base_poses=bases, include_floor=False)
    tree = _standalone_xml(spec, upstream_root(), Path(config.MENAGERIE_DIR) / "franka_emika_panda/assets")
    for key in tree.findall("keyframe"):
        tree.remove(key)
    parents = {c: p for p in tree.iter() for c in p}
    for side in ["left", "right"]:
        body = tree.find(f'.//body[@name="{side}_link0"]')
        parent = parents[body]
        wrapper = ET.Element(
            "body", name=side + "_mount", pos=body.attrib.pop("pos", "0 0 0"), quat=body.attrib.pop("quat", "1 0 0 0")
        )
        ET.SubElement(wrapper, "inertial", mass="1", pos="0 0 0", diaginertia=".01 .01 .01")
        for kind, prefix, limit in [("slide", "t", 0.3), ("hinge", "r", np.pi / 3)]:
            for axis, vector in zip("xyz", ["1 0 0", "0 1 0", "0 0 1"]):
                ET.SubElement(
                    wrapper,
                    "joint",
                    name=f"{side}_static_{prefix}{axis}",
                    type=kind,
                    axis=vector,
                    range=f"{-limit} {limit}",
                    limited="true",
                )
        parent.remove(body)
        wrapper.append(body)
        parent.append(wrapper)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    return model


def allowed_contact(model, c):
    b = [model.body(int(model.geom_bodyid[g])).name or "" for g in [c.geom1, c.geom2]]
    if not all(n.startswith(("left_", "right_")) for n in b):
        return False
    if _is_known_baseline_self_contact(*b):
        return False
    if all("_hand_" in n for n in b) and b[0].split("_")[0] == b[1].split("_")[0]:
        return False
    return True


def remap_arms(
    old_scene, old_track, times, output, max_base_candidates=4, target_positions=None, target_rotations=None
):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    P, R = palm_track(old_scene, old_track)
    if target_positions is not None:
        P = np.asarray(target_positions, float)
        R = np.asarray(target_rotations, float)
        assert P.shape == (len(old_track), 2, 3) and R.shape == (len(old_track), 2, 3, 3)
    n = len(old_track)
    data = mujoco.MjData(old_scene)
    data.qpos[:] = old_track[0]
    mujoco.mj_forward(old_scene, data)
    bases = {
        s: dict(
            pos=data.xpos[old_scene.body(s + "_link0").id].copy(),
            quat=data.xquat[old_scene.body(s + "_link0").id].copy(),
        )
        for s in ["left", "right"]
    }
    vm = virtual_model(bases)
    vd = mujoco.MjData(vm)
    # Alternative narrow phase is confined to the search model. Exported
    # physical models keep MuJoCo's default native CCD for independent checks.
    vm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    vm.geom_margin[:] = np.maximum(vm.geom_margin, 0.004)
    old_names = [old_scene.joint(j).name for j in range(old_scene.njnt)]
    track = np.tile(vm.qpos0, (n, 1))
    for j in range(vm.njnt):
        name = vm.joint(j).name
        if name in old_names:
            track[:, int(vm.jnt_qposadr[j])] = old_track[:, int(old_scene.jnt_qposadr[old_scene.joint(name).id])]
    arm = np.r_[indices(vm, "left")["arm_qadr"], indices(vm, "right")["arm_qadr"]]
    base = np.array(
        [
            int(vm.joint(f"{s}_static_{k}{axis}").qposadr[0])
            for s in ["left", "right"]
            for k in ["t", "r"]
            for axis in "xyz"
        ]
    )
    qa = np.r_[arm, base]
    jids = [next(j for j in range(vm.njnt) if vm.jnt_qposadr[j] == a) for a in qa]
    va = vm.jnt_dofadr[jids]
    x = np.r_[track[:, arm].ravel(), np.zeros(12)]
    N = len(x)
    lo = np.r_[np.tile(vm.jnt_range[jids[:14], 0] + 1e-7, n), vm.jnt_range[jids[14:], 0] + 1e-7]
    hi = np.r_[np.tile(vm.jnt_range[jids[:14], 1] - 1e-7, n), vm.jnt_range[jids[14:], 1] - 1e-7]
    D = sparse.hstack(
        [
            sparse.kron(sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)), sparse.eye(14)),
            sparse.csr_matrix(((n - 1) * 14, 12)),
        ],
        format="csr",
    )
    A = sparse.hstack(
        [
            sparse.kron(
                sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)),
                sparse.eye(14),
            ),
            sparse.csr_matrix(((n - 2) * 14, 12)),
        ],
        format="csr",
    )
    bids = [vm.body(root_name(s)).id for s in ["left", "right"]]
    jp = np.zeros((3, vm.nv))
    jr = np.zeros_like(jp)
    jp2 = np.zeros_like(jp)
    full = track.copy()
    history = []
    best = None

    def expand(v):
        full[:, arm] = v[:-12].reshape(n, 14)
        full[:, base] = v[-12:]
        return full

    def evaluate(v, midpoint):
        q = expand(v)
        residual = []
        rowidx = []
        colidx = []
        values = []
        count = 0

        def block(r, J, t, half=False):
            nonlocal count
            r = np.asarray(r).ravel()
            J = np.asarray(J).reshape(len(r), 26)
            residual.extend(r)
            nz = np.nonzero(J)
            rows = nz[0] + count
            cols = nz[1]
            vv = J[nz]
            arm_mask = cols < 14
            for frame, weight in [(t, 0.5), (t + 1, 0.5)] if half else [(t, 1.0)]:
                rowidx.extend(rows[arm_mask])
                colidx.extend(t * 14 + cols[arm_mask] if not half else frame * 14 + cols[arm_mask])
                values.extend(weight * vv[arm_mask])
            rowidx.extend(rows[~arm_mask])
            colidx.extend(n * 14 + cols[~arm_mask] - 14)
            values.extend(vv[~arm_mask])
            count += len(r)

        for sample in range(2 * n - 1 if midpoint else n):
            half = bool(sample % 2) if midpoint else False
            t = sample // 2 if midpoint else sample
            row = 0.5 * (q[t] + q[t + 1]) if half else q[t]
            vd.qpos[:] = row
            mujoco.mj_forward(vm, vd)
            if not half:
                for side, bid in enumerate(bids):
                    err = vd.xpos[bid] - P[t, side]
                    rot = Rotation.from_matrix(R[t, side].T @ vd.xmat[bid].reshape(3, 3)).as_rotvec()
                    mujoco.mj_jacBody(vm, vd, jp, jr, bid)
                    Jp = jp[:, va]
                    Jr = _log_jacobian(rot) @ R[t, side].T @ jr[:, va]
                    block(np.r_[300 * err, 30 * rot], np.vstack([300 * Jp, 30 * Jr]), t)
                    norm = np.linalg.norm(err)
                    if norm > 0.0035:
                        block([8000 * (norm - 0.0035)], 8000 * (err / norm) @ Jp, t)
                    angle = np.linalg.norm(rot)
                    if angle > 0.10:
                        block([500 * (angle - 0.10)], 500 * (rot / angle) @ Jr, t)
            for c in vd.contact:
                if c.dist >= 0.003 or not allowed_contact(vm, c):
                    continue
                g1, g2 = int(c.geom1), int(c.geom2)
                normal = c.frame[:3]
                p1 = c.pos - 0.5 * c.dist * normal
                p2 = c.pos + 0.5 * c.dist * normal
                mujoco.mj_jac(vm, vd, jp, None, p1, int(vm.geom_bodyid[g1]))
                mujoco.mj_jac(vm, vd, jp2, None, p2, int(vm.geom_bodyid[g2]))
                block([2000 * (0.003 - float(c.dist))], -2000 * normal @ (jp2 - jp)[:, va], t, half)
        J = sparse.csr_matrix((values, (rowidx, colidx)), shape=(count, N))
        r = np.asarray(residual)
        prior = np.r_[np.full(n * 14, 0.002), np.tile([0.3, 0.3, 0.3, 0.05, 0.05, 0.05], 2)]
        J = sparse.vstack([J, 0.12 * D, 2 * A, sparse.diags(prior)], format="csr")
        return np.r_[r, 0.12 * (D @ v), 2 * (A @ v), prior * (v - np.r_[track[:, arm].ravel(), np.zeros(12)])], J

    def physical(v, label):
        nonlocal best
        q = expand(v).copy()
        vd.qpos[:] = q[0]
        mujoco.mj_forward(vm, vd)
        actual_bases = {
            s: dict(pos=vd.xpos[vm.body(s + "_link0").id].copy(), quat=vd.xquat[vm.body(s + "_link0").id].copy())
            for s in ["left", "right"]
        }
        spec = build(base_poses=actual_bases, include_floor=False)
        m = spec.compile()
        qq = np.tile(m.qpos0, (n, 1))
        for j in range(m.njnt):
            qq[:, int(m.jnt_qposadr[j])] = q[:, int(vm.joint(m.joint(j).name).qposadr[0])]
        err, rot = pose_errors(m, qq, P, R)
        contacts = arm_contacts(m, qq)
        connection = audit_connection(m, qq)
        speed = float(np.max(abs(D @ v) / np.repeat(np.diff(times), 14)))
        passed = bool(
            err.max() <= 0.005 and rot.max() <= 0.15 and speed <= 3 + 1e-5 and contacts["arm_collision_samples"] == 0
        )
        record = dict(
            label=label,
            passed=passed,
            palm_position_max_m=float(err.max()),
            palm_rotation_max_rad=float(rot.max()),
            arm_speed_max_rad_s=speed,
            arm_collision_samples=contacts["arm_collision_samples"],
            seconds=time.monotonic() - started,
        )
        rank = (not passed, record["arm_collision_samples"], max(err.max() / 0.005, rot.max() / 0.15))
        if best is None or rank < best[0]:
            best = (
                rank,
                dict(
                    spec=spec,
                    model=m,
                    qpos=qq,
                    bases=actual_bases,
                    record=record,
                    collisions=contacts,
                    connection=connection,
                    goal_positions=P,
                    goal_rotations=R,
                ),
            )
        np.savez_compressed(output / (label + ".npz"), x=v, qpos=qq)
        print("COUPLED_PHYSICAL", record, flush=True)
        return passed

    physical(x, "initial")
    for phase, mid in enumerate([False, True], 1):
        radius = 0.15
        damping = 10.0
        for iteration in range(64):
            r, J = evaluate(x, mid)
            cost = float(r @ r) / 2
            caps = np.r_[np.full(n * 14, radius), np.tile([0.02, 0.02, 0.02, 0.06, 0.06, 0.06], 2) * (radius / 0.15)]
            step, info = bounded_step(
                J, r, x, lo, hi, D, np.repeat(3 * np.diff(times) - 1e-6, 14), caps, damping=damping
            )
            row = dict(phase=phase, iteration=iteration, cost=cost, qp=info)
            if step is None:
                row["failed_step"] = True
                history.append(row)
                radius *= 0.5
                damping *= 10
            else:
                ok = False
                for f in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.015625]:
                    trial = x + f * step
                    rr, _ = evaluate(trial, mid)
                    new = float(rr @ rr) / 2
                    if new < cost - 1e-8:
                        x = trial
                        row.update(cost_after=new, fraction=f)
                        ok = True
                        break
                if not ok:
                    radius *= 0.5
                    damping *= 10
                elif row["fraction"] < 0.125:
                    radius = max(0.0002, radius * 0.5)
                    damping *= 2
                else:
                    radius = min(0.15, radius * 1.2)
                    damping = max(10.0, damping * 0.8)
                history.append(row)
            (output / "optimizer.json").write_text(json.dumps(history, indent=2))
            np.savez_compressed(output / "checkpoint.npz", x=x)
            print("COUPLED_ITER", phase, iteration, row.get("cost_after", cost), flush=True)
            if iteration % 12 == 11 or iteration == 63:
                if physical(x, f"phase{phase}_iter{iteration}"):
                    return best[1]
            if radius < 1e-5 or damping > 1e10:
                break
        if physical(x, f"phase{phase}_end"):
            break
    return best[1]
