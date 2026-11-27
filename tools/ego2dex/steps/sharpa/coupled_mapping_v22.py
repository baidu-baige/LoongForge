# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Whole-track Panda IK for a physically fixed Sharpa attachment.

Arm joints and one static base pose per arm are optimized together. Finger
coordinates, camera, source clock, and palm world goals remain unchanged.
The exported model has the ordinary 58 joints, with no virtual wrist/base DOFs.
"""

from steps.hands.acceptance import gate as _acceptance_gate
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
from .task_tcp import pose_and_jac as task_pose_and_jac, track as task_track, angular_metrics, opening_and_jac
from steps.retarget.joint_pose import angular_eligible


def virtual_model(bases, wide=False):
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
        for kind, prefix, limit in [("slide", "t", 0.6 if wide else 0.3), ("hinge", "r", np.pi if wide else np.pi / 3)]:
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
    old_scene,
    old_track,
    times,
    output,
    max_base_candidates=4,
    target_positions=None,
    target_rotations=None,
    posture_sign=1,
    initialization="existing",
    warm_seed=None,
    optimize_fingers=True,
    finger_targets=None,
    palm_reference_positions=None,
    palm_reference_rotations=None,
    target_masks=None,
    angular_baseline=None,
    joint_warm_seed=None,
):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kernel_source_sha256 = __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()
    P, R = palm_track(old_scene, old_track)
    if target_positions is not None:
        P = np.asarray(target_positions, float)
        R = np.asarray(target_rotations, float)
        assert P.shape == (len(old_track), 2, 3) and R.shape == (len(old_track), 2, 3, 3)
    n = len(old_track)
    data = mujoco.MjData(old_scene)
    data.qpos[:] = old_track[0]
    mujoco.mj_forward(old_scene, data)
    wristP, wristR = palm_track(old_scene, old_track)
    if palm_reference_positions is not None:
        wristP = np.asarray(palm_reference_positions)
        wristR = np.asarray(palm_reference_rotations)
    masks = np.ones((n, 2), bool) if target_masks is None else np.asarray(target_masks, bool)
    bases = {
        s: dict(
            pos=data.xpos[old_scene.body(s + "_link0").id].copy(),
            quat=data.xquat[old_scene.body(s + "_link0").id].copy(),
        )
        for s in ["left", "right"]
    }
    # A new physical tool requires an arm/base initialization for its own TCP.
    # Keeping an old parallel-gripper seed at elbow/wrist limits traps local
    # IK. Start at a well-conditioned Panda posture, and place each *static*
    # base so that the fixed Sharpa palm matches the middle source frame.
    home = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])
    homes = {s: home.copy() for s in ["left", "right"]}
    if initialization.startswith("swivel"):
        homes["left"][2] = -1.2 * posture_sign
        homes["right"][2] = 1.2 * posture_sign
    neutral = {s: dict(pos=[0, 0, 0], quat=[1, 0, 0, 0]) for s in ["left", "right"]}
    hm = build(base_poses=neutral, include_floor=False).compile()
    hd = mujoco.MjData(hm)
    for side in ["left", "right"]:
        hd.qpos[indices(hm, side)["arm_qadr"]] = homes[side]
    mujoco.mj_forward(hm, hd)
    anchor = n // 2
    for sideidx, side in enumerate(["left", "right"]):
        if initialization == "existing":
            continue
        bid = hm.body(root_name(side)).id
        F = hd.xmat[bid].reshape(3, 3)
        offset = hd.xpos[bid]
        targetR = (
            Rotation.from_matrix(R[:, sideidx]).mean().as_matrix() if initialization == "center" else R[anchor, sideidx]
        )
        targetP = (
            0.5 * (P[:, sideidx].min(axis=0) + P[:, sideidx].max(axis=0))
            if initialization == "center"
            else P[anchor, sideidx]
        )
        rotation = targetR @ F.T
        bases[side] = dict(pos=targetP - rotation @ offset, quat=Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]])
    (output / "static_base_initialization.json").write_text(
        json.dumps(
            dict(
                method="alternative redundant elbow posture; fixed-base inverse FK at middle frame",
                anchor_frame=anchor,
                home_arm_joints={s: h.tolist() for s, h in homes.items()},
                target_positions_and_rotations_unchanged=True,
            ),
            indent=2,
        )
    )
    vm = virtual_model(bases, wide=initialization in ("wide", "swivelwide", "existing"))
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
    if initialization != "existing":
        track[:, arm] = np.r_[homes["left"], homes["right"]]
    base = np.array(
        [
            int(vm.joint(f"{s}_static_{k}{axis}").qposadr[0])
            for s in ["left", "right"]
            for k in ["t", "r"]
            for axis in "xyz"
        ]
    )
    control = np.array([int(vm.joint(name).qposadr[0]) for name in old_names]) if optimize_fingers else arm
    jcount = len(control)
    qa = np.r_[control, base]
    jids = [next(j for j in range(vm.njnt) if vm.jnt_qposadr[j] == a) for a in qa]
    va = vm.jnt_dofadr[jids]
    x = np.r_[track[:, arm].ravel(), np.zeros(12)]
    from .coarse_seed_qp import initialize

    if warm_seed is None and initialization != "existing":
        x = initialize(vm, track, arm, base, wristP, wristR, times, output)
    elif warm_seed is None:
        pass
    else:
        import io
        import zipfile

        for retry in range(5):
            try:
                x = np.load(io.BytesIO(Path(warm_seed).read_bytes()))["x"].copy()
                break
            except (zipfile.BadZipFile, EOFError, ValueError):
                time.sleep(0.2)
        assert x.shape == (n * 14 + 12,)
        (output / "warm_seed.json").write_text(
            json.dumps(dict(source=str(warm_seed), initialization=initialization, posture_sign=posture_sign), indent=2)
        )
    if optimize_fingers:
        expanded_seed = track[:, control].copy()
        arm_columns = [int(np.flatnonzero(control == a)[0]) for a in arm]
        expanded_seed[:, arm_columns] = x[:-12].reshape(n, 14)
        x = np.r_[expanded_seed.ravel(), x[-12:]]
    if joint_warm_seed is not None:
        x = np.load(joint_warm_seed)["x"].copy()
        assert x.shape == (n * jcount + 12,)
        (output / "joint_warm_seed.json").write_text(
            json.dumps(
                dict(
                    source=str(joint_warm_seed),
                    sha256=__import__("hashlib").sha256(Path(joint_warm_seed).read_bytes()).hexdigest(),
                ),
                indent=2,
            )
        )
    N = len(x)
    lo = np.r_[np.tile(vm.jnt_range[jids[:jcount], 0] + 1e-7, n), vm.jnt_range[jids[jcount:], 0] + 1e-7]
    hi = np.r_[np.tile(vm.jnt_range[jids[:jcount], 1] - 1e-7, n), vm.jnt_range[jids[jcount:], 1] - 1e-7]
    D = sparse.hstack(
        [
            sparse.kron(sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)), sparse.eye(jcount)),
            sparse.csr_matrix(((n - 1) * jcount, 12)),
        ],
        format="csr",
    )
    A = sparse.hstack(
        [
            sparse.kron(
                sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)),
                sparse.eye(jcount),
            ),
            sparse.csr_matrix(((n - 2) * jcount, 12)),
        ],
        format="csr",
    )
    # Framewise numerical seeds need not obey velocity limits. Project them
    # into the identical joint/3rad/s feasible set before trust-region steps;
    # otherwise a small trust region can make every initial QP infeasible.
    velocity_caps = np.repeat(3 * np.diff(times) - 1e-6, jcount)
    if np.max(abs(D @ x) - velocity_caps) > 1e-7 or np.max(lo - x) > 1e-7 or np.max(x - hi) > 1e-7:
        import clarabel

        length = n * jcount
        difference = D[:, :length]
        eye = sparse.eye(length, format="csc")
        constraints = sparse.vstack([difference, -difference, eye, -eye], format="csc")
        upper = np.r_[velocity_caps, velocity_caps, hi[:length], -lo[:length]]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 150
        settings.tol_gap_abs = 1e-8
        settings.tol_feas = 1e-9
        settings.tol_gap_rel = 1e-9
        solution = clarabel.DefaultSolver(
            eye, -x[:length], constraints, upper, [clarabel.NonnegativeConeT(len(upper))], settings
        ).solve()
        if str(solution.status) not in ("Solved", "AlmostSolved"):
            raise RuntimeError("Initial speed projection failed: " + str(solution.status))
        original = x.copy()
        x[:length] = np.asarray(solution.x)
        assert np.max(abs(D @ x) - velocity_caps) < 1e-5 and np.max(lo - x) < 1e-5 and np.max(x - hi) < 1e-5
        np.savez_compressed(output / "speed_projected_initial.npz", x=x, unprojected_x=original)
        (output / "speed_projection.json").write_text(
            json.dumps(
                dict(
                    status=str(solution.status),
                    speed_max=float(np.max(abs(D @ x) / np.repeat(np.diff(times), jcount))),
                    max_adjustment=float(np.max(abs(x - original))),
                    same_limits_as_final=True,
                ),
                indent=2,
            )
        )
        print("INITIAL_SPEED_FEASIBILITY_PROJECTED", flush=True)
    joint_prior = np.array([0.2 if a in arm else 2.0 for a in control])
    acc_weights = np.array([2.0 if a in arm else 6.0 for a in control])
    weighted_A = sparse.diags(np.tile(acc_weights, n - 2)) @ A
    bids = [vm.body(root_name(s)).id for s in ["left", "right"]]
    jp = np.zeros((3, vm.nv))
    jr = np.zeros_like(jp)
    jp2 = np.zeros_like(jp)
    full = track.copy()
    history = []
    best = None
    latest_guard = None
    latest_violation = 0.0
    tips = [
        vm.site(f"{s}_hand_{s}_{digit}_tip").id
        for s in ["left", "right"]
        for digit in ["thumb", "index", "middle", "ring", "pinky"]
    ]

    def expand(v):
        full[:, control] = v[:-12].reshape(n, jcount)
        full[:, base] = v[-12:]
        return full

    from .search_masks import configure

    mask_proof = configure(vm, expand(x)[0].copy(), output)
    print("SEARCH_CONTACT_MASK_EQUIVALENCE", mask_proof, flush=True)

    def evaluate(v, midpoint):
        nonlocal latest_guard, latest_violation
        q = expand(v)
        residual = []
        rowidx = []
        colidx = []
        values = []
        count = 0
        grows = []
        gcols = []
        gvalues = []
        gu = []
        violations = []

        def guard(J, t, upper, half=False):
            J = np.asarray(J).reshape(-1, jcount + 12)
            start = len(gu)
            gu.extend(np.asarray(upper).ravel())
            rr, cc = np.nonzero(J)
            vv = J[rr, cc]
            dynamic = cc < jcount
            for frame, weight in [(t, 0.5), (t + 1, 0.5)] if half else [(t, 1.0)]:
                grows.extend(rr[dynamic] + start)
                gcols.extend(frame * jcount + cc[dynamic])
                gvalues.extend(weight * vv[dynamic])
            grows.extend(rr[~dynamic] + start)
            gcols.extend(n * jcount + cc[~dynamic] - jcount)
            gvalues.extend(vv[~dynamic])

        def block(r, J, t, half=False):
            nonlocal count
            r = np.asarray(r).ravel()
            J = np.asarray(J).reshape(len(r), jcount + 12)
            residual.extend(r)
            nz = np.nonzero(J)
            rows = nz[0] + count
            cols = nz[1]
            vv = J[nz]
            arm_mask = cols < jcount
            for frame, weight in [(t, 0.5), (t + 1, 0.5)] if half else [(t, 1.0)]:
                rowidx.extend(rows[arm_mask])
                colidx.extend(frame * jcount + cols[arm_mask])
                values.extend(weight * vv[arm_mask])
            rowidx.extend(rows[~arm_mask])
            colidx.extend(n * jcount + cols[~arm_mask] - jcount)
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
                    mujoco.mj_jacBody(vm, vd, jp, jr, bid)
                    rootJp = jp[:, va].copy()
                    rootJr = jr[:, va].copy()
                    rootErr = vd.xpos[bid] - wristP[t, side]
                    rootRot = Rotation.from_matrix(wristR[t, side].T @ vd.xmat[bid].reshape(3, 3)).as_rotvec()
                    block(
                        np.r_[15 * rootErr, 3 * rootRot],
                        np.vstack([15 * rootJp, 3 * _log_jacobian(rootRot) @ wristR[t, side].T @ rootJr]),
                        t,
                    )
                    if not masks[t, side]:
                        continue
                    width, Jwidth = opening_and_jac(vm, vd, ["left", "right"][side], va)
                    if finger_targets is not None:
                        target = np.asarray(finger_targets[t, side])
                        opening = target[0] - 0.7 * target[1] - 0.3 * target[2]
                        if np.isfinite(opening).all():
                            block([100 * (width - np.linalg.norm(opening))], 100 * Jwidth, t)
                    if width < 0.003:
                        block([10000 * (width - 0.003)], 10000 * Jwidth, t)
                    guard(-Jwidth, t, [width - 0.001])
                    cp, cR, Jp, cJr = task_pose_and_jac(vm, vd, ["left", "right"][side], va)
                    err = cp - P[t, side]
                    rot = Rotation.from_matrix(R[t, side].T @ cR).as_rotvec()
                    Jr = _log_jacobian(rot) @ R[t, side].T @ cJr
                    block(np.r_[300 * err, 30 * rot], np.vstack([300 * Jp, 30 * Jr]), t)
                    norm = np.linalg.norm(err)
                    if norm > 0.0035:
                        block([8000 * (norm - 0.0035)], 8000 * (err / norm) @ Jp, t)
                    if norm > 0.00475:
                        block([30000 * (norm - 0.00475)], 30000 * (err / norm) @ Jp, t)
                    angle = np.linalg.norm(rot)
                    if angle > 0.10:
                        block([500 * (angle - 0.10)], 500 * (rot / angle) @ Jr, t)
                    if angle > 0.14:
                        block([10000 * (angle - 0.14)], 10000 * (rot / angle) @ Jr, t)
                    if norm > 1e-12:
                        guard((err / norm) @ Jp, t, [_acceptance_gate(0.00495, "position_m", -0.00005) - norm])
                    if angle > 1e-12:
                        guard((rot / angle) @ Jr, t, [_acceptance_gate(0.149, "rotation_rad", -0.001) - angle])
                    violations.extend(
                        [
                            max(norm - _acceptance_gate(0.00495, "position_m", -0.00005), 0)
                            / _acceptance_gate(0.005, "position_m"),
                            max(angle - _acceptance_gate(0.149, "rotation_rad", -0.001), 0)
                            / _acceptance_gate(0.15, "rotation_rad"),
                        ]
                    )
                if optimize_fingers and finger_targets is not None:
                    for tip, target in zip(tips, np.asarray(finger_targets[t]).reshape(10, 3)):
                        if not np.isfinite(target).all():
                            continue
                        mujoco.mj_jacSite(vm, vd, jp, None, tip)
                        block(100 * (vd.site_xpos[tip] - target), 100 * jp[:, va], t)
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
                if c.dist < 0:
                    block([12000 * (-float(c.dist))], -12000 * normal @ (jp2 - jp)[:, va], t, half)
                guard(-normal @ (jp2 - jp)[:, va], t, [float(c.dist) - 0.001], half)
                violations.append(max(0.001 - float(c.dist), 0) / 0.003)
        J = sparse.csr_matrix((values, (rowidx, colidx)), shape=(count, N))
        r = np.asarray(residual)
        prior = np.r_[np.tile(joint_prior, n), np.tile([0.3, 0.3, 0.3, 0.05, 0.05, 0.05], 2)]
        J = sparse.vstack([J, 0.12 * D, weighted_A, sparse.diags(prior)], format="csr")
        latest_guard = (
            sparse.csr_matrix((gvalues, (grows, gcols)), shape=(len(gu), N)),
            np.full(len(gu), -np.inf),
            np.asarray(gu),
        )
        latest_violation = float(np.dot(violations, violations))
        return np.r_[r, 0.12 * (D @ v), weighted_A @ v, prior * (v - np.r_[track[:, control].ravel(), np.zeros(12)])], J

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
        palmErr, palmRot = pose_errors(m, qq, wristP, wristR)
        cp, cR = task_track(m, qq)
        allErr = np.linalg.norm(cp - P, axis=-1)
        allRot = Rotation.from_matrix((R.transpose(0, 1, 3, 2) @ cR).reshape(-1, 3, 3)).magnitude().reshape(n, 2)
        err = allErr[masks]
        rot = allRot[masks]
        contacts = arm_contacts(m, qq)
        connection = audit_connection(m, qq)
        speed = float(np.max(abs(D @ v) / np.repeat(np.diff(times), jcount)))
        am = angular_metrics(cR, R, masks)
        angular_gate = angular_eligible(am, angular_baseline) if angular_baseline is not None else None
        passed = bool(
            err.max() <= _acceptance_gate(0.005, "position_m")
            and rot.max() <= _acceptance_gate(0.15, "rotation_rad")
            and speed <= 3 + 1e-5
            and contacts["arm_collision_samples"] == 0
            and angular_gate is not False
        )
        record = dict(
            label=label,
            passed=passed,
            eef_position_max_m=float(err.max()),
            eef_rotation_max_rad=float(rot.max()),
            task_tcp_position_max_m=float(err.max()),
            task_tcp_rotation_max_rad=float(rot.max()),
            goal_profile="frozen_orientation_eq12_articulated_sharpa_tcp",
            original_orientation_angular_gate_passed=angular_gate,
            angular_metrics=am,
            kernel_source_sha256=kernel_source_sha256,
            palm_reference_position_max_m=float(palmErr.max()),
            palm_reference_rotation_max_rad=float(palmRot.max()),
            palm_reference_5mm_passed=bool(palmErr.max() <= 0.005),
            arm_speed_max_rad_s=speed,
            all_controlled_speed_max_rad_s=speed,
            optimized_joint_count=jcount,
            arm_collision_samples=contacts["arm_collision_samples"],
            seconds=time.monotonic() - started,
        )
        rank = (
            not passed,
            record["arm_collision_samples"],
            max(err.max() / _acceptance_gate(0.005, "position_m"), rot.max() / _acceptance_gate(0.15, "rotation_rad")),
        )
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
                    palm_reference_positions=wristP,
                    palm_reference_rotations=wristR,
                    target_masks=masks,
                ),
            )
        np.savez_compressed(output / (label + ".npz"), x=v, qpos=qq)
        print("COUPLED_PHYSICAL", record, flush=True)
        return passed

    if physical(x, "initial"):
        return best[1]
    for phase, mid in enumerate([False, True], 1):
        radius = 0.15
        damping = 10.0
        for iteration in range(64):
            r, J = evaluate(x, mid)
            cost = float(r @ r) / 2
            violation = latest_violation
            caps = np.r_[
                np.full(n * jcount, radius), np.tile([0.02, 0.02, 0.02, 0.06, 0.06, 0.06], 2) * (radius / 0.15)
            ]
            step, info = bounded_step(
                J,
                r,
                x,
                lo,
                hi,
                D,
                np.repeat(3 * np.diff(times) - 1e-6, jcount),
                caps,
                damping=damping,
                linear_guard=latest_guard,
            )
            if step is None:
                failed = info
                step, info = bounded_step(
                    J, r, x, lo, hi, D, np.repeat(3 * np.diff(times) - 1e-6, jcount), caps, damping=damping
                )
                info["hard_guard_attempt"] = failed
            row = dict(phase=phase, iteration=iteration, cost=cost, qp=info, violation_before=violation)
            if step is None:
                row["failed_step"] = True
                history.append(row)
                radius *= 0.5
                damping *= 10
            else:
                ok = False
                for f in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.015625]:
                    trial = x + f * step
                    try:
                        rr, _ = evaluate(trial, mid)
                    except ValueError as error:
                        if "Degenerate zero-width articulated TCP" not in str(error):
                            raise
                        row.setdefault("rejected_degenerate_trials", []).append(f)
                        continue
                    new = float(rr @ rr) / 2
                    feasible_improvement = violation > 1e-10 and latest_violation < violation - max(
                        1e-12, violation * 1e-4
                    )
                    protected_descent = violation <= 1e-10 and latest_violation <= 1e-10 and new < cost - 1e-8
                    if feasible_improvement or protected_descent:
                        x = trial
                        row.update(cost_after=new, fraction=f, violation_after=latest_violation)
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
