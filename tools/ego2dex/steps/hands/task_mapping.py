# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
from .acceptance import gate as _acceptance_gate
import json
import os
from pathlib import Path
import mujoco
import numpy as np
from scipy.signal import savgol_filter
from steps.sharpa.task_tcp import pose_and_jac
from .registry import profile, project_seed, selected
from .panda_scene import build, indices
from .initialize import initialize
from .acceptance import current, reusable_record


def finish(result, ref, manifest, points, masks, output, stage):
    from .projection_quality import verify

    if stage:
        stage("all_frame_original_camera_pixel_check")
    pixel = verify(
        result["spec"],
        result["qpos"],
        ref["obs_head_pose"],
        ref["intrinsics"],
        (manifest["height"], manifest["width"]),
        points,
        masks,
        output / "projection_verification.json",
    )
    if not pixel["passed"]:
        raise RuntimeError("Kinematics passed; actual raster view check failed")
    report = {
        s: dict(
            physical_finger_dofs=profile(s)["physical_joint_count"],
            independent_motor_dofs=len(profile(s)["motor_names"]),
            original_finger_mapping=profile(s)["digits"],
            coupling_enforced_in_solver=True,
            hand_contact_quality_passed=False,
        )
        for s in ["left", "right"]
    }
    return result, report


def solve_fixed_tool(model, rows, times, points, valid, seed_path, reference_path, output, stage=None):
    acceptance = current()
    from .coupled_place import remap_arms as place
    from .coupled_polish import remap_arms as polish

    if os.environ.get("EGO2ROBOT_HAND_SOLVE") == "feasibility":
        from .coupled_place_feasible import remap_arms as place
        from .coupled_polish_feasible import remap_arms as polish
    elastic = os.environ.get("EGO2ROBOT_HAND_SOLVE") == "elastic"
    from .v28_recovery import policy, recover as recover_v28

    v28_policy = policy()
    bounded_regression = os.environ.get("EGO2ROBOT_REGRESSION_PASS") == "1"
    if elastic:
        from .elastic_mapping import remap_arms as place

        def polish(*args, **kwargs):
            return place(*args, **dict(kwargs, fixed_bases=True, max_iterations=25 if bounded_regression else 35))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "v28_recovery.json").write_text(
        json.dumps(dict(policy=v28_policy, attempted=False, reason="not_needed", selected="baseline"), indent=2)
    )
    (output / "v29_recovery.json").write_text(
        json.dumps(
            dict(
                policy=v28_policy,
                attempted=False,
                reason="disabled" if v28_policy != "v28-v29" else "not_needed",
                selected="baseline",
            ),
            indent=2,
        )
    )
    seed = np.load(seed_path)
    P, R, masks = seed["p"], seed["R"], seed["masks"].astype(bool)
    n = len(rows)
    before = json.loads(
        Path(seed_path).with_name(Path(seed_path).name.replace("_joint_pose_seed.npz", "_joint_pose.json")).read_text()
    )["before"]
    ref = np.load(reference_path)
    manifest = json.loads(Path(reference_path).with_name("manifest.json").read_text())
    case = Path(reference_path).parent.name
    cached = None
    repair_view = os.environ.get("EGO2ROBOT_REPAIR_VIEW") == "1"
    if all((output / f).exists() for f in ["solved_scene.xml", "solved_track.npz", "task_tcp_mapping_result.json"]):
        cached_spec = mujoco.MjSpec.from_file(str(output / "solved_scene.xml"))
        cached_model = cached_spec.compile()
        cached_q = np.load(output / "solved_track.npz")["qpos"]
        record = json.loads((output / "task_tcp_mapping_result.json").read_text())
        prior_goals = output / "task_tcp_goals.npz"
        same_goals = False
        if prior_goals.exists():
            old_goals = np.load(prior_goals)
            same_goals = (
                np.array_equal(old_goals["positions"], P)
                and np.array_equal(old_goals["rotations"], R)
                and np.array_equal(old_goals["masks"].astype(bool), masks)
            )
        if record["passed"] and not reusable_record(record):
            record = dict(
                record, passed=False, reason="Previous profile is looser than the active profile; re-solve required"
            )
        if record["passed"] and not same_goals:
            record = dict(
                record,
                passed=False,
                previous_goal_pass_not_reused=True,
                reason="Current task goals differ or have no recorded cache fingerprint",
            )
        from .mount_contract import audit_connection

        audit_connection(cached_model, cached_q)
        data = mujoco.MjData(cached_model)
        data.qpos[:] = cached_q[0]
        mujoco.mj_forward(cached_model, data)
        cached = dict(
            spec=cached_spec,
            model=cached_model,
            qpos=cached_q,
            record=record,
            bases={
                s: dict(
                    pos=data.xpos[cached_model.body(s + "_link0").id].copy(),
                    quat=data.xquat[cached_model.body(s + "_link0").id].copy(),
                )
                for s in ["left", "right"]
            },
        )
        if record["passed"] and not repair_view:
            if stage:
                stage("reuse_saved_physical_solution")
            return finish(cached, ref, manifest, points, masks, output, stage)
        from .conic_finish import refine as conic_refine

        corrected = (
            conic_refine(
                cached,
                times,
                P,
                R,
                masks,
                ref["obs_head_pose"],
                ref["intrinsics"],
                (manifest["height"], manifest["width"]),
                before,
                output / "cached_conic_finish",
            )
            if not repair_view and os.environ.get("EGO2ROBOT_TASK_FRAME_SOLVE") != "geometric"
            else cached
        )
        if corrected["record"]["passed"] and not repair_view:
            from .panda_scene import standalone
            import xml.etree.ElementTree as ET

            (output / "solved_scene.xml").write_text(ET.tostring(standalone(corrected["spec"]), encoding="unicode"))
            np.savez_compressed(output / "solved_track.npz", qpos=corrected["qpos"], timestamp=times)
            (output / "task_tcp_mapping_result.json").write_text(json.dumps(corrected["record"], indent=2))
            return finish(corrected, ref, manifest, points, masks, output, stage)
        cached = corrected
    # Optional arm seed (EGO2ROBOT_ARM_INITIALIZER): a directory, or a directory of
    # per-case directories, holding a passed independent_validation.json and either
    # arm_seed.npz (static base poses + arm joints) or scene.xml + trajectory.npz,
    # e.g. the accepted Sharpa result for the same episode. Only static bases and
    # arm joints are reused; hands and all task, contact and view gates are re-solved.
    # Without a seed the solve starts from the initial pose.
    arm_initializer = os.environ.get("EGO2ROBOT_ARM_INITIALIZER", "")
    if arm_initializer:
        folder = Path(arm_initializer)
        if (folder / case).is_dir():
            folder = folder / case
        assert (folder / "independent_validation.json").exists() and json.loads(
            (folder / "independent_validation.json").read_text()
        )["passed"], folder
        if (folder / "arm_seed.npz").exists():
            z = np.load(folder / "arm_seed.npz")
            bases = {s: dict(pos=z[s + "_base_pos"], quat=z[s + "_base_quat"]) for s in ["left", "right"]}
            arms = {s: z[s + "_arm_qpos"] for s in ["left", "right"]}
        else:
            sm = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
            sq = np.load(folder / "trajectory.npz")["qpos"]
            sd = mujoco.MjData(sm)
            sd.qpos[:] = sq[0]
            mujoco.mj_forward(sm, sd)
            bases = {
                s: dict(pos=sd.xpos[sm.body(s + "_link0").id].copy(), quat=sd.xquat[sm.body(s + "_link0").id].copy())
                for s in ["left", "right"]
            }
            arms = {
                s: np.stack([sq[:, int(sm.joint(f"{s}_joint{j}").qposadr[0])] for j in range(1, 8)], axis=1)
                for s in ["left", "right"]
            }
        spec = build(base_poses=bases, include_floor=False)
        model = spec.compile()
        rows = np.tile(model.qpos0, (n, 1))
        for s in ["left", "right"]:
            assert arms[s].shape == (n, 7), (s, arms[s].shape, n)
            for j in range(1, 8):
                rows[:, int(model.joint(f"{s}_joint{j}").qposadr[0])] = arms[s][:, j - 1]
        (output / "numerical_initializer.json").write_text(
            json.dumps(
                dict(pipeline="Orientation17", source=str(folder), only_arms_and_static_bases_reused=True), indent=2
            )
        )
    if cached is not None:
        model = cached["model"]
        rows = cached["qpos"]
    strategy = os.environ.get("EGO2ROBOT_HAND_SEED", "source")
    if cached is None and strategy == "rigid_track":
        from scipy.spatial.transform import Rotation
        from steps.sharpa.task_tcp import track

        rows = project_seed(model, rows, times)
        flags = int(model.opt.disableflags)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        actualP, actualR = track(model, rows)
        data = mujoco.MjData(model)
        data.qpos[:] = rows[0]
        mujoco.mj_forward(model, data)
        new_bases = {}
        fit = []
        for side, s in enumerate(["left", "right"]):
            active = masks[:, side]
            relative = R[active, side] @ actualR[active, side].transpose(0, 2, 1)
            delta = Rotation.from_matrix(relative).mean().as_matrix() if active.any() else np.eye(3)
            shift = (
                np.median(P[active, side] - actualP[active, side] @ delta.T, axis=0) if active.any() else np.zeros(3)
            )
            base = model.body(s + "_link0").id
            baseR = data.xmat[base].reshape(3, 3)
            new_bases[s] = dict(
                pos=(delta @ data.xpos[base] + shift).tolist(),
                quat=Rotation.from_matrix(delta @ baseR).as_quat()[[3, 0, 1, 2]].tolist(),
            )
            fit.append(dict(side=s, rigid_rotation=delta.tolist(), rigid_translation=shift.tolist()))
        model.opt.disableflags = flags
        old_names = [model.joint(i).name for i in range(model.njnt)]
        model = build(base_poses=new_bases, include_floor=False).compile()
        assert old_names == [model.joint(i).name for i in range(model.njnt)]
        (output / "rigid_track_initializer.json").write_text(
            json.dumps(
                dict(
                    method="rigid world alignment of the accepted Orientation arm trajectory for the new tool",
                    static_bases=new_bases,
                    fit=fit,
                    mount_transform_changed=False,
                    original_targets_changed=False,
                ),
                indent=2,
            )
        )
    elif cached is None and strategy not in ["source", "palm_out", "palm_in"]:
        from scipy.spatial.transform import Rotation

        sign = 1 if strategy == "elbow_out" else -1
        neutral = {s: dict(pos=[0, 0, 0], quat=[1, 0, 0, 0]) for s in ["left", "right"]}
        nm = build(base_poses=neutral, include_floor=False).compile()
        seed_rows = np.tile(nm.qpos0, (n, 1))
        for side, s in enumerate(["left", "right"]):
            seed_rows[:, indices(nm, s)["arm_qadr"]] = [0, -0.4, sign * (-1.2 if s == "left" else 1.2), -2, 0, 1.6, 0.8]
        seed_rows = project_seed(nm, seed_rows, times)
        nd = mujoco.MjData(nm)
        nd.qpos[:] = seed_rows[n // 2]
        nm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_forward(nm, nd)
        new_bases = {}
        for side, s in enumerate(["left", "right"]):
            cp, cR, _, _ = pose_and_jac(nm, nd, s, [])
            delta = R[n // 2, side] @ cR.T
            new_bases[s] = dict(
                pos=(P[n // 2, side] - delta @ cp).tolist(),
                quat=Rotation.from_matrix(delta).as_quat()[[3, 0, 1, 2]].tolist(),
            )
        model = build(base_poses=new_bases, include_floor=False).compile()
        rows = seed_rows
        (output / "alternative_branch_seed.json").write_text(
            json.dumps(
                dict(strategy=strategy, static_bases=new_bases, mount_unchanged=True, anchor_frame=n // 2), indent=2
            )
        )
    rows = project_seed(model, rows, times)
    tips = np.stack([points[s][:, [4, 8, 12, 16, 20]].copy() for s in ["left", "right"]], axis=1)
    window = min(11, n if n % 2 else n - 1)
    if window >= 5:
        tips = savgol_filter(tips, window, 3, axis=0)
    wristP = np.stack([points[s][:, 0] for s in ["left", "right"]], axis=1)
    wristR = np.empty_like(R)
    d = mujoco.MjData(model)
    d.qpos[:] = rows[0]
    flags = int(model.opt.disableflags)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    mujoco.mj_forward(model, d)
    for side, s in enumerate(["left", "right"]):
        _, taskR, _, _ = pose_and_jac(model, d, s, [])
        palmR = d.xmat[model.body(f"{s}_hand_{s}_hand_C_MC").id].reshape(3, 3)
        wristR[:, side] = R[:, side] @ (taskR.T @ palmR)
        tips[~(valid[s] & masks[:, side]), side] = np.nan
        if len(profile(s)["digits"]) == 4:
            tips[:, side, 4] = np.nan
    model.opt.disableflags = flags
    if elastic:
        from scipy.spatial.transform import Rotation, Slerp
        from .panda_scene import mounting

        for side, s in enumerate(["left", "right"]):
            observed = np.flatnonzero(valid[s] & masks[:, side])
            if not len(observed):
                continue
            key = points[s][observed]
            z = key[:, 9] - key[:, 0]
            z /= np.linalg.norm(z, axis=1)[:, None]
            x = key[:, 5] - key[:, 13]
            x -= z * np.sum(x * z, axis=1)[:, None]
            x /= np.linalg.norm(x, axis=1)[:, None]
            y = np.cross(z, x)
            anatomy = np.stack([x, y, z], axis=2)
            palm_to_anatomy = np.asarray(mounting(selected(), s)["palm_anatomy_rotation"])
            desired = anatomy @ palm_to_anatomy.T
            wristR[:, side] = (
                Slerp(times[observed], Rotation.from_matrix(desired))(
                    np.clip(times, times[observed[0]], times[observed[-1]])
                ).as_matrix()
                if len(observed) > 1
                else desired[0]
            )
            for j in range(3):
                wristP[:, side, j] = np.interp(times, times[observed], points[s][observed, 0, j])
        if window >= 5:
            wristP = savgol_filter(wristP, window, 3, axis=0)
        (output / "palm_reference_definition.json").write_text(
            json.dumps(
                dict(
                    method="human MCP anatomy and vendor palm anatomy",
                    soft_objective_only=True,
                    original_task_TCP_unchanged=True,
                ),
                indent=2,
            )
        )
    # Retarget human finger proportions to the actual vendor skeleton. This
    # scales soft keypoint TARGETS only: no robot geometry, camera or original
    # task TCP is scaled. Recenter the resulting morphology around original P.
    morphology = []
    for side, s in enumerate(["left", "right"]):
        p = profile(s)
        palm = model.body(f"{s}_hand_{p['palm']}").id
        vendor = []
        human = []
        for k, tip in enumerate(p["tips"]):
            chain = [0, *range(1 + 4 * k, 5 + 4 * k)]
            length = np.linalg.norm(np.diff(points[s][:, chain], axis=1), axis=-1).sum(axis=1)
            observed = valid[s] & masks[:, side]
            human.append(float(np.median(length[observed])) if observed.any() else 0.0)
            body = model.body(s + "_hand_" + tip).id
            distance = 0.0
            while body != palm and body > 0:
                parent = int(model.body_parentid[body])
                distance += float(np.linalg.norm(d.xpos[body] - d.xpos[parent]))
                body = parent
            assert body == palm, (s, tip, p["palm"])
            vendor.append(distance)
        scale = float(np.mean(vendor) / max(np.mean(human), 1e-8))
        raw_wrist = wristP[:, side].copy()
        tips[:, side] = raw_wrist[:, None] + scale * (tips[:, side] - raw_wrist[:, None])
        center = 0.5 * (tips[:, side, 0] + 0.7 * tips[:, side, 1] + 0.3 * tips[:, side, 2])
        shift = P[:, side] - center
        # Invalid soft observations remain NaN and have no objective weight.
        shift[~np.isfinite(shift).all(axis=1)] = 0.0
        tips[:, side] += shift[:, None]
        wristP[:, side] = raw_wrist + shift
        morphology.append(
            dict(
                side=s,
                uniform_soft_target_scale=scale,
                vendor_chain_lengths_m=vendor,
                human_median_chain_lengths_m=human,
                geometry_scaled=False,
                original_task_goals_unchanged=True,
                source_camera_unchanged=True,
            )
        )
    (output / "finger_target_normalization.json").write_text(json.dumps(morphology, indent=2))
    if cached is None and strategy in ["palm_out", "palm_in"]:
        from scipy.spatial.transform import Rotation

        neutral = {s: dict(pos=[0, 0, 0], quat=[1, 0, 0, 0]) for s in ["left", "right"]}
        nm = build(base_poses=neutral, include_floor=False).compile()
        home = np.tile(nm.qpos0, (n, 1))
        sign = 1 if strategy == "palm_out" else -1
        for side, s in enumerate(["left", "right"]):
            home[:, indices(nm, s)["arm_qadr"]] = [0, -0.4, sign * (-1.2 if s == "left" else 1.2), -2, 0, 1.6, 0.8]
        home = project_seed(nm, home, times)
        nd = mujoco.MjData(nm)
        nd.qpos[:] = home[0]
        nm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_forward(nm, nd)
        new_bases = {}
        for side, s in enumerate(["left", "right"]):
            active = np.flatnonzero(masks[:, side])
            anchor = int(active[len(active) // 2]) if len(active) else n // 2
            bid = nm.body(f"{s}_hand_{s}_hand_C_MC").id
            delta = wristR[anchor, side] @ nd.xmat[bid].reshape(3, 3).T
            new_bases[s] = dict(
                pos=(wristP[anchor, side] - delta @ nd.xpos[bid]).tolist(),
                quat=Rotation.from_matrix(delta).as_quat()[[3, 0, 1, 2]].tolist(),
            )
        model = build(base_poses=new_bases, include_floor=False).compile()
        rows = home
        (output / "anatomical_palm_base_initializer.json").write_text(
            json.dumps(
                dict(strategy=strategy, bases=new_bases, mount_unchanged=True, source_task_targets_unchanged=True),
                indent=2,
            )
        )
    sample = np.unique(np.r_[np.arange(0, n, 3), n - 1]).astype(int)
    adaptive = os.environ.get("EGO2ROBOT_ADAPTIVE_SAMPLE") == "1"
    if adaptive:
        from scipy.spatial.transform import Rotation

        turn = (
            Rotation.from_matrix((R[:-1].transpose(0, 1, 3, 2) @ R[1:]).reshape(-1, 3, 3)).magnitude().reshape(n - 1, 2)
        )
        rapid = np.flatnonzero((turn > 0.20).any(axis=1))
        extra = np.concatenate([rapid + k for k in range(-2, 4)]) if len(rapid) else np.array([], int)
        sample = np.unique(np.r_[sample, extra[(extra >= 0) & (extra < n)]]).astype(int)
        (output / "adaptive_initialization_frames.json").write_text(
            json.dumps(
                dict(frames=sample.tolist(), rapid_source_turn_frames=rapid.tolist(), all_frames_still_validated=True),
                indent=2,
            )
        )
    if stage:
        stage("independent_motor_coordinates_task_initialization")
    if os.environ.get("EGO2ROBOT_FRAMEWISE_INITIALIZER") == "0":
        q = rows[sample]
    else:
        q = (
            rows[sample]
            if cached is not None
            else initialize(
                model,
                rows[sample],
                P[sample],
                R[sample],
                masks[sample],
                tips[sample],
                wristP[sample],
                wristR[sample],
                times[sample],
                output / "motor_initialization",
            )
        )
    if stage:
        stage("camera_aware_static_base_and_joint_optimization")
    common = dict(
        initialization="existing",
        optimize_fingers=True,
        angular_baseline=before,
        camera_K=ref["intrinsics"],
        image_shape=(manifest["height"], manifest["width"]),
    )
    workspace = np.stack([points[s][:, [4, 8, 12]] for s in ["left", "right"]], axis=1) if elastic else None
    result = place(
        model,
        q,
        times[sample],
        output / "placement",
        target_positions=P[sample],
        target_rotations=R[sample],
        finger_targets=tips[sample],
        palm_reference_positions=wristP[sample],
        palm_reference_rotations=wristR[sample],
        target_masks=masks[sample],
        head_poses=ref["obs_head_pose"][sample],
        **dict(common, **({"hand_workspace_points": workspace[sample]} if elastic else {})),
    )
    if (
        elastic
        and not result["record"]["passed"]
        and result["record"]["eef_position_max_m"] <= 1.5 * acceptance.position_m
        and result["record"]["eef_rotation_max_rad"] <= 1.2 * acceptance.rotation_rad
        and result["record"]["view_layout"]["proximal_layout_max_violation_m"]
        <= _acceptance_gate(0.025, "layout_m", scale=0.025 / 0.014)
    ):
        from .conic_finish import refine as coarse_finish

        result = coarse_finish(
            result,
            times[sample],
            P[sample],
            R[sample],
            masks[sample],
            ref["obs_head_pose"][sample],
            ref["intrinsics"],
            (manifest["height"], manifest["width"]),
            before,
            output / "coarse_conic_finish",
        )
    if elastic and bounded_regression and not result["record"]["passed"]:
        import xml.etree.ElementTree as ET
        from .panda_scene import standalone

        (output / "coarse_rejection.json").write_text(
            json.dumps(
                dict(
                    record=result["record"], checked_frame_indices=sample.tolist(), full_frame_validation_complete=False
                ),
                indent=2,
            )
        )
        (output / "coarse_scene.xml").write_text(ET.tostring(standalone(result["spec"]), encoding="unicode"))
        np.savez_compressed(
            output / "coarse_track.npz", qpos=result["qpos"], timestamp=times[sample], source_frame_indices=sample
        )
        raise RuntimeError("Coarse physical candidate failed unchanged acceptance; retained for a separate repair pass")
    q = np.stack([np.interp(times, times[sample], result["qpos"][:, j]) for j in range(model.nq)], axis=1)
    if stage:
        stage("all_original_frames_temporal_polish")
    result = polish(
        result["model"],
        q,
        times,
        output / "all_frame_polish",
        target_positions=P,
        target_rotations=R,
        finger_targets=tips,
        palm_reference_positions=wristP,
        palm_reference_rotations=wristR,
        target_masks=masks,
        head_poses=ref["obs_head_pose"],
        **dict(common, **({"hand_workspace_points": workspace} if elastic else {})),
    )
    from .conic_finish import refine as finish_static_base

    if not result["record"]["passed"]:
        result = finish_static_base(
            result,
            times,
            P,
            R,
            masks,
            ref["obs_head_pose"],
            ref["intrinsics"],
            (manifest["height"], manifest["width"]),
            before,
            output / "conic_finish",
        )
    if elastic and not result["record"]["passed"] and v28_policy in ("v28", "v28-v29"):
        if stage:
            stage("v28_arm_and_static_base_recovery")
        result = recover_v28(
            result,
            times,
            P,
            R,
            masks,
            ref,
            manifest,
            points,
            tips,
            wristP,
            wristR,
            workspace,
            before,
            output,
            place,
            polish,
        )
    if elastic and not result["record"]["passed"] and v28_policy == "v28-v29":
        from .v29_recovery import recover as recover_v29

        if stage:
            stage("v29_collision_aware_initialization_recovery")
        result = recover_v29(
            result,
            times,
            P,
            R,
            masks,
            ref,
            manifest,
            points,
            tips,
            wristP,
            wristR,
            workspace,
            before,
            output,
            place,
            polish,
        )
    # A different hand/forearm can require a second static-base placement.
    # Continue from its best physical candidate; no completed video is rerun.
    for attempt in range(0 if bounded_regression else 2):
        if result["record"]["passed"]:
            break
        if stage:
            stage(f"continue_base_placement_{attempt + 1}")
        retry = output / f"continue_{attempt + 1}"
        retry.mkdir(exist_ok=True)
        result = place(
            result["model"],
            result["qpos"][sample],
            times[sample],
            retry / "placement",
            target_positions=P[sample],
            target_rotations=R[sample],
            finger_targets=tips[sample],
            palm_reference_positions=wristP[sample],
            palm_reference_rotations=wristR[sample],
            target_masks=masks[sample],
            head_poses=ref["obs_head_pose"][sample],
            **common,
        )
        q = np.stack([np.interp(times, times[sample], result["qpos"][:, j]) for j in range(model.nq)], axis=1)
        result = polish(
            result["model"],
            q,
            times,
            retry / "all_frame_polish",
            target_positions=P,
            target_rotations=R,
            finger_targets=tips,
            palm_reference_positions=wristP,
            palm_reference_rotations=wristR,
            target_masks=masks,
            head_poses=ref["obs_head_pose"],
            **common,
        )
        if not result["record"]["passed"]:
            result = finish_static_base(
                result,
                times,
                P,
                R,
                masks,
                ref["obs_head_pose"],
                ref["intrinsics"],
                (manifest["height"], manifest["width"]),
                before,
                retry / "conic_finish",
            )
    from .panda_scene import standalone
    import xml.etree.ElementTree as ET

    (output / "solved_scene.xml").write_text(ET.tostring(standalone(result["spec"]), encoding="unicode"))
    np.savez_compressed(output / "solved_track.npz", qpos=result["qpos"], timestamp=times)
    (output / "task_tcp_mapping_result.json").write_text(json.dumps(result["record"], indent=2))
    np.savez_compressed(output / "task_tcp_goals.npz", positions=P, rotations=R, masks=masks)
    if not result["record"]["passed"]:
        raise RuntimeError("Registered hand task/kinematic/view gate failed; rejected trajectory retained")
    return finish(result, ref, manifest, points, masks, output, stage)
