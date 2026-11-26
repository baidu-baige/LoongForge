# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Formal rigid-tool mapping against the frozen Orientation task definition."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def solve_fixed_tool(model, rows, times, points, valid, seed_path, reference_path, output, stage=None):
    from .hand_swap import _fingers, sha
    from .rigid_retarget import palm_track
    from .coupled_mapping_v6 import remap_arms as initialize_palm
    from .coupled_mapping_v17 import remap_arms
    from .panda_scene import indices
    from steps.retarget.targets import smooth_retarget_targets
    from retargeting.pipeline.process_dataset import compute_wrist_rotation

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from .recovery import new_session, recover, _protected_copy

    recovery_session = new_session(output)
    seed_path = Path(seed_path)
    seed = np.load(seed_path)
    P, R, masks = seed["p"], seed["R"], seed["masks"].astype(bool)
    assert P.shape == (len(rows), 2, 3) and R.shape == (len(rows), 2, 3, 3)
    before = json.loads(
        seed_path.with_name(seed_path.name.replace("_joint_pose_seed.npz", "_joint_pose.json")).read_text()
    )["before"]
    reference = np.load(reference_path)
    manifest = json.loads(Path(reference_path).with_name("manifest.json").read_text())
    wristP, wristR = palm_track(model, rows)
    observed = {}
    for side, s in enumerate(["left", "right"]):
        observed[s] = valid[s] & masks[:, side]
        good = np.flatnonzero(valid[s])
        if len(good):
            for dim in range(3):
                wristP[:, side, dim] = np.interp(np.arange(len(rows)), good, points[s][good, 0, dim])
            raw = compute_wrist_rotation(points[s][good], is_right=s == "right").as_matrix()
            wristR[:, side] = (
                Slerp(good, Rotation.from_matrix(raw))(np.clip(np.arange(len(rows)), good[0], good[-1])).as_matrix()
                if len(good) > 1
                else raw[0]
            )
        wristP[:, side], wristR[:, side], _ = smooth_retarget_targets(
            wristP[:, side], wristR[:, side], np.zeros(len(rows)), window=11, polyorder=3, orientation_sigma=2.0
        )
    if stage:
        stage("anatomical_palm_initialization_only")
    # This is only an initializer. Its wrist-proxy result never certifies the
    # final Sharpa trajectory; the original articulated task TCP does below.
    initial = initialize_palm(
        model, rows, times, output / "palm_initialization", target_positions=wristP, target_rotations=wristR
    )
    if stage:
        stage("finger_initialization_under_fixed_mount")
    initial_dir = output / "finger_initialization"
    initial_dir.mkdir(exist_ok=True)
    q, initial_report = _fingers(initial["model"], initial["qpos"], points, observed, times, initial_dir)
    np.savez_compressed(initial_dir / "trajectory.npz", qpos=q, timestamp=times)
    (initial_dir / "report.json").write_text(json.dumps(initial_report, indent=2))
    tips = np.stack([points[s][:, [4, 8, 12, 16, 20]].copy() for s in ["left", "right"]], axis=1)
    for side, s in enumerate(["left", "right"]):
        tips[~observed[s], side] = np.nan
    if stage:
        stage("joint58_original_orientation_task_tcp")
    if len(q) > 500:
        from .task_precondition import initialize as initialize_task
        from .coupled_mapping_v22 import remap_arms as remap_projected

        if stage:
            stage("long_track_coarse_initialization")
        q = initialize_task(
            initial["model"], q, P, R, masks, tips, wristP, wristR, output / "task_frame_initialization"
        )
        sample = np.unique(np.r_[np.arange(0, len(q), 4), len(q) - 1]).astype(int)
        coarse = remap_projected(
            initial["model"],
            q[sample],
            times[sample],
            output / "task_tcp_coarse_initialization",
            target_positions=P[sample],
            target_rotations=R[sample],
            initialization="existing",
            optimize_fingers=True,
            finger_targets=tips[sample],
            palm_reference_positions=wristP[sample],
            palm_reference_rotations=wristR[sample],
            target_masks=masks[sample],
            angular_baseline=before,
        )
        q = np.stack([np.interp(times, times[sample], coarse["qpos"][:, j]) for j in range(58)], axis=1)
        # Downsampling is confined to initialization. Final checks always
        # include every original frame, plus every adjacent midpoint.
        if stage:
            stage("joint58_all_original_frames")
        coupled = remap_projected(
            coarse["model"],
            q,
            times,
            output / "task_tcp_mapping",
            target_positions=P,
            target_rotations=R,
            initialization="existing",
            optimize_fingers=True,
            finger_targets=tips,
            palm_reference_positions=wristP,
            palm_reference_rotations=wristR,
            target_masks=masks,
            angular_baseline=before,
        )
    else:
        coupled = remap_arms(
            initial["model"],
            q,
            times,
            output / "task_tcp_mapping",
            target_positions=P,
            target_rotations=R,
            initialization="existing",
            optimize_fingers=True,
            finger_targets=tips,
            palm_reference_positions=wristP,
            palm_reference_rotations=wristR,
            target_masks=masks,
            angular_baseline=before,
        )
    if not coupled["record"]["passed"]:
        (output / "task_tcp_primary_rejection.json").write_text(json.dumps(coupled["record"], indent=2))
        from .task_precondition import initialize as initialize_task
        from .coupled_mapping_v22 import remap_arms as remap_projected

        if stage:
            stage("task_tcp_initialization_and_speed_feasibility_retry")
        q = initialize_task(
            coupled["model"], coupled["qpos"], P, R, masks, tips, wristP, wristR, output / "task_frame_initialization"
        )
        coupled = remap_projected(
            coupled["model"],
            q,
            times,
            output / "task_tcp_projected_retry",
            target_positions=P,
            target_rotations=R,
            initialization="existing",
            optimize_fingers=True,
            finger_targets=tips,
            palm_reference_positions=wristP,
            palm_reference_rotations=wristR,
            target_masks=masks,
            angular_baseline=before,
        )
    # The real registered-hand global initializers run on the Sharpa model
    # backend; their own outputs are never acceptance certificates. Both the
    # TCP-rejection and later raster-rejection hooks share this task session,
    # so each actual initializer can execute at most once.
    if not coupled["record"]["passed"] and recovery_session["mode"] != "off":
        coupled = recover(
            coupled,
            times,
            P,
            R,
            masks,
            reference,
            manifest,
            points,
            tips,
            wristP,
            wristR,
            before,
            output,
            stage,
            session=recovery_session,
        )
    (output / "task_tcp_mapping_result.json").write_text(json.dumps(coupled["record"], indent=2))
    if not coupled["record"]["passed"]:
        raise RuntimeError("Sharpa original task-TCP mapping failed; rejected candidate and evidence preserved")
    # A selected recovery has already passed fresh native58, command/speed,
    # camera/workspace, collision and all-frame raster checks. Do not run a
    # legacy optional scene polish again and destroy that feasible result.
    if not coupled["record"].get("native_and_pixel_acceptance_passed"):
        from .scene_mapping import refine

        try:
            coupled = refine(
                _protected_copy(coupled),
                times,
                P,
                R,
                masks,
                before,
                wristP,
                wristR,
                points,
                reference["obs_head_pose"],
                reference["intrinsics"],
                (manifest["height"], manifest["width"]),
                output / "view_and_temporal_mapping",
                stage,
            )
        except RuntimeError:
            # Baseline keeps its native solver/refiner path. Only after its
            # scene or raster rejection may a real global initializer recover
            # it; force does not bypass candidate acceptance or the once limit.
            if recovery_session["mode"] == "off":
                raise
            recovered = recover(
                coupled,
                times,
                P,
                R,
                masks,
                reference,
                manifest,
                points,
                tips,
                wristP,
                wristR,
                before,
                output,
                stage,
                force=True,
                session=recovery_session,
            )
            if recovered is coupled or not recovered["record"].get("native_and_pixel_acceptance_passed"):
                raise
            coupled = recovered
    (output / "task_tcp_mapping_result.json").write_text(json.dumps(coupled["record"], indent=2))
    np.savez_compressed(output / "task_tcp_goals.npz", positions=P, rotations=R, masks=masks)
    np.savez_compressed(output / "palm_goals.npz", positions=wristP, rotations=wristR)
    (output / "target_definition.json").write_text(
        json.dumps(
            dict(
                goal_profile="frozen_orientation_eq12_articulated_sharpa_tcp",
                task_goal_seed=str(seed_path),
                task_goal_seed_sha256=sha(seed_path),
                source_reference_sha256=sha(reference_path),
                palm_wrist_is_soft_morphology_reference=True,
            ),
            indent=2,
        )
    )
    (output / "collisions.json").write_text(json.dumps(coupled["collisions"], indent=2))
    # Report the actual final jointly optimized fingers. Never run finger IK
    # after the final task solve, which would invalidate its accepted TCP.
    final_model = coupled["model"]
    data = mujoco.MjData(final_model)
    finger_report = {}
    for side, s in enumerate(["left", "right"]):
        adr = indices(final_model, s)["finger_qadr"]
        errors = []
        for t, row in enumerate(coupled["qpos"]):
            if not observed[s][t]:
                continue
            data.qpos[:] = row
            mujoco.mj_forward(final_model, data)
            actual = np.array(
                [
                    data.site_xpos[final_model.site(f"{s}_hand_{s}_{digit}_tip").id]
                    for digit in ["thumb", "index", "middle", "ring", "pinky"]
                ]
            )
            errors.extend(np.linalg.norm(actual - tips[t, side], axis=1))
        finger_report[s] = dict(
            fingertip_position_mean_m=float(np.mean(errors)) if errors else None,
            fingertip_position_max_m=float(np.max(errors)) if errors else None,
            finger_dofs=22,
            joint_speed_max_rad_s=float(
                np.max(abs(np.diff(coupled["qpos"][:, adr], axis=0) / np.diff(times)[:, None]))
            ),
            observed_frames=int(observed[s].sum()),
            jointly_optimized_with_arms=True,
            contact_and_self_collision_validated=False,
        )
    return coupled, finger_report
