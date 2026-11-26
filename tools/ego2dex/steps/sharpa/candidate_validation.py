# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fresh native Sharpa acceptance for an initializer or post-refinement result.

No solver pass flag or search-model contact result is trusted. A candidate is
rebuilt from native assets at its static arm bases, serialized at full mount
precision, compiled again, and checked at all source frames. Related native
robot contacts are additionally checked at every adjacent midpoint. Legacy
silhouette terms are diagnostic only; camera/workspace geometry and calibrated
all-frame raster measurements are separate mandatory acceptance checks.
"""

from steps.hands.acceptance import gate as _acceptance_gate
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from steps.hands.acceptance import current, pixel_checks
from steps.retarget.joint_pose import angular_eligible
from . import initializer_backend as backend
from .task_tcp import track, angular_metrics
from .rigid_retarget import arm_contacts
from .view_constraints_v8 import ViewConstraints
from .projection_quality import verify


def _write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _view_audit(model, q, head, intrinsics, shape, P, masks):
    """Split native terms by their actual kind; do not infer camera from max."""
    view = ViewConstraints(head, intrinsics, shape, P, masks)
    data = mujoco.MjData(model)
    maxima = dict(camera=0.0, workspace=0.0, view=0.0, hand_view=0.0)
    counts = dict(camera=0, workspace=0, view=0, hand_view=0)
    worst_hard, worst_soft = [], []
    for frame, row in enumerate(q):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        for value, _, link, kind in view.terms(model, data, frame, []):
            if kind not in maxima or not np.isfinite(value):
                raise ValueError("Unknown or non-finite native view term: " + str(kind))
            value = float(value)
            maxima[kind] = max(maxima[kind], value)
            if value > 0:
                counts[kind] += 1
            item = dict(frame=frame, link=int(link), kind=kind, violation_m=value)
            if kind in ("camera", "workspace") and value > _acceptance_gate(0.014, "layout_m"):
                worst_hard.append(item)
            elif kind in ("view", "hand_view") and value > 0:
                worst_soft.append(item)
    return dict(
        camera_keepout_max_violation_m=maxima["camera"],
        workspace_envelope_max_violation_m=maxima["workspace"],
        projected_mesh_proxy_max_violation_m=max(maxima["view"], maxima["hand_view"]),
        measured_view_geometry_passed=max(maxima["camera"], maxima["workspace"]) <= _acceptance_gate(0.014, "layout_m"),
        camera_and_workspace_limit_m=_acceptance_gate(0.014, "layout_m"),
        frame_samples=len(q),
        term_counts=counts,
        worst_hard_samples=sorted(worst_hard, key=lambda item: item["violation_m"], reverse=True)[:12],
        worst_soft_silhouette_samples=sorted(worst_soft, key=lambda item: item["violation_m"], reverse=True)[:12],
        silhouette_is_acceptance_gate=False,
        actual_pixel_verification_required=True,
        reconstructed_scene_collision_validated=False,
        geometry_hidden_to_improve_view=False,
    )


def check(candidate, times, P, R, masks, ref, manifest, points, before, output, *, label="candidate", stage=None):
    """Return a freshly checked native candidate and its independent report.

    A rejected candidate has no acceptance marker. Numerical/model failures
    raise explicitly; a caller may retain its baseline instead. This function
    never changes the supplied model, trajectory, goals, clock or camera.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    profile = current()
    source_model = candidate["model"]
    source_q = np.asarray(candidate["qpos"])
    times = np.asarray(times, dtype=float)
    P, R = np.asarray(P), np.asarray(R)
    masks = np.asarray(masks, dtype=bool)
    n = len(times)
    if (
        n < 2
        or source_q.shape != (n, 58)
        or not np.isfinite(source_q).all()
        or P.shape != (n, 2, 3)
        or R.shape != (n, 2, 3, 3)
        or masks.shape != (n, 2)
        or not masks.any()
        or not np.isfinite(P).all()
        or not np.isfinite(R).all()
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("Malformed Sharpa candidate, task goals or source clock")
    if int(manifest["frames"]) != n:
        raise ValueError("Candidate frame count differs from immutable reference")
    head = np.asarray(ref["obs_head_pose"])
    K = np.asarray(ref["intrinsics"])
    if head.shape != (n, 7) or K.shape != (3, 3):
        raise ValueError("Malformed source camera reference")
    expected_times = (np.asarray(ref["timestamps_ns"]) - ref["timestamps_ns"][0]) / 1e9
    np.testing.assert_allclose(times, expected_times, rtol=0, atol=1e-9)
    for side in ("left", "right"):
        if np.asarray(points[side]).shape != (n, 21, 3):
            raise ValueError("Malformed source hand keypoints")
    backend.model_contract(source_model)

    # A distinct native model prevents inherited search flags, enlarged
    # margins, altered geometry, hidden visual meshes or virtual base joints
    # from certifying themselves. Only static base poses and named qpos move.
    source_data = mujoco.MjData(source_model)
    source_data.qpos[:] = source_q[0]
    mujoco.mj_forward(source_model, source_data)
    bases = {
        side: dict(
            pos=source_data.xpos[source_model.body(side + "_link0").id].copy(),
            quat=source_data.xquat[source_model.body(side + "_link0").id].copy(),
        )
        for side in ("left", "right")
    }
    native_spec = backend.build(base_poses=bases, include_floor=False)
    tree = backend.standalone(native_spec)
    scene_path = output / "candidate_scene.xml"
    scene_path.write_text(ET.tostring(tree, encoding="unicode"))
    spec = mujoco.MjSpec.from_file(str(scene_path))
    model = spec.compile()
    contract = backend.model_contract(model)
    forbidden = int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    if int(model.opt.disableflags) & forbidden:
        raise ValueError("Fresh physical model disables native collision acceptance")
    old_names = [source_model.joint(joint).name for joint in range(source_model.njnt)]
    new_names = [model.joint(joint).name for joint in range(model.njnt)]
    if len(set(old_names)) != 58 or set(old_names) != set(new_names):
        raise ValueError("Native candidate joint identity differs from source")
    q = np.empty((n, model.nq), dtype=float)
    for joint, name in enumerate(new_names):
        q[:, int(model.jnt_qposadr[joint])] = source_q[:, int(source_model.joint(name).qposadr[0])]
    np.savez_compressed(output / "candidate_track.npz", qpos=q, timestamp=times)
    if stage:
        stage("fresh_native58_recovery_acceptance")
    actual_p, actual_r = track(model, q)
    errors_p = np.linalg.norm(actual_p - P, axis=-1)
    errors_r = Rotation.from_matrix((R.transpose(0, 1, 3, 2) @ actual_r).reshape(-1, 3, 3)).magnitude().reshape(n, 2)
    position, rotation = float(np.max(errors_p[masks])), float(np.max(errors_r[masks]))
    angular = angular_metrics(actual_r, R, masks)
    angular_ok = bool(angular_eligible(angular, before))
    limits = float(max(0.0, np.max(model.jnt_range[:, 0] - q), np.max(q - model.jnt_range[:, 1])))
    motors = q[:, contract["actuator_addresses"]]
    np.testing.assert_array_equal(motors @ contract["physical_from_motor_matrix"].T, q)
    limited = np.flatnonzero(model.actuator_ctrllimited)
    command_violation = (
        float(
            max(
                0.0,
                np.max(model.actuator_ctrlrange[limited, 0] - motors[:, limited]),
                np.max(motors[:, limited] - model.actuator_ctrlrange[limited, 1]),
            )
        )
        if len(limited)
        else 0.0
    )
    speed = np.max(np.abs(np.diff(q, axis=0) / np.diff(times)[:, None]), axis=0)
    ratio = float(np.max(speed / backend.CONFIGURED_SPEED_CAP_RAD_S))
    mounts = backend.audit_connection(model, q)
    contacts = arm_contacts(model, q)
    layout = _view_audit(model, q, head, K, (manifest["height"], manifest["width"]), P, masks)
    checks = dict(
        task_tcp_position=position <= profile.position_m,
        task_full_rotation=rotation <= profile.rotation_rad,
        original_orientation_angular=angular_ok,
        all_joint_limits=limits <= 1e-5,
        actuator_command_limits=command_violation <= 1e-5,
        all_pipeline_speed_caps=ratio <= 1 + 1e-5,
        independent_actuator_reconstruction=True,
        rigid_mount=bool(mounts["passed"]),
        native_arm_contacts_zero=contacts["arm_collision_samples"] == 0,
        camera_keepout=layout["camera_keepout_max_violation_m"] <= _acceptance_gate(0.014, "layout_m"),
        workspace_envelope=layout["workspace_envelope_max_violation_m"] <= _acceptance_gate(0.014, "layout_m"),
    )
    pixels = dict(passed=False, evaluated=False, reason="A non-pixel hard gate rejected the candidate")
    if all(checks.values()):
        if stage:
            stage("fresh_all_frame_recovery_raster_acceptance")
        pixels = verify(
            spec,
            q,
            head,
            K,
            (manifest["height"], manifest["width"]),
            points,
            masks,
            output / "projection_verification.json",
        )
        pixels = dict(pixels, evaluated=True)
        pixels["passed"] = bool(pixels["passed"] and all(pixel_checks(pixels["measurements"], profile).values()))
    checks["all_frame_camera_pixels"] = bool(pixels["passed"] and pixels.get("frames") == n)
    accepted = all(checks.values())
    acceptance_report = profile.report()
    acceptance_report.update(
        native_speed_caps=False,
        configured_pipeline_speed_caps=True,
        speed_limit_source="configured 3 rad/s; native Sharpa MJCF has no velocity rating",
    )
    pixels["acceptance_profile"] = acceptance_report
    _write(output / "projection_verification.json", pixels)
    record = dict(
        label=label,
        passed=accepted,
        acceptance_profile=acceptance_report,
        eef_position_max_m=position,
        eef_rotation_max_rad=rotation,
        task_tcp_position_max_m=position,
        task_tcp_rotation_max_rad=rotation,
        goal_profile="frozen_orientation_eq12_articulated_sharpa_tcp",
        angular_metrics=angular,
        original_orientation_angular_gate_passed=angular_ok,
        view_layout=layout,
        projection_quality_passed=checks["all_frame_camera_pixels"],
        all_frame_pixels_evaluated=bool(pixels.get("evaluated")),
        arm_collision_samples=contacts["arm_collision_samples"],
        arm_speed_max_rad_s=float(np.max(speed)),
        all_controlled_speed_max_rad_s=float(np.max(speed)),
        speed_to_pipeline_cap_ratio=ratio,
        configured_speed_cap_rad_s=3.0,
        vendor_velocity_limit_declared=False,
        joint_limit_violation_max_rad=limits,
        actuator_command_limit_violation_max_rad=command_violation,
        optimized_joint_count=58,
        independent_motor_count=58,
        coupling_residual_max_rad=0.0,
        checks=checks,
        frames=n,
        frame_and_midpoint_samples=contacts["frame_and_midpoint_samples"],
        native_model_recompiled=True,
        native_collision_flags=int(model.opt.disableflags),
        goals_sha256=dict(positions=_array_hash(P), rotations=_array_hash(R), masks=_array_hash(masks)),
        source_reference_sha256=dict(
            camera=_array_hash(head), intrinsics=_array_hash(K), timestamps=_array_hash(ref["timestamps_ns"])
        ),
        geometry_hidden_to_improve_view=False,
        reconstructed_scene_collision_validated=False,
        hardware_validated=False,
        dynamics_validated=False,
        object_contact_validated=False,
    )
    if accepted:
        record["native_and_pixel_acceptance_passed"] = True
    report = dict(
        record=record,
        passed=accepted,
        checks=checks,
        pixels=pixels,
        collisions=contacts,
        mount=mounts,
        native_scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
    )
    _write(output / "native_acceptance.json", report)
    result = dict(
        spec=spec,
        model=model,
        qpos=q,
        bases=bases,
        record=record,
        collisions=contacts,
        connection=mounts,
        goal_positions=P.copy(),
        goal_rotations=R.copy(),
        target_masks=masks.copy(),
    )
    return result, report
