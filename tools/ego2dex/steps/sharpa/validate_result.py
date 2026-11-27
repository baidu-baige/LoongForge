# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Independent acceptance of exported native Sharpa models and videos.

Reloads physical results and measures every frame with native contact checks
at frames and adjacent midpoints. Solver pass assertions are never trusted.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from steps.hands.acceptance import current, pixel_checks
from steps.retarget.joint_pose import angular_eligible
from .mount_contract import audit_connection
from .projection_quality import verify as verify_pixels
from .rigid_retarget import arm_contacts
from .task_tcp import track, angular_metrics


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def motor_mapping(model):
    """Infer only the pinned native one-joint/one-actuator position model."""
    if not (model.nq == model.nv == model.njnt == model.nu == 58):
        raise ValueError("Sharpa validation requires 58 scalar physical joints and independent actuators")
    if model.neq or model.ntendon:
        raise ValueError("Unexpected Sharpa coupling/transmission; an explicit mapping is required")
    matrix = np.zeros((58, 58))
    addresses, motor_names, actuator_names = [], [], []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError("Only scalar native Sharpa coordinates are supported")
        if int(model.jnt_qposadr[j]) != j or int(model.jnt_dofadr[j]) != j:
            raise ValueError("Native Sharpa scalar coordinate order changed")
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Non-joint transmission cannot be inferred from physical qpos")
        np.testing.assert_array_equal(model.actuator_gear[a], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if int(model.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE):
            raise ValueError("Unexpected stateful actuator")
        if int(model.actuator_gaintype[a]) != int(mujoco.mjtGain.mjGAIN_FIXED) or int(
            model.actuator_biastype[a]
        ) != int(mujoco.mjtBias.mjBIAS_AFFINE):
            raise ValueError("Unexpected nonlinear actuator")
        joint = int(model.actuator_trnid[a, 0])
        if not 0 <= joint < model.njnt:
            raise ValueError("Invalid actuator joint")
        kp = float(model.actuator_gainprm[a, 0])
        bias = np.asarray(model.actuator_biasprm[a, :3])
        if not (kp > 0 and bias[0] == 0 and bias[1] == -kp and bias[2] <= 0):
            raise ValueError("Native actuator is not a stateless position target")
        address = int(model.jnt_qposadr[joint])
        addresses.append(address)
        matrix[address, a] = 1.0
        motor_names.append(model.joint(joint).name)
        actuator_names.append(model.actuator(a).name)
    if set(addresses) != set(range(58)) or len(set(addresses)) != 58:
        raise ValueError("Actuator mapping is not a bijection over the native physical coordinates")
    return np.asarray(addresses, dtype=int), matrix, motor_names, actuator_names


def _decode_video(path, width, height, frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Cannot decode Sharpa output video")
    count = 0
    dimensions_ok = True
    while True:
        ok, image = capture.read()
        if not ok:
            break
        dimensions_ok &= image.shape[:2] == (height, width)
        count += 1
    capture.release()
    return count, bool(dimensions_ok and count == frames)


def validate(
    folder,
    hand="sharpa",
    case=None,
    reference_dir=None,
    seed_source=None,
    *,
    report_path=None,
    write_artifacts=True,
    require_goal_match=True,
):
    """Recompute raw/stabilized task and exported model/video acceptance.

    Raw-reference comparisons can set require_goal_match=False because the
    frozen raw target intentionally differs from a stabilized target. Formal
    active-target validation requires the recorded task goals to match.
    """
    if hand != "sharpa":
        raise ValueError("Sharpa validator cannot validate a different hand")
    folder = Path(folder)
    reference = Path(reference_dir)
    manifest = json.loads((reference / "manifest.json").read_text())
    episode = Path(manifest["episode"]).name
    case = str(case) if case is not None else reference.name
    seed_path = Path(seed_source)
    report_path = Path(report_path) if report_path is not None else folder / "independent_validation.json"
    pixel_path = (
        (folder / "projection_verification.json")
        if report_path.name == "independent_validation.json"
        else report_path.with_name(report_path.stem + "_projection_verification.json")
    )
    acceptance = current()
    acceptance_record = acceptance.report()
    acceptance_record.update(
        native_speed_caps=False,
        configured_pipeline_speed_caps=True,
        speed_limit_source="Sharpa configured 3 rad/s; vendor MJCF has no declared velocity rating",
    )
    ref = np.load(reference / "hand_reference.npz")
    seed = np.load(seed_path)
    trajectory = np.load(folder / "trajectory.npz")
    q = np.asarray(trajectory["qpos"], dtype=float)
    n = int(manifest["frames"])
    if n < 2 or q.shape != (n, 58) or not np.isfinite(q).all():
        raise ValueError("Sharpa trajectory must contain finite native physical coordinates for all source frames")
    for key, refkey in [
        ("timestamps_ns", "timestamps_ns"),
        ("source_frame_indices", "source_frame_indices"),
        ("obs_head_pose", "obs_head_pose"),
        ("camera_K", "intrinsics"),
    ]:
        np.testing.assert_array_equal(trajectory[key], ref[refkey], err_msg=key)
    np.testing.assert_array_equal(trajectory["source_frame_indices"], np.arange(n))
    times = (ref["timestamps_ns"] - ref["timestamps_ns"][0]) / 1e9
    np.testing.assert_allclose(trajectory["timestamp"], times, atol=1e-9, rtol=0)
    if not np.all(np.diff(times) > 0):
        raise ValueError("Source clock is not strictly increasing")
    P, R, masks = seed["p"], seed["R"], seed["masks"].astype(bool)
    if P.shape != (n, 2, 3) or R.shape != (n, 2, 3, 3) or masks.shape != (n, 2) or not masks.any():
        raise ValueError("Source task goals do not match the frozen frame/hand contract")
    goals = np.load(folder / "task_tcp_goals.npz")
    goals_match = bool(
        np.array_equal(goals["positions"], P)
        and np.array_equal(goals["rotations"], R)
        and np.array_equal(goals["masks"].astype(bool), masks)
    )
    before = json.loads(seed_path.with_name(episode + "_joint_pose.json").read_text())["before"]
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    addresses, matrix, motor_names, actuator_names = motor_mapping(model)
    names = [model.joint(j).name for j in range(model.njnt)]
    np.testing.assert_array_equal(trajectory["state_names"], np.asarray(names))
    flags = int(model.opt.disableflags)
    native_contacts_enabled = not (
        flags & (int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))
    )
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    try:
        actual_p, actual_r = track(model, q)
        connection = audit_connection(model, q)
    finally:
        model.opt.disableflags = flags
    err_p = np.linalg.norm(actual_p - P, axis=-1)
    err_r = Rotation.from_matrix((R.transpose(0, 1, 3, 2) @ actual_r).reshape(-1, 3, 3)).magnitude().reshape(n, 2)
    position, rotation = float(err_p[masks].max()), float(err_r[masks].max())
    angular = angular_metrics(actual_r, R, masks)
    angular_ok = bool(angular_eligible(angular, before))
    collision = (
        arm_contacts(model, q)
        if native_contacts_enabled
        else {
            "arm_collision_samples": None,
            "arm_collisions": [],
            "frame_and_midpoint_samples": 0,
            "reason": "Exported model disabled native contacts; cannot certify collision acceptance",
        }
    )
    speed = np.max(np.abs(np.diff(q, axis=0) / np.diff(times)[:, None]), axis=0)
    speed_ratio = float(np.max(speed / 3.0))
    limited = np.flatnonzero(model.jnt_limited)
    limit_error = (
        float(
            max(
                0.0,
                np.max(model.jnt_range[limited, 0] - q[:, limited]),
                np.max(q[:, limited] - model.jnt_range[limited, 1]),
            )
        )
        if len(limited)
        else 0.0
    )
    motors = q[:, addresses]
    np.testing.assert_allclose(motors @ matrix.T, q, atol=0, rtol=0)
    controls = np.flatnonzero(model.actuator_ctrllimited)
    ctrl_error = (
        float(
            max(
                0.0,
                np.max(model.actuator_ctrlrange[controls, 0] - motors[:, controls]),
                np.max(motors[:, controls] - model.actuator_ctrlrange[controls, 1]),
            )
        )
        if len(controls)
        else 0.0
    )
    pixels = verify_pixels(
        mujoco.MjSpec.from_file(str(folder / "scene.xml")),
        q,
        ref["obs_head_pose"],
        ref["intrinsics"],
        (manifest["height"], manifest["width"]),
        {s: ref[s + "_obs_keypoints"].reshape(n, 21, 3) for s in ("left", "right")},
        masks,
        pixel_path,
    )
    pixels["acceptance_profile"] = acceptance_record
    _write(pixel_path, pixels)
    count, video_ok = _decode_video(folder / "robot_on_bg.mp4", manifest["width"], manifest["height"], n)
    checks = dict(
        task_tcp_position=position <= acceptance.position_m,
        task_full_rotation=rotation <= acceptance.rotation_rad,
        original_orientation_angular=angular_ok,
        all_joint_limits=limit_error <= 1e-5,
        all_configured_speed_caps=speed_ratio <= 1 + 1e-5,
        actuator_command_limits=ctrl_error <= 1e-5,
        coupling_model_verified=model.neq == 0 and model.nu == 58,
        independent_motor_reconstruction=True,
        rigid_mount=connection["passed"],
        native_collision_engine_enabled=native_contacts_enabled,
        native_arm_contacts_zero=collision["arm_collision_samples"] == 0,
        all_frame_camera_pixels=bool(pixels["passed"] and all(pixel_checks(pixels["measurements"]).values())),
        video_all_original_frames=video_ok,
        source_clock_camera_frames_preserved=True,
        active_target_goals_recorded=goals_match if require_goal_match else True,
    )
    if write_artifacts:
        np.savez_compressed(
            folder / "ik_quality.npz",
            ik_ok=(err_p <= acceptance.position_m) & (err_r <= acceptance.rotation_rad) & masks,
            ik_err_pos=err_p,
            ik_err_rot=err_r,
            target_masks=masks,
        )
        np.savez_compressed(
            folder / "motor_actions.npz",
            motor_state=motors,
            motor_target_next=np.r_[motors[1:], motors[-1:]],
            motor_names=np.asarray(motor_names),
            actuator_names=np.asarray(actuator_names),
            physical_state=q,
            physical_state_names=np.asarray(names),
            timestamp=times,
            source_frame_indices=trajectory["source_frame_indices"],
            physical_from_motor_matrix=matrix,
            physical_from_motor_offset=np.zeros(model.nq),
            hardware_execution_validated=False,
        )
    report = dict(
        hand=hand,
        case=case,
        episode=episode,
        acceptance_profile=acceptance_record,
        passed=all(checks.values()),
        checks=checks,
        frames=n,
        decoded_video_frames=count,
        task_tcp_position_max_m=position,
        task_tcp_rotation_max_rad=rotation,
        angular_metrics=angular,
        speed_to_configured_cap_max_ratio=speed_ratio,
        configured_speed_cap_rad_s=3.0,
        native_vendor_speed_limit_declared=False,
        joint_limit_violation_max_rad=limit_error,
        actuator_command_limit_violation_max_rad=ctrl_error,
        coupling_residual_max_rad=0.0,
        coupling_scope="58 independent joint actuators; no equality or tendon constraints in native asset",
        mount=connection,
        collisions=collision,
        physical_state_dimension=58,
        independent_action_dimension=58,
        recorded_goals_match_this_reference=goals_match,
        active_goal_match_required=require_goal_match,
        source_camera_clock_frames_preserved=True,
        seed_source=str(seed_path),
        object_contact_validated=False,
        internal_hand_contact_validated=False,
        dynamics_validated=False,
        hardware_validated=False,
        training_validated=False,
        sha256={
            p.name: _sha(p)
            for p in (folder / "scene.xml", folder / "trajectory.npz", folder / "robot_on_bg.mp4", seed_path)
        },
    )
    _write(report_path, report)
    print("INDEPENDENT", hand, case, report["passed"], position, rotation, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--seed")
    args = parser.parse_args()
    result = validate(args.output, "sharpa", args.case, reference_dir=args.reference, seed_source=args.seed)
    raise SystemExit(0 if result["passed"] else 2)
