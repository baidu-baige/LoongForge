# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reload exported results and recompute acceptance outside the optimizer."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from steps.sharpa.task_tcp import track, angular_metrics
from steps.sharpa.rigid_retarget import arm_contacts
from steps.retarget.joint_pose import angular_eligible
from .mount_contract import audit_connection
from .acceptance import current, pixel_checks
from .registry import whole_affine, coupling_error, speed_limits


def validate(folder, hand, case, reference_dir=None, seed_source=None):
    acceptance = current()
    os.environ["EGO2ROBOT_END_EFFECTOR"] = hand
    folder = Path(folder)
    z = np.load(folder / "trajectory.npz")
    q = z["qpos"]
    m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    refdir = Path(reference_dir)
    ref = np.load(refdir / "hand_reference.npz")
    manifest = json.loads((refdir / "manifest.json").read_text())
    episode = Path(manifest["episode"]).name
    seedpath = Path(seed_source)
    seed = np.load(seedpath)
    masks = seed["masks"].astype(bool)
    np.testing.assert_array_equal(z["timestamps_ns"], ref["timestamps_ns"])
    np.testing.assert_array_equal(z["source_frame_indices"], ref["source_frame_indices"])
    np.testing.assert_array_equal(z["obs_head_pose"], ref["obs_head_pose"])
    np.testing.assert_array_equal(z["camera_K"], ref["intrinsics"])
    assert len(q) == manifest["frames"]
    before = json.loads(seedpath.with_name(episode + "_joint_pose.json").read_text())["before"]
    flags = int(m.opt.disableflags)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    P, R = track(m, q)
    position = float(np.max(np.linalg.norm(P - seed["p"], axis=-1)[masks]))
    angle = float(
        np.max(
            Rotation.from_matrix((seed["R"].transpose(0, 1, 3, 2) @ R).reshape(-1, 3, 3))
            .magnitude()
            .reshape(len(q), 2)[masks]
        )
    )
    mount = audit_connection(m, q)
    am = angular_metrics(R, seed["R"], masks)
    angular = angular_eligible(am, before)
    m.opt.disableflags = flags
    assert flags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) == 0
    collisions = arm_contacts(m, q)
    times = z["timestamp"]
    speed = np.max(abs(np.diff(q, axis=0) / np.diff(times)[:, None]), axis=0)
    ratio = float(np.max(speed / speed_limits(m)))
    limits = float(max(0.0, np.max(m.jnt_range[:, 0] - q), np.max(q - m.jnt_range[:, 1])))
    coupling = coupling_error(m, q)
    pixel = json.loads((folder / "projection_verification.json").read_text())
    assert pixel["frames"] == len(q) and pixel["all_source_frames_checked"]
    pixel["passed"] = all(pixel_checks(pixel["measurements"]).values())
    cap = cv2.VideoCapture(str(folder / "robot_on_bg.mp4"))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    checks = dict(
        task_tcp_position=position <= acceptance.position_m,
        task_full_rotation=angle <= acceptance.rotation_rad,
        original_orientation_angular=bool(angular),
        all_joint_limits=limits <= 1e-5,
        all_native_speed_caps=ratio <= 1 + 1e-5,
        coupling=coupling < 1e-8,
        rigid_mount=mount["passed"],
        native_arm_contacts_zero=collisions["arm_collision_samples"] == 0,
        all_frame_camera_pixels=bool(pixel["passed"]),
        video_all_original_frames=count == len(q),
    )
    errp = np.linalg.norm(P - seed["p"], axis=-1)
    errr = Rotation.from_matrix((seed["R"].transpose(0, 1, 3, 2) @ R).reshape(-1, 3, 3)).magnitude().reshape(len(q), 2)
    np.savez_compressed(
        folder / "ik_quality.npz",
        ik_ok=(errp <= acceptance.position_m) & (errr <= acceptance.rotation_rad) & masks,
        ik_err_pos=errp,
        ik_err_rot=errr,
        target_masks=masks,
    )
    b, M, names = whole_affine(m)
    motors = np.linalg.lstsq(M, (q - b).T, rcond=None)[0].T
    np.testing.assert_allclose(b + motors @ M.T, q, atol=1e-8, rtol=0)
    np.savez_compressed(
        folder / "motor_actions.npz",
        motor_state=motors,
        motor_target_next=np.r_[motors[1:], motors[-1:]],
        motor_names=np.asarray(names),
        physical_state=q,
        physical_state_names=z["state_names"],
        timestamp=times,
        source_frame_indices=z["source_frame_indices"],
        physical_from_motor_matrix=M,
        physical_from_motor_offset=b,
        hardware_execution_validated=False,
    )
    report = dict(
        hand=hand,
        case=case,
        acceptance_profile=acceptance.report(),
        passed=all(checks.values()),
        checks=checks,
        frames=len(q),
        decoded_video_frames=count,
        task_tcp_position_max_m=position,
        task_tcp_rotation_max_rad=angle,
        angular_metrics=am,
        speed_to_native_cap_max_ratio=ratio,
        joint_limit_violation_max_rad=limits,
        coupling_residual_max_rad=coupling,
        mount=mount,
        collisions=collisions,
        physical_state_dimension=m.nq,
        independent_action_dimension=len(names),
        object_contact_validated=False,
        internal_hand_contact_validated=False,
        dynamics_validated=False,
        hardware_validated=False,
        training_validated=False,
        sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [folder / "scene.xml", folder / "trajectory.npz", folder / "robot_on_bg.mp4", seedpath]
        },
    )
    (folder / "independent_validation.json").write_text(json.dumps(report, indent=2))
    print("INDEPENDENT", hand, case, report["passed"], position, angle, flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    validate(a.output, a.hand, a.case)
