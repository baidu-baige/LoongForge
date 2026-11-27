# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collision-aware v29 recovery; only native gates and raster checks accept it."""

import json
import time
from pathlib import Path

import numpy as np


def recover(
    result, times, P, R, masks, ref, manifest, points, tips, wristP, wristR, workspace, before, output, place, polish
):
    from .global_initializer_v29 import initialize
    from .projection_quality import verify
    from .registry import coupling_error, speed_limits
    from .mount_contract import audit_connection

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "v29_recovery.json"
    report = dict(attempted=False, selected="baseline", acceptance_gates_unchanged=True, started=time.time())
    if result["record"]["passed"]:
        report["reason"] = "already_passed"
        report_path.write_text(json.dumps(report, indent=2))
        return result
    if len(times) < 4:
        report["reason"] = "fewer_than_four_frames_baseline_retained"
        report_path.write_text(json.dumps(report, indent=2))
        return result
    report.update(attempted=True, reason="baseline_and_v28_remain_rejected")
    report_path.write_text(json.dumps(report, indent=2))
    root = output / "v29_recovery"
    root.mkdir(parents=True, exist_ok=True)
    try:
        _, model, q = initialize(result["model"], result["qpos"], times, P, R, masks, root / "initializer")
        # This initializer optimizes real hand motor coordinates as well as
        # arms/bases. Its soft contact penalties never certify acceptance.
        assert np.isfinite(q).all() and q.shape == result["qpos"].shape
        assert coupling_error(model, q) < 1e-8
        assert np.all(q >= model.jnt_range[:, 0] - 1e-5)
        assert np.all(q <= model.jnt_range[:, 1] + 1e-5)
        assert np.max(abs(np.diff(q, axis=0) / np.diff(times)[:, None]) / speed_limits(model)) <= 1 + 1e-5
        assert audit_connection(model, q)["passed"]
        report["initializer_completed"] = True
        report_path.write_text(json.dumps(report, indent=2))
        sample = np.unique(np.r_[np.arange(0, len(times), 3), len(times) - 1]).astype(int)
        if len(sample) < 4:
            sample = np.arange(len(times))
        common = dict(
            initialization="existing",
            optimize_fingers=True,
            angular_baseline=before,
            camera_K=ref["intrinsics"],
            image_shape=(manifest["height"], manifest["width"]),
        )
        candidate = place(
            model,
            q[sample],
            times[sample],
            root / "placement",
            target_positions=P[sample],
            target_rotations=R[sample],
            target_masks=masks[sample],
            head_poses=ref["obs_head_pose"][sample],
            finger_targets=tips[sample],
            palm_reference_positions=wristP[sample],
            palm_reference_rotations=wristR[sample],
            hand_workspace_points=workspace[sample],
            **common,
        )
        full_q = np.stack([np.interp(times, times[sample], candidate["qpos"][:, j]) for j in range(model.nq)], axis=1)
        candidate = polish(
            candidate["model"],
            full_q,
            times,
            root / "all_frame_polish",
            target_positions=P,
            target_rotations=R,
            target_masks=masks,
            head_poses=ref["obs_head_pose"],
            finger_targets=tips,
            palm_reference_positions=wristP,
            palm_reference_rotations=wristR,
            hand_workspace_points=workspace,
            **common,
        )
        if not candidate["record"]["passed"]:
            from .conic_finish import refine

            candidate = refine(
                candidate,
                times,
                P,
                R,
                masks,
                ref["obs_head_pose"],
                ref["intrinsics"],
                common["image_shape"],
                before,
                root / "conic_finish",
            )
        report["candidate_record"] = candidate["record"]
        if candidate["record"]["passed"]:
            pixels = verify(
                candidate["spec"],
                candidate["qpos"],
                ref["obs_head_pose"],
                ref["intrinsics"],
                common["image_shape"],
                points,
                masks,
                root / "projection_verification.json",
            )
            report["all_frame_pixels_passed"] = bool(pixels["passed"])
            if pixels["passed"]:
                report.update(selected="v29", reason="physical_and_pixel_gates_passed")
                return candidate
        report["reason"] = "candidate_rejected_baseline_retained"
        return result
    except (RuntimeError, ValueError, AssertionError, np.linalg.LinAlgError) as exc:
        report.update(reason="recovery_failed_baseline_retained", error=str(exc))
        return result
    finally:
        report["finished"] = time.time()
        report_path.write_text(json.dumps(report, indent=2))
