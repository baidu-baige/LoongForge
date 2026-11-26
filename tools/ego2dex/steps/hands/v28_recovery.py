# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Use the v28 arm-only initializer as a bounded recovery, never as acceptance."""

import json
import os
import time
from pathlib import Path

import numpy as np


def policy():
    mode = os.environ.get("EGO2ROBOT_HAND_INITIALIZER", "v28-v29")
    if mode not in ("v28-v29", "v28", "off"):
        raise ValueError("EGO2ROBOT_HAND_INITIALIZER must be v28-v29, v28 or off")
    return mode


def recover(
    result, times, P, R, masks, ref, manifest, points, tips, wristP, wristR, workspace, before, output, place, polish
):
    from .global_initializer_v28 import initialize
    from .projection_quality import verify
    from .registry import affine

    output = Path(output)
    report_path = output / "v28_recovery.json"
    report = dict(
        policy=policy(), attempted=False, selected="baseline", acceptance_gates_unchanged=True, started=time.time()
    )
    if report["policy"] == "off" or result["record"]["passed"]:
        report["reason"] = "disabled" if report["policy"] == "off" else "already_passed"
        report_path.write_text(json.dumps(report, indent=2))
        return result
    report.update(attempted=True, reason="full_frame_baseline_rejected")
    report_path.write_text(json.dumps(report, indent=2))
    root = output / "v28_recovery"
    root.mkdir(parents=True, exist_ok=True)
    try:
        # The archived implementation optimizes 7 arm coordinates per side
        # and a static 6D base. Fingers keep their current projected posture.
        model = result["model"]
        _, recovered_model, q = initialize(model, result["qpos"], times, P, R, masks, root / "initializer")
        finger = np.concatenate([affine(model, side)[0] for side in ("left", "right")])
        np.testing.assert_allclose(q[:, finger], result["qpos"][:, finger], atol=1e-6, rtol=0)
        report["initializer_completed"] = True
        report_path.write_text(json.dumps(report, indent=2))
        sample = np.unique(np.r_[np.arange(0, len(times), 3), len(times) - 1]).astype(int)
        common = dict(
            initialization="existing",
            optimize_fingers=True,
            angular_baseline=before,
            camera_K=ref["intrinsics"],
            image_shape=(manifest["height"], manifest["width"]),
        )
        candidate = place(
            recovered_model,
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
        full_q = np.stack(
            [np.interp(times, times[sample], candidate["qpos"][:, j]) for j in range(recovered_model.nq)], axis=1
        )
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
                report["selected"] = "v28"
                return candidate
        report["reason"] = "candidate_rejected_baseline_retained"
        return result
    except (RuntimeError, ValueError, AssertionError, np.linalg.LinAlgError) as exc:
        report.update(reason="recovery_failed_baseline_retained", error=str(exc))
        return result
    finally:
        report["finished"] = time.time()
        report_path.write_text(json.dumps(report, indent=2))
