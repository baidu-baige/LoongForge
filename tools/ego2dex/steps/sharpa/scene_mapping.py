# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Conservative task-workspace/view placement and full-track temporal polish."""

import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter


def refine(
    initial,
    times,
    goalsP,
    goalsR,
    masks,
    angular_baseline,
    wristP,
    wristR,
    raw_points,
    head,
    K,
    shape,
    output,
    stage=None,
):
    from .coupled_mapping_v53 import remap_arms as place
    from .coupled_mapping_v54 import remap_arms as smooth
    from .projection_quality import verify

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n = len(times)
    if stage:
        stage("camera_and_workspace_base_placement")
    sample = np.unique(np.r_[np.arange(0, n, 3), n - 1]).astype(int)
    tips = np.stack([raw_points[s][:, [4, 8, 12, 16, 20]].copy() for s in ["left", "right"]], axis=1)
    window = min(11, n if n % 2 else n - 1)
    if window >= 5:
        tips = savgol_filter(tips, window, 3, axis=0)
    for side in range(2):
        tips[~masks[:, side], side] = np.nan
    coarse = place(
        initial["model"],
        initial["qpos"][sample],
        times[sample],
        output / "placement",
        target_positions=goalsP[sample],
        target_rotations=goalsR[sample],
        initialization="existing",
        optimize_fingers=True,
        finger_targets=tips[sample],
        palm_reference_positions=wristP[sample],
        palm_reference_rotations=wristR[sample],
        target_masks=masks[sample],
        angular_baseline=angular_baseline,
        head_poses=head[sample],
        camera_K=K,
        image_shape=shape,
    )
    q = np.stack([np.interp(times, times[sample], coarse["qpos"][:, col]) for col in range(58)], axis=1)
    if stage:
        stage("all_source_frames_fixed_base_joint58_temporal_polish")
    result = smooth(
        coarse["model"],
        q,
        times,
        output / "full_polish",
        target_positions=goalsP,
        target_rotations=goalsR,
        initialization="existing",
        optimize_fingers=True,
        finger_targets=tips,
        palm_reference_positions=wristP,
        palm_reference_rotations=wristR,
        target_masks=masks,
        angular_baseline=angular_baseline,
        head_poses=head,
        camera_K=K,
        image_shape=shape,
    )
    (output / "scene_mapping_result.json").write_text(json.dumps(result["record"], indent=2))
    if not result["record"]["passed"]:
        raise RuntimeError("Scene/view placement or final task-TCP mapping failed; candidate retained and not rendered")
    if stage:
        stage("all_frame_calibrated_camera_projection_verification")
    projection = verify(
        result["spec"], result["qpos"], head, K, shape, raw_points, masks, output / "projection_verification.json"
    )
    result["record"]["projection_quality_passed"] = projection["passed"]
    if not projection["passed"]:
        raise RuntimeError(
            "Task kinematics passed but rendered arm occlusion failed; separate projection evidence retained"
        )
    result["record"]["scene_geometry_reconstructed"] = False
    result["record"]["view_scope"] = (
        "Conservative task-workspace and camera avoidance; no reconstructed cloth/object collision claim"
    )
    return result
