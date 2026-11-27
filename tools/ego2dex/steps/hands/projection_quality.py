# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Measure actual visible robot coverage in the original calibrated camera."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .panda_scene import standalone as _standalone_xml
from .view_constraints import housing_body_names, proximal_housing_names
from .acceptance import current, pixel_checks
from steps.sharpa.source_camera import render_to_calibrated_pixels
from steps.retarget.rendering import mat_to_quat
from steps.retarget.calibration import covering_focal, warp_camera


from .rendering import make_renderer, render_geom_subset_mask


def verify(spec, q, head, K, shape, points, masks, output):
    """All-frame raster checks; this does not certify object/scene contact."""
    H, W = shape
    h, w = H // 2, W // 2
    focal = covering_focal(K, shape)
    tree = _standalone_xml(spec)
    world = tree.find("worldbody")
    for child in list(world):
        if child.tag == "body" and child.get("mocap") == "true":
            world.remove(child)
    cam = ET.SubElement(world, "body", name="quality_camera_root", mocap="true")
    ET.SubElement(cam, "camera", name="quality_camera", fovy=str(np.degrees(2 * np.arctan(H / (2 * focal)))))
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    data = mujoco.MjData(model)
    # Rendering and projection depend on FK only. Collision acceptance is
    # checked separately on the exported model with native contact enabled.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    camera = model.camera("quality_camera").id
    mc = int(model.body("quality_camera_root").mocapid[0])
    renderer = make_renderer(model, w, h)
    warp = render_to_calibrated_pixels(np.diag([0.5, 0.5, 1]) @ K, (h, w), focal * 0.5)
    prox = []
    arms = []
    hands = []
    rows = []
    housing = housing_body_names(model)
    proximal_housing = proximal_housing_names(model)
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name or ""
        if model.geom_group[g] == 3 or body in ["left_link0", "right_link0"]:
            model.geom_rgba[g, 3] = 0
        if model.geom_rgba[g, 3] <= 0:
            continue
        if "_hand_" in body and body not in housing:
            hands.append(g)
        elif body.startswith(("left_", "right_")):
            arms.append(g)
            if body in proximal_housing or body in [
                side + "_link" + str(k) for side in ["left", "right"] for k in range(6)
            ]:
                prox.append(g)
    source = np.stack([points[s] for s in ["left", "right"]], axis=1)
    try:
        for t, state in enumerate(q):
            R = Rotation.from_quat(head[t, [4, 5, 6, 3]]).as_matrix()
            data.qpos[:] = state
            data.mocap_pos[mc] = head[t, :3]
            data.mocap_quat[mc] = mat_to_quat(R @ np.diag([1, -1, -1]))
            mujoco.mj_forward(model, data)
            from .rendering import visible_category_masks

            pm, am = visible_category_masks(model, data, renderer, camera, [prox, arms])
            pa = warp_camera(pm.astype(np.uint8), warp, True) > 0
            aa = warp_camera(am.astype(np.uint8), warp, True) > 0
            hand = (
                warp_camera(render_geom_subset_mask(model, data, renderer, camera, hands).astype(np.uint8), warp, True)
                > 0
            )
            pc = (source[t] - head[t, :3]) @ R
            uv = pc @ (np.diag([0.5, 0.5, 1]) @ K).T
            uv = uv[:, :, :2] / np.maximum(uv[:, :, 2:], 0.01)
            valid = (pc[:, :, 2] > 0.05) & masks[t, :, None]
            uv = uv[:, [4, 8, 12]][valid[:, [4, 8, 12]]]
            roi = np.zeros((h, w), np.uint8)
            if len(uv) >= 3:
                cv2.fillConvexPoly(roi, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
                roi = cv2.dilate(roi, np.ones((21, 21), np.uint8))
            rows.append(
                dict(
                    frame=t,
                    arm_image_fraction=float(aa.mean()),
                    work_region_occluded_fraction=float((pa & (roi > 0)).sum() / max(int(roi.sum()), 1)),
                    hand_occluded_by_proximal_fraction=float((pa & hand).sum() / max(int(hand.sum()), 1)),
                )
            )
    finally:
        renderer.close()
    values = dict(
        mean_work_region_occlusion=float(np.mean([r["work_region_occluded_fraction"] for r in rows])),
        max_work_region_occlusion=max(r["work_region_occluded_fraction"] for r in rows),
        mean_hand_proximal_occlusion=float(np.mean([r["hand_occluded_by_proximal_fraction"] for r in rows])),
        max_arm_image_fraction=max(r["arm_image_fraction"] for r in rows),
    )
    checks = pixel_checks(values)
    report = dict(
        acceptance_profile=current().report(),
        passed=all(checks.values()),
        frames=len(rows),
        all_source_frames_checked=True,
        source_camera_preserved=True,
        checks=checks,
        measurements=values,
        rows=rows,
        object_contact_or_scene_depth_certified=False,
        category_visibility_method="explicit physical geometry groups; semantic sites and rangefinder debug rays disabled",
    )
    Path(output).write_text(json.dumps(report, indent=2))
    return report
