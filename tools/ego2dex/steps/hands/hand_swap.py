# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Replace Orientation's Panda grippers and composite Sharpa into the source video.

The fixed factory mount and58joint solver preserve the original task TCP,
source camera, clock and cached background. Contact/physics remain separate.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import mink
import mujoco
import numpy as np
from mink.limits import Constraint
from scipy.spatial.transform import Rotation

from .panda_scene import build, indices
from steps.sharpa.arm_continuity import FrameConfigurationLimit
from steps.sharpa.source_camera import render_to_calibrated_pixels
from steps.retarget.calibration import covering_focal, warp_camera
from steps.retarget.rendering import (
    feather_mask,
    render_rgb_and_mask,
    render_geom_subset_mask,
    tone_map_robot,
    patch_arm_holes,
    temporal_median_depth_frame,
    depth_visibility_alpha,
    load_scene_depth,
    mat_to_quat,
)


from .rendering import make_renderer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


from .panda_scene import standalone as _standalone_xml


def _fixed_mounts(model, arm_rows, hand_points, valid):
    """Compatibility entry: the adapter is fixed by its model, never by a clip."""
    from .mount_contract import canonical_mounts

    return canonical_mounts()


class FrozenJoints:
    def __init__(self, model, moving):
        frozen = np.setdiff1d(np.arange(model.nv), moving)
        G = np.eye(model.nv)[frozen]
        self.constraint = Constraint(G=np.vstack([G, -G]), h=np.zeros(2 * len(frozen)))

    def compute_qp_inequalities(self, configuration, dt):
        return self.constraint


def _fingers(model, arm_rows, points, valid, times, output):
    rows = arm_rows.copy()
    configuration = mink.Configuration(model, q=rows[0])
    report = {}
    for side in ["left", "right"]:
        ids = indices(model, side)
        adr = ids["finger_qadr"]
        joints = np.array([np.flatnonzero(model.jnt_qposadr == a)[0] for a in adr])
        lo, hi = model.jnt_range[joints].T
        first = rows[0].copy()
        first[adr] = np.clip(model.qpos0[adr], lo + 0.001, hi - 0.001)
        configuration.update(q=first)
        site_names = [f"{side}_hand_{side}_{digit}_tip" for digit in ["thumb", "index", "middle", "ring", "pinky"]]
        tasks = [
            mink.FrameTask(frame_name=name, frame_type="site", position_cost=10.0, orientation_cost=0.0, lm_damping=1.0)
            for name in site_names
        ]
        posture = mink.PostureTask(model, cost=0.005)
        lock = FrozenJoints(model, model.jnt_dofadr[joints])
        limit = FrameConfigurationLimit(mink.ConfigurationLimit(model), model, adr)
        previous = first[adr].copy()
        errors = []
        for i in range(len(rows)):
            if not valid[side][i]:
                rows[i, adr] = previous
                continue
            initial = rows[i].copy()
            initial[adr] = previous
            configuration.update(q=initial)
            posture.set_target(initial)
            if i:
                limit.set_previous(previous, float(times[i] - times[i - 1]), max_speed=3.0)
            else:
                limit.clear()
            for task, target in zip(tasks, points[side][i, [4, 8, 12, 16, 20]]):
                task.set_target(mink.SE3(wxyz_xyz=np.r_[[1.0, 0, 0, 0], target]))
            steps = 160 if i == 0 else 10
            dt = 0.01 if i == 0 else float(times[i] - times[i - 1]) / steps
            for _ in range(steps):
                velocity = mink.solve_ik(
                    configuration, [posture, *tasks], dt, "daqp", damping=1e-5, limits=[limit, lock]
                )
                configuration.integrate_inplace(velocity, dt)
            previous = configuration.q[adr].copy()
            rows[i, adr] = previous
            tip = np.array([configuration.data.site(name).xpos for name in site_names])
            errors.append(np.linalg.norm(tip - points[side][i, [4, 8, 12, 16, 20]], axis=1))
            if i % 40 == 0:
                print("FINGER_RETARGET", side, i, len(rows), flush=True)
        error = np.asarray(errors)
        report[side] = dict(
            fingertip_position_mean_m=float(error.mean()) if len(error) else None,
            fingertip_position_max_m=float(error.max()) if len(error) else None,
            observed_frames=int(valid[side].sum()),
            joint_speed_max_rad_s=float(abs(np.diff(rows[:, adr], axis=0) / np.diff(times)[:, None]).max()),
            original_joint_limit_violation_max_rad=float(
                max(0.0, (lo - rows[:, adr]).max(), (rows[:, adr] - hi).max())
            ),
            finger_dofs=22,
            contact_and_self_collision_validated=False,
        )
        (output / "finger_ik_progress.json").write_text(json.dumps(report, indent=2))
    return rows, report


def swap_grippers(
    orientation_dir, prepared_input, output, stage=None, finger_seed=None, depth_mode="alpha", make_comparison=False
):
    from steps.retarget.episode import _estimate_scene_depth_alignment

    cache, inp, out = Path(orientation_dir), Path(prepared_input), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inp / "manifest.json").read_text())
    episode = Path(manifest["episode"]).name
    ref = np.load(inp / "hand_reference.npz")
    head = ref["obs_head_pose"].reshape(-1, 7)
    times = (ref["timestamps_ns"] - ref["timestamps_ns"][0]).astype(float) / 1e9
    n = len(times)
    old_path = cache / f"06_retarget/{episode}_ik.npz"
    seed_path = cache / f"06_retarget/{episode}_joint_pose_seed.npz"
    old = np.load(old_path if old_path.exists() else seed_path)
    assert len(old["qpos"]) == n == manifest["frames"]
    old_names = (
        old["state_names"].tolist()
        if "state_names" in old
        else [
            f"{side}_{joint}"
            for side in ["left", "right"]
            for joint in [*[f"joint{i}" for i in range(1, 8)], "finger_joint1", "finger_joint2"]
        ]
    )
    np.testing.assert_array_equal(ref["source_frame_indices"], np.arange(n))
    W, H = manifest["width"], manifest["height"]
    K = ref["intrinsics"]
    if "camera_K" in old:
        np.testing.assert_allclose(K, old["camera_K"], rtol=0, atol=1e-5)
    bases = (
        {side: dict(pos=old[side + "_base_pos"], quat=old[side + "_base_quat"]) for side in ["left", "right"]}
        if "left_base_pos" in old
        else {
            side: dict(pos=old["positions"][i], quat=old["quaternions"][i]) for i, side in enumerate(["left", "right"])
        }
    )
    if not old_path.exists():
        old_path = seed_path
    spec = build(base_poses=bases, include_floor=False)
    model = spec.compile()
    rows = np.tile(model.qpos0, (n, 1))
    for side in ["left", "right"]:
        ids = indices(model, side)
        rows[:, ids["arm_qadr"]] = old["qpos"][:, [old_names.index(name) for name in ids["arm_joint_names"]]]
    points = {side: ref[side + "_obs_keypoints"].reshape(n, 21, 3) for side in ["left", "right"]}
    assert all(np.isfinite(v).all() and abs(v).max() < 1e3 for v in points.values())
    if stage:
        stage("fixed_mount_and_bimanual_fingers")
    valid = {}
    for side in ["left", "right"]:
        z = points[side][:, 9] - points[side][:, 0]
        y = points[side][:, 5] - points[side][:, 13]
        valid[side] = (np.linalg.norm(z, axis=1) > 1e-7) & (np.linalg.norm(np.cross(y, z), axis=1) > 1e-9)
    # The mount is a model property. Fitting it to a video's human wrist puts
    # the palm off the axial couplers and can make it float or intersect link7.
    from .mount_contract import audit_connection
    from .panda_scene import mounting
    from .registry import selected as end_effector_name, profile
    from .task_mapping import solve_fixed_tool

    mounts = {s: mounting(end_effector_name(), s) for s in ["left", "right"]}
    (out / "fixed_mounts.json").write_text(json.dumps(mounts, indent=2))
    if finger_seed is not None:
        raise ValueError("Legacy finger-only cache cannot replace the final joint58 task solution")
    coupled, finger_report = solve_fixed_tool(
        model, rows, times, points, valid, seed_path, inp / "hand_reference.npz", out, stage
    )
    spec, model, rows = coupled["spec"], coupled["model"], coupled["qpos"]
    bases = coupled["bases"]
    np.savez_compressed(out / "finger_trajectory_pre_render.npz", qpos=rows, timestamp=times)
    connection_report = audit_connection(model, rows)
    (out / "rigid_connection_validation.json").write_text(json.dumps(connection_report, indent=2))
    assert not any(
        "finger_joint1" in model.joint(j).name or "finger_joint2" in model.joint(j).name for j in range(model.njnt)
    )
    xml = _standalone_xml(spec)
    world = xml.find("worldbody")
    focal = covering_focal(K, (H, W))
    body = ET.SubElement(world, "body", name="source_camera_mocap", mocap="true")
    ET.SubElement(body, "camera", name="original_ego_camera", fovy=str(np.degrees(2 * np.arctan(H / (2 * focal)))))
    (out / "scene.xml").write_text(ET.tostring(xml, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    data = mujoco.MjData(model)
    camera_id = model.camera("original_ego_camera").id
    mocap = int(model.body("source_camera_mocap").mocapid[0])
    hand_geom = []
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name or ""
        if name in ["left_link0", "right_link0"] or model.geom_group[g] == 3:
            model.geom_rgba[g, 3] = 0
        if "_hand_" in name and model.geom_rgba[g, 3] > 0:
            hand_geom.append(g)
        for side in ["left", "right"]:
            if not valid[side].any() and name.startswith(side + "_"):
                model.geom_rgba[g, 3] = 0.0
    bg_path = cache / f"04_inpaint/{episode}/bg.mp4"
    mask_path = cache / f"03_mask/{episode}/hand_arm_masks.npz"
    depth_path = cache / f"05_depth/{episode}/scene_depth.npz"
    background = cv2.VideoCapture(str(bg_path))
    source = cv2.VideoCapture(str(inp / "source.mp4")) if make_comparison else None
    human = np.load(mask_path)["masks"].astype(np.uint8)
    assert human.shape == (n, H, W)
    human = np.stack([cv2.dilate(m, np.ones((17, 17), np.uint8)) > 0 for m in human])
    use_depth = depth_mode == "depth-aware"
    if use_depth:
        depth, depth_meta = load_scene_depth(depth_path)
        assert depth.shape == (n, H, W), (depth.shape, (n, H, W), depth_meta)
        scale, shift, depth_fit = _estimate_scene_depth_alignment(
            depth, head, [(points[side].reshape(n, 63), np.ones(n, dtype=bool)) for side in ["left", "right"]], K
        )
        depth = depth * scale + shift
    else:
        depth_fit = dict(
            mode="alpha",
            reason="Use inherited Orientation alpha compositing to avoid eroding the larger articulated hand with the parallel-gripper depth heuristic.",
        )
    ss = 2
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W * ss)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H * ss)
    renderer = make_renderer(model, W * ss, H * ss)
    image_scale = np.diag([ss, ss, 1])
    homography = render_to_calibrated_pixels(image_scale @ K, (H * ss, W * ss), focal * ss)
    fps = float(manifest["fps"])

    def encoder(path, width):
        return subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{H}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    final = encoder(out / "robot_on_bg.mp4", W)
    comparison = encoder(out / "source_comparison.mp4", W * 2) if make_comparison else None
    selected = set(np.linspace(0, n - 1, 6).round().astype(int))
    sheets, masks, visibility, pixel_counts = [], [], [], []
    if stage:
        stage("original_camera_render_and_background_composite")
    try:
        for i, row in enumerate(rows):
            ok, bg = background.read()
            ok2, src = source.read() if source else (True, None)
            assert ok and ok2 and bg.shape == (H, W, 3)
            data.qpos[:] = row
            data.mocap_pos[mocap] = head[i, :3]
            R = Rotation.from_quat(head[i, [4, 5, 6, 3]]).as_matrix()
            data.mocap_quat[mocap] = mat_to_quat(R @ np.diag([1.0, -1.0, -1.0]))
            robot, mask, robot_depth = render_rgb_and_mask(model, data, renderer, camera_id, with_depth=use_depth)
            hand_mask = render_geom_subset_mask(model, data, renderer, camera_id, hand_geom) if use_depth else None
            robot = warp_camera(robot, homography)
            mask = warp_camera(mask, homography, True)
            if use_depth:
                robot_depth = warp_camera(robot_depth, homography, True)
                hand_mask = warp_camera(hand_mask, homography, True)
            robot = patch_arm_holes(tone_map_robot(cv2.medianBlur(robot, 3)), mask)
            robot = cv2.resize(robot, (W, H), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            if use_depth:
                hand_mask = cv2.resize(hand_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                robot_depth = cv2.resize(robot_depth, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
            alpha = feather_mask(mask, 5)
            if use_depth:
                stable = temporal_median_depth_frame(depth, i, 5)
                visible = depth_visibility_alpha(hand_mask, robot_depth, stable, margin=0.02, transition_width=0.04)
                visible[hand_mask & human[i]] = 1.0
                combined = np.ones((H, W), dtype=np.float32)
                combined[hand_mask] = visible[hand_mask]
                alpha *= combined[..., None]
            composed = (alpha * cv2.cvtColor(robot, cv2.COLOR_RGB2BGR) + (1 - alpha) * bg).astype(np.uint8)
            final.stdin.write(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB).tobytes())
            pair = np.concatenate([src, composed], axis=1) if comparison else composed
            if comparison:
                comparison.stdin.write(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB).tobytes())
            pixel_counts.append(int(mask.sum()))
            visibility.append(float(visible[hand_mask].mean()) if use_depth and hand_mask.any() else 1.0)
            masks.append(mask)
            if i in selected:
                sheets.append(cv2.resize(pair, (W, H)))
                cv2.imwrite(str(out / f"review_{i:03d}.jpg"), pair)
            if i % 40 == 0:
                print("SOURCE_COMPOSITE", i, n, flush=True)
        assert not background.read()[0]
        if source:
            assert not source.read()[0]
    finally:
        background.release()
        if source:
            source.release()
        renderer.close()
        final.stdin.close()
        assert final.wait() == 0
        if comparison:
            comparison.stdin.close()
            assert comparison.wait() == 0
    for filename in ["robot_on_bg.mp4"] + (["source_comparison.mp4"] if comparison else []):
        cap = cv2.VideoCapture(str(out / filename))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        assert count == n
    cv2.imwrite(str(out / "contact_sheet.jpg"), np.concatenate(sheets, axis=0))
    np.savez_compressed(out / "robot_masks.npz", masks=np.asarray(masks), source_frame_indices=np.arange(n))
    names = [model.joint(i).name for i in range(model.njnt)]
    state = rows.astype(np.float32)
    np.savez_compressed(
        out / "trajectory.npz",
        qpos=rows,
        state=state,
        state_names=np.asarray(names),
        timestamp=times,
        timestamps_ns=ref["timestamps_ns"],
        source_frame_indices=np.arange(n),
        camera_K=K,
        obs_head_pose=head,
        robot_type="panda_" + end_effector_name() + "_source_video",
    )
    report = dict(
        source_video_replacement_generated=True,
        frames=n,
        state_dim=int(state.shape[1]),
        orientation_cache=str(cache.resolve()),
        prepared_input=str(inp.resolve()),
        original_orientation_arm_trajectory_unchanged=False,
        unchanged_arm_dofs=0,
        arm_joint_count=14,
        rigid_arm_mapping=coupled["record"],
        rigid_connection=connection_report,
        hand_models=[s + " " + end_effector_name() for s in ["left", "right"]],
        finger_dofs=sum(profile(s)["physical_joint_count"] for s in ["left", "right"]),
        parallel_gripper_geoms_removed=True,
        same_source_camera_intrinsics=True,
        same_source_camera_head_poses=True,
        propainter_background_reused=True,
        inherited_depth_compositing=use_depth,
        compositing_mode=depth_mode,
        reused_finger_solution=bool(finger_seed),
        occlusion_scope="Alpha preserves complete robot geometry; object/front-back occlusion is approximate."
        if not use_depth
        else "Inherited gripper depth heuristic; visible hand erosion remains possible.",
        input_hashes={
            str(path): sha(path)
            for path in [old_path, inp / "hand_reference.npz", bg_path, mask_path] + ([depth_path] if use_depth else [])
        },
        source_episode=manifest["episode"],
        source_timestamps_s=times.tolist(),
        finger_ik=finger_report,
        fixed_mounts=mounts,
        depth_alignment=depth_fit,
        rendered_robot_pixel_min=int(min(pixel_counts)),
        rendered_robot_pixel_max=int(max(pixel_counts)),
        hand_depth_visible_fraction_mean=float(np.mean(visibility)),
        contact_quality_passed=False,
        coupled_physics_validated=False,
        training_validated=False,
        quality_passed=False,
        scope="Original Orientation background/camera with a model-defined rigid flange-to-hand mount. All registered mechanical joints track the original Orientation task TCP; human wrist is a soft reference; the mount is fixed. Finger contact/dynamics remain separate.",
        finished_unix=time.time(),
    )
    report.update(
        result_video_only=not make_comparison,
        comparison_generated=make_comparison,
        source_pipeline="JointPose-Orientation",
    )
    report["replacement_flow_passed"] = True
    (out / "replacement_report.json").write_text(json.dumps(report, indent=2))
    (out / "artifact_sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(out)): sha(p)
                for p in out.rglob("*")
                if p.is_file() and p.name not in ["artifact_sha256.json", "status.json"]
            },
            indent=2,
        )
    )
    return report
