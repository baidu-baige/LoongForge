# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo model construction, camera placement, rendering, and grippers."""

from pathlib import Path

import cv2
import numpy as np
import mujoco

try:
    from .. import config
    from ..robot_registry import (
        get_robot_spec,
        get_model_path,
        resolve_prefixed_ee_ref,
        resolve_robot_spec,
    )
except ImportError:
    import config
    from robot_registry import (
        get_robot_spec,
        get_model_path,
        resolve_prefixed_ee_ref,
        resolve_robot_spec,
    )

from .targets import _quat_to_mat

PANDA_DIR = Path(config.MENAGERIE_DIR) / "franka_emika_panda"  # compatibility alias
PANDA_XML = PANDA_DIR / "panda.xml"  # compatibility alias

# ============ Single-arm and dual-arm wrapper construction ============

DUAL_WRAPPER_TMPL = """<mujoco model="panda_dual">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast"/>
  <asset>
    <model name="robot" file="{xml_path}"/>
  </asset>
  <visual>
    <global offwidth="3200" offheight="2000"/>
    <headlight ambient="1.0 1.0 1.0" diffuse="1.0 1.0 1.0" specular="0.05 0.05 0.05"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.45 0.45 0.45"/>
    <light pos="0 -0.9 0.4" dir="0 1 -0.35" diffuse="0.3 0.3 0.3"/>
    <body name="left_mount" pos="{lp}" quat="{lq}">
      <attach model="robot" body="{base_body}" prefix="left_"/>
    </body>
    <body name="right_mount" pos="{rp}" quat="{rq}">
      <attach model="robot" body="{base_body}" prefix="right_"/>
    </body>
    <body name="ego_cam_body" mocap="true" pos="0 0 0">
      <camera name="ego" pos="0 0 0" quat="1 0 0 0" fovy="{fovy:.6f}"/>
    </body>
    <body name="wrist_l_m" mocap="true" pos="0 0 0">
      <camera name="wrist_l" pos="0 0 0" quat="1 0 0 0" fovy="{fovy:.6f}"/>
    </body>
    <body name="wrist_r_m" mocap="true" pos="0 0 0">
      <camera name="wrist_r" pos="0 0 0" quat="1 0 0 0" fovy="{fovy:.6f}"/>
    </body>
  </worldbody>
</mujoco>
"""

# OpenCV (x right, y down, z forward) -> MuJoCo camera (view -z, up +y): rotate 180 degrees around x.
RX180 = np.diag([1.0, -1.0, -1.0])

# Wrist cameras use the common parallel-gripper TCP frame shared by all
# morphologies.  The camera sits slightly behind the contact center and looks
# along TCP +z (MuJoCo cameras look along local -z).
WRIST_TCP_OFFSET = np.array([0.0, 0.015, -0.10])
WRIST_CAM_ROT = np.diag([-1.0, 1.0, -1.0])
# Preserve the calibrated Panda wrist-camera placement from the legacy
# pipeline.  Other morphologies use the common TCP-frame calibration above.
WRIST_LINK_OFFSET = np.array([0.000, 0.015, 0.050])
WRIST_CAM_QUAT_TO_MAT = np.array([[-1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, -1.0]])


# ============ Rendering helpers (copied inline from the original step4_g1_pipeline.py) ============


def patch_arm_holes(rgb, mask, dark_thresh=12):
    """Repair near-black holes inside the rendered robot mask."""
    gray = rgb.mean(axis=2)
    holes = (gray < dark_thresh) & (mask > 0)
    if not bool(np.any(holes)):
        return rgb
    inpaint_mask = np.zeros_like(gray, dtype=np.uint8)
    inpaint_mask[holes] = 255
    return cv2.inpaint(rgb, inpaint_mask, 2, cv2.INPAINT_TELEA)


def mat_to_quat(R):
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    return q


def make_renderer(model, width, height):
    """Create a MuJoCo renderer and restore the legacy default lighting."""
    model.vis.headlight.ambient[:] = [0.25, 0.25, 0.25]
    model.vis.headlight.diffuse[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.specular[:] = [0.03, 0.03, 0.03]
    return mujoco.Renderer(model, height=height, width=width)


def feather_mask(mask, ksize=5):
    """Feather a mask and normalize it to [0, 1]."""
    m = mask.astype(np.float32)
    if ksize > 1:
        m = cv2.GaussianBlur(m, (ksize | 1, ksize | 1), 0)
    return np.clip(m, 0.0, 1.0)[..., None]


def temporal_median_depth_frame(depth_sequence, frame_index, window=5):
    """Return one scene-depth frame after a short centered median.

    Monocular depth is inferred independently for every video frame. A median
    suppresses isolated jumps without averaging depth across object edges.
    """
    depth_sequence = np.asarray(depth_sequence)
    if depth_sequence.ndim != 3:
        raise ValueError(
            f"depth_sequence must have shape (N,H,W), got {depth_sequence.shape}")
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("depth temporal window must be a positive odd integer")
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= len(depth_sequence):
        raise IndexError(f"depth frame index out of range: {frame_index}")
    if window == 1:
        return depth_sequence[frame_index]
    radius = window // 2
    start = max(0, frame_index - radius)
    stop = min(len(depth_sequence), frame_index + radius + 1)
    return np.median(depth_sequence[start:stop], axis=0).astype(
        np.float32, copy=False)


def depth_visibility_alpha(mask, robot_depth, scene_depth, margin=0.02,
                           transition_width=0.04):
    """Compute robot visibility with a soft monocular-depth tolerance.

    The transition band prevents centimeter-scale scene-depth noise from
    toggling complete links between visible and hidden in adjacent frames.
    Invalid scene depth is unknown and must not be treated as an occluder.
    """
    mask = np.asarray(mask, dtype=bool)
    robot_depth = np.asarray(robot_depth, dtype=np.float32)
    scene_depth = np.asarray(scene_depth, dtype=np.float32)
    if robot_depth.shape != mask.shape or scene_depth.shape != mask.shape:
        raise ValueError(
            "mask, robot_depth, and scene_depth must have matching shapes")
    transition_width = float(transition_width)
    if not np.isfinite(transition_width) or transition_width < 0:
        raise ValueError("depth transition width must be finite and non-negative")

    alpha = np.zeros(mask.shape, dtype=np.float32)
    robot_valid = mask & np.isfinite(robot_depth) & (robot_depth > 0.0)
    scene_valid = np.isfinite(scene_depth) & (scene_depth > 0.0)
    alpha[robot_valid & ~scene_valid] = 1.0
    compare = robot_valid & scene_valid
    clearance = scene_depth[compare] - robot_depth[compare]
    if transition_width == 0.0:
        alpha[compare] = clearance > float(margin)
        return alpha

    lower = float(margin) - transition_width
    x = np.clip((clearance - lower) / (2.0 * transition_width), 0.0, 1.0)
    alpha[compare] = x * x * (3.0 - 2.0 * x)
    return alpha


def read_video_frames(path):
    """Read all MP4 frames into a list of BGR ndarrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def load_scene_depth(path):
    """Load metric scene depth and its image-alignment metadata."""
    with np.load(path, allow_pickle=False) as z:
        depth = z["depth"].astype(np.float32)
        metadata = {
            "height": int(z["height"]) if "height" in z else depth.shape[1],
            "width": int(z["width"]) if "width" in z else depth.shape[2],
            "crop_top": int(z["crop_top"]) if "crop_top" in z else 0,
            "crop_bottom": int(z["crop_bottom"]) if "crop_bottom" in z else 0,
        }
    if depth.ndim != 3:
        raise ValueError(f"scene depth must have shape (N,H,W), got {depth.shape}")
    return depth, metadata


def build_single_arm_model(spec=None, return_spec=False):
    """Load one registered morphology for base search and IK."""
    spec = spec or get_robot_spec("panda")
    model = mujoco.MjModel.from_xml_path(str(get_model_path(spec)))
    resolved = resolve_robot_spec(model, spec)
    return (model, resolved) if return_spec else model


def tone_map_robot(rgb):
    """Restore the legacy gamma lift used on robot renderings."""
    return (255.0 * (rgb.astype(np.float32) / 255.0) ** 0.55).astype(np.uint8)


def build_dual_model(left_pos, left_quat, right_pos, right_quat, fovy_deg,
                     spec=None):
    """Instantiate two copies of a registered morphology and an ego camera."""
    spec = spec or get_robot_spec("panda")
    xml = DUAL_WRAPPER_TMPL.format(
        xml_path=str(get_model_path(spec)),
        base_body=spec.base_body,
        lp=" ".join(f"{v:.6f}" for v in left_pos),
        lq=" ".join(f"{v:.6f}" for v in left_quat),
        rp=" ".join(f"{v:.6f}" for v in right_pos),
        rq=" ".join(f"{v:.6f}" for v in right_quat),
        fovy=fovy_deg,
    )
    # Base-pair screening runs concurrently in the multi-GPU launcher. A shared
    # morphology-named file in /tmp can be truncated by one worker while another
    # worker is parsing it, producing intermittent "ParseXML: empty file"
    # failures. The referenced robot asset path is absolute, so parsing this
    # small wrapper directly from memory is both sufficient and race-free.
    return mujoco.MjModel.from_xml_string(xml)


def hide_non_arm_geoms(model, spec=None, active_sides=None):
    """Hide base, bed, decorative, and all group-3 collision geometry.

    Keep bodies related to the arm and hand. The general policy retains link0 through
    link7, hand, and fingers, though link0 may be hidden for a cleaner view.
    """
    # Keep link1..link7, hand, and fingers; hide the bulky and visually distracting link0 base.
    spec = spec or get_robot_spec("panda")
    if active_sides is not None:
        active_sides = {str(side).lower() for side in active_sides}
    keep_kw = spec.visual_keywords
    hide_kw = spec.visual_hide_keywords
    hidden = 0
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        keep = any(kw in bname for kw in keep_kw)
        force_hide = any(kw in bname for kw in hide_kw)
        if active_sides is not None:
            if bname.startswith("left_") and "left" not in active_sides:
                force_hide = True
            elif bname.startswith("right_") and "right" not in active_sides:
                force_hide = True
        if not keep or force_hide or model.geom_group[gid] == 3:
            model.geom_rgba[gid, 3] = 0.0
            hidden += 1
    return hidden


def gripper_geom_ids(model, spec=None, active_sides=None):
    """Return visible geom IDs belonging to the end-effector/gripper.

    The boundary is inferred from the kinematic tree: each finger joint's body
    and descendants are included, together with the resolved end-effector body
    (the palm or gripper base) and its descendants. This avoids morphology-
    specific body-name heuristics and keeps the proximal arm out of the depth
    comparison.
    """
    spec = spec or get_robot_spec("panda")
    active = ({"left", "right"} if active_sides is None else
              {str(side).lower() for side in active_sides})
    gripper_bodies = set()

    for side in active:
        prefix = f"{side}_"
        roots = set()
        for joint_name in spec.finger_joints:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint_name)
            if joint_id >= 0:
                roots.add(int(model.jnt_bodyid[joint_id]))
        try:
            ref_kind, ref_id = resolve_prefixed_ee_ref(model, spec, prefix)
            roots.add(int(model.site_bodyid[ref_id])
                      if ref_kind == "site" else int(ref_id))
        except ValueError:
            pass
        for body_id in range(model.nbody):
            current = body_id
            while current > 0:
                if current in roots:
                    gripper_bodies.add(body_id)
                    break
                current = int(model.body_parentid[current])

    return np.asarray([
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in gripper_bodies and
        model.geom_rgba[geom_id, 3] > 0.0
    ], dtype=np.int32)


def render_geom_subset_mask(model, data, renderer, cam_id, geom_ids):
    """Render a binary mask for an explicit subset of visible geoms."""
    geom_ids = np.asarray(geom_ids, dtype=np.int32).reshape(-1)
    if geom_ids.size == 0:
        return np.zeros((renderer.height, renderer.width), dtype=bool)
    original_alpha = model.geom_rgba[:, 3].copy()
    try:
        keep = np.zeros(model.ngeom, dtype=bool)
        keep[geom_ids] = True
        model.geom_rgba[~keep, 3] = 0.0
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_id)
        depth = renderer.render().copy().astype(np.float32)
        far = float(model.vis.map.zfar * model.stat.extent)
        return np.isfinite(depth) & (depth > 0.0) & (depth < far * 0.999)
    finally:
        renderer.disable_depth_rendering()
        model.geom_rgba[:, 3] = original_alpha


def set_ego_camera(data, head_pose7):
    """Set the mocap ego camera position and orientation from world-frame head_pose7."""
    head_pos = head_pose7[:3]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(head_pose7[3:7], dtype=float))
    R_head = R.reshape(3, 3)
    R_mj = R_head @ RX180
    data.mocap_pos[0] = head_pos
    data.mocap_quat[0] = mat_to_quat(R_mj)


def set_wrist_cameras(data, model, spec, opening_widths=None):
    """Place both wrist cameras from the morphology's calibrated TCP frame."""
    widths = (None, None) if opening_widths is None else opening_widths
    for (prefix, mocap_body), opening_width in zip(
            (("left_", "wrist_l_m"), ("right_", "wrist_r_m")), widths):
        ref_kind, ref_id = resolve_prefixed_ee_ref(model, spec, prefix)
        if ref_kind == "site":
            ref_pos = data.site_xpos[ref_id]
            ref_rot = data.site_xmat[ref_id].reshape(3, 3)
            tcp_offset = np.asarray(spec.tcp_pos_site, dtype=float).copy()
            if opening_width is not None:
                tcp_offset += (
                    float(np.clip(opening_width, 0.0, spec.gripper_max)) *
                    np.asarray(spec.tcp_pos_site_width_gain, dtype=float)
                )
            tcp_pos = ref_pos + ref_rot @ tcp_offset
            side_rot = (spec.tcp_rot_site_left if prefix == "left_"
                        else spec.tcp_rot_site_right)
            if side_rot is None:
                side_rot = spec.tcp_rot_site
            tcp_rot = ref_rot @ _quat_to_mat(side_rot)
        else:
            ref_pos = data.xpos[ref_id]
            ref_rot = data.xmat[ref_id].reshape(3, 3)
            tcp_pos = ref_pos + ref_rot @ np.asarray(spec.tcp_pos_body, dtype=float)
            tcp_rot = ref_rot @ _quat_to_mat(spec.tcp_rot_body)

        if spec.name == "panda" and ref_kind == "body":
            # Exact legacy Panda placement: camera is mounted on the hand
            # body, not on the physical TCP contact center.
            cam_pos = ref_pos + ref_rot @ WRIST_LINK_OFFSET
            cam_rot = ref_rot @ WRIST_CAM_QUAT_TO_MAT
        else:
            cam_pos = tcp_pos + tcp_rot @ WRIST_TCP_OFFSET
            cam_rot = tcp_rot @ WRIST_CAM_ROT

        mocap_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, mocap_body)
        if mocap_body_id < 0:
            raise ValueError(f"wrist camera mocap body not found: {mocap_body}")
        mocap_id = int(model.body_mocapid[mocap_body_id])
        if mocap_id < 0:
            raise ValueError(f"body is not mocap-enabled: {mocap_body}")
        data.mocap_pos[mocap_id] = cam_pos
        data.mocap_quat[mocap_id] = mat_to_quat(cam_rot)


def render_rgb_and_mask(model, data, renderer, cam_id, with_depth=True):
    """Render ego RGB/mask and optionally camera-coordinate metric depth."""
    mujoco.mj_forward(model, data)
    renderer.disable_segmentation_rendering()
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()
    depth = None
    if with_depth:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_id)
        depth = renderer.render().copy().astype(np.float32)
        renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_id)
    try:
        seg = renderer.render()
        mask = (seg[:, :, 0] >= 0)
    except (IndexError, ValueError, RuntimeError):
        # MuJoCo 3.11 can return stale ID-color pixels for models whose
        # attached XML contains non-contiguous geom IDs.  The scene contains
        # only the target robot, so the metric depth pass provides a robust
        # fallback: pixels at the camera far plane are background.
        if depth is None:
            # Alpha mode skips the normal depth pass, but still needs this
            # lazy fallback when segmentation IDs are invalid.
            renderer.disable_segmentation_rendering()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=cam_id)
            depth = renderer.render().copy().astype(np.float32)
            renderer.disable_depth_rendering()
        far = float(renderer._model.vis.map.zfar * renderer._model.stat.extent)
        mask = np.isfinite(depth) & (depth > 0.0) & (depth < far * 0.999)
        # ``Renderer.render`` raises before restoring these flags on the
        # indexing error; clear them so the next RGB pass is not ID-colored.
        renderer._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        renderer._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
    renderer.disable_segmentation_rendering()
    return rgb, mask, depth


def gripper_width_to_qpos(w, spec):
    """Map human total opening to the registered robot gripper joints."""
    half = float(np.clip(w, 0.0, spec.gripper_max) * 0.5)
    if spec.gripper_mode == "xarm_driver":
        q = float(np.clip(w / max(spec.gripper_max, 1e-8) * 0.85, 0.0, 0.85))
        return (q, q)
    if spec.gripper_mode == "robotiq_driver":
        # Robotiq 2F-85 uses a single-sided [0, 0.8] driver range, but its
        # direction is opposite to the human opening width: q=0 is open and
        # q=0.8 is closed.  Invert the normalized width before mapping it to
        # the driver angle.  The remaining joints are the 2F-85 four-bar
        # linkage, rather than independent finger joints.  The MuJoCo asset
        # closes that linkage with a connect constraint; provide a compatible
        # kinematic state here because qpos is rendered directly without a
        # dynamics step.
        normalized = float(np.clip(w / max(spec.gripper_max, 1e-8), 0.0, 1.0))
        q = float((1.0 - normalized) * 0.8)
        # Order: right driver/coupler/spring/follower, then left equivalent.
        # The follower ratio is the measured 2F-85 linkage ratio over the
        # usable driver range and keeps the connect residual small.
        follower = -0.96 * q
        return (q, 0.0, q, follower, q, 0.0, q, follower)
    if spec.gripper_mode == "franka_hand":
        # Franka Hand has two mirrored prismatic joints, each moving up to
        # 40 mm.  The requested width is the total opening between fingers.
        q = float(np.clip(w * 0.5, 0.0, 0.04))
        return (q, q)
    if spec.gripper_mode == "sawyer_electric":
        # Native Sawyer Electric Gripper uses mirrored prismatic joints with
        # 20.833 mm travel per finger (Intera Xacro, standard_narrow fingers).
        q = float(np.clip(
            w / max(spec.gripper_max, 1e-8) * 0.020833,
            0.0, 0.020833,
        ))
        return (q, -q)
    if spec.gripper_mode == "jaco":
        # The Jaco hand has three coupled hinge joints, with a 0.15..1.35 rad range.
        q = float(np.clip(
            1.35 - w / max(spec.gripper_max, 1e-8) * 1.2,
            0.15, 1.35,
        ))
        return (q, q, q)
    if spec.gripper_mode == "viper_asymmetric":
        # Front pad-center gap is 2*q - 25 mm.  Map to that physical gap,
        # instead of interpolating the rail joint range directly.
        q = float(np.clip(
            (float(np.clip(w, 0.0, spec.gripper_max)) + 0.025) * 0.5,
            0.021, 0.057,
        ))
        return (q, -q)
    if spec.gripper_mode == "widowx_asymmetric":
        # Front contact-marker gap is 2*q - 18 mm.
        q = float(np.clip(
            (float(np.clip(w, 0.0, spec.gripper_max)) + 0.018) * 0.5,
            0.015, 0.037,
        ))
        return (q, -q)
    if spec.name == "arx_l5":
        # ARX front pad-center gap is 3.8 mm + 2*q.
        q = float(np.clip(
            (float(np.clip(w, 0.0, spec.gripper_max)) - 0.0038) * 0.5,
            0.0, 0.044,
        ))
        return (q, -q)
    if spec.name == "piper":
        # Piper front pad-center gap is 5 mm + 2*q.
        q = float(np.clip(
            (float(np.clip(w, 0.0, spec.gripper_max)) - 0.005) * 0.5,
            0.0, 0.035,
        ))
        return (q, -q)
    if spec.gripper_mode == "aloha":
        # The visual finger meshes overlap by 15.7 mm at q=0; their actual
        # inner-edge opening therefore follows gap = 2*q - 15.7 mm.  The
        # named finger sites have a different 1.8 mm zero gap and are useful
        # for IK only, but using their separation here makes the rendered
        # gripper visibly too narrow relative to the human fingertips.
        visual_inner_overlap = 0.0157
        q = float(np.clip(
            (float(np.clip(w, 0.0, spec.gripper_max)) + visual_inner_overlap) * 0.5,
            0.0, 0.041,
        ))
        return (q, q)
    if spec.gripper_mode == "so101":
        # The upstream README notes that LeRobot's linear opening is not yet
        # mapped in the MJCF. Measurements from the official visual meshes
        # give this monotonic fingertip-gap calibration. Beyond about 0.8 rad
        # the linkage folds back and the visual gap decreases, so never use
        # the complete 1.745 rad joint range as a linear opening range.
        gaps = np.array([0.0089, 0.0221, 0.0405, 0.0601, 0.0778, 0.0800])
        angles = np.array([-0.174533, 0.0, 0.25, 0.50, 0.75, 0.781])
        target_gap = float(np.clip(w, gaps[0], min(spec.gripper_max, gaps[-1])))
        q = float(np.interp(target_gap, gaps, angles))
        return (q,)
    if spec.name == "yam":
        return (half, -half)
    return (half, half)


def set_gripper_qpos(qpos, qadr, spec, width):
    """Set gripper qpos, including coupled joints for composite grippers."""
    values = gripper_width_to_qpos(width, spec)
    for i, qa in enumerate(qadr):
        if i < len(values):
            qpos[qa] = values[i]
    if spec.gripper_all_joints:
        for qa in qadr[len(values):]:
            qpos[qa] = values[0]


def name_to_dof(model, names, prefix=""):
    """Look up joint IDs, qpos addresses, and DOF addresses by joint name."""
    ids, qadr, vadr = [], [], []
    for nm in names:
        full = f"{prefix}{nm}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
        assert jid >= 0, f"joint not found: {full}"
        ids.append(jid)
        qadr.append(int(model.jnt_qposadr[jid]))
        vadr.append(int(model.jnt_dofadr[jid]))
    return ids, qadr, vadr
