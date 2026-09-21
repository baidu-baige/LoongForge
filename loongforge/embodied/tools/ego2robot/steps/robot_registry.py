# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Robot morphology registry used by the retargeting and dataset steps.

The registry deliberately describes the kinematic interface used by this
pipeline rather than every detail of a robot XML.  Models with no integrated
parallel gripper are not registered as supported render targets until a
gripper asset and TCP calibration are provided.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from . import config
except ImportError:
    import config


@dataclass(frozen=True)
class RobotSpec:
    """Describe the kinematic and asset configuration for a robot."""

    name: str
    model_dir: str
    xml_name: str
    base_body: str
    arm_joints: tuple[str, ...]
    finger_joints: tuple[str, ...]
    ee_body: str | None
    ee_site: str | None
    reach: float
    gripper_max: float
    tcp_pos_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_rot_body: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    # Offset from a named TCP site to the physical finger contact center,
    # expressed in the site frame.
    tcp_pos_site: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Linear motion of the physical contact center per metre of total gripper
    # opening. This is non-zero for asymmetric, single-moving-jaw grippers.
    tcp_pos_site_width_gain: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Fixed rotation from the named site frame to the common TCP frame,
    # represented as a wxyz quaternion.
    tcp_rot_site: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    # Optional side-specific site calibrations for mirrored dual-arm models.
    tcp_rot_site_left: tuple[float, ...] | None = None
    tcp_rot_site_right: tuple[float, ...] | None = None
    # Per-axis orientation cost in the IK reference frame.  A zero releases
    # that rotational axis for underactuated morphologies.
    # Qwen-RobotManip's control loop is position-first (orientation_cost=0.05).
    # Keeping orientation soft avoids forcing a redundant arm through a new IK
    # branch merely to chase noisy human wrist rotations.
    ik_orientation_cost: tuple[float, float, float] = (0.05, 0.05, 0.05)
    # Optional single-view orientation constraint. The selected local EEF axis
    # is required to lie in the camera plane that projects to the observed 2D
    # hand-opening line; its unobservable depth component remains free.
    ik_projection_axis: tuple[float, float, float] | None = None
    ik_projection_cost: float = 0.0
    # Optional second image-space axis. This is useful for a 5-DoF arm where
    # the jaw-opening line alone still leaves the gripper approach visibly
    # underconstrained.
    ik_secondary_projection_axis: tuple[float, float, float] | None = None
    ik_secondary_projection_cost: float = 0.0
    # Underactuated arms cannot always satisfy position and all enabled
    # orientation axes simultaneously. Prefer a continuous position-priority
    # fallback over emitting an off-target pose or interpolating IK branches.
    ik_position_priority: bool = False
    ik_branch_jump_threshold: float = 0.0
    # Desktop arms must remain upright and share one mounting plane.  Larger
    # morphologies retain the paper's unconstrained independent base search.
    base_orientation_mode: str = "free"
    coplanar_bases: bool = False
    aligned_base_depths: bool = False
    # Optional morphology-specific offsets along the head viewing direction,
    # normalized by arm reach. Desktop arms extend from behind the workspace.
    base_forward_offsets: tuple[float, ...] | None = None
    # Optional absolute geometric reach gate used only by base-search
    # prefiltering. Candidate-grid spacing and reach-ratio scoring still use
    # ``reach`` so this can correct a conservative gate without moving the grid.
    base_max_target_distance: float | None = None
    # Estimate a static horizontal mounting surface from metric scene depth and
    # constrain base-search candidates to contact it. Disabled unless a
    # morphology explicitly opts in.
    scene_support_surface: bool = False
    finger_bodies: tuple[str, ...] = ()
    gripper_mode: str = "symmetric"
    gripper_all_joints: tuple[str, ...] = ()
    visual_keywords: tuple[str, ...] = ("link", "hand", "finger", "gripper", "knuckle")
    visual_hide_keywords: tuple[str, ...] = ("link0",)
    gripper_model_dir: str | None = None
    gripper_xml_name: str | None = None
    gripper_attach_body: str | None = None
    gripper_mount_body: str = "base_mount"
    gripper_attach_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_attach_quat: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    model_path_override: str | None = None
    gripper_model_path_override: str | None = None

    @property
    def model_path(self) -> Path:
        """Return the path to the robot arm MJCF model."""
        if self.model_path_override:
            return Path(self.model_path_override)
        return Path(config.MENAGERIE_DIR) / self.model_dir / self.xml_name

    @property
    def gripper_model_path(self) -> Path | None:
        """Return the optional path to the gripper MJCF model."""
        if self.gripper_model_path_override:
            return Path(self.gripper_model_path_override)
        if not self.gripper_model_dir or not self.gripper_xml_name:
            return None
        return Path(config.MENAGERIE_DIR) / self.gripper_model_dir / self.gripper_xml_name

    @property
    def state_names(self) -> tuple[str, ...]:
        """Return arm and finger joint names in state-vector order."""
        return tuple(self.arm_joints) + tuple(self.finger_joints)

    @property
    def state_dim(self) -> int:
        """Return the dimension of the position-and-velocity state vector."""
        return 2 * len(self.state_names)


ROBOT_SPECS: dict[str, RobotSpec] = {
    "panda": RobotSpec(
        name="panda",
        model_dir="franka_emika_panda",
        xml_name="panda.xml",
        base_body="link0",
        arm_joints=tuple(f"joint{i}" for i in range(1, 8)),
        finger_joints=("finger_joint1", "finger_joint2"),
        ee_body="hand",
        ee_site=None,
        reach=1.272,
        gripper_max=0.08,
        tcp_pos_body=(0.0, 0.0, 0.1034),
        visual_hide_keywords=("link0",),
    ),
    "xarm7": RobotSpec(
        name="xarm7",
        model_dir="ufactory_xarm7",
        xml_name="xarm7.xml",
        base_body="link_base",
        arm_joints=tuple(f"joint{i}" for i in range(1, 8)),
        finger_joints=("left_driver_joint", "right_driver_joint"),
        ee_body=None,
        ee_site="link_tcp",
        reach=1.290,
        gripper_max=0.085,
        gripper_mode="xarm_driver",
        gripper_all_joints=(
            "left_driver_joint", "left_finger_joint", "left_inner_knuckle_joint",
            "right_driver_joint", "right_finger_joint", "right_inner_knuckle_joint",
        ),
        visual_hide_keywords=("link_base",),
    ),
    "arx_l5": RobotSpec(
        name="arx_l5",
        model_dir="arx_l5",
        xml_name="arx_l5.xml",
        base_body="base_link",
        arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
        finger_joints=("joint7", "joint8"),
        ee_body="link6",
        ee_site=None,
        reach=0.855,
        gripper_max=0.088,
        # link7/link8 origins are 86.57 mm in front of link6.  The ARX
        # fingertip collision pads are centered another 50 mm forward, so
        # their pinch center is at x=0.13657 m in the link6 frame.  Do not
        # auto-calibrate from the finger body origins.
        tcp_pos_body=(0.13657, 0.0, -0.00024363),
        # link6 local x is approach and local y is opening.  Map common TCP
        # z -> body x and common TCP y -> body y; the remaining axis is chosen
        # to keep a proper right-handed rotation.
        tcp_rot_body=(0.70710678, 0.0, 0.70710678, 0.0),
        visual_hide_keywords=("base_link",),
    ),
    "piper": RobotSpec(
        name="piper",
        model_dir="agilex_piper",
        xml_name="piper.xml",
        base_body="base_link",
        arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
        finger_joints=("joint7", "joint8"),
        ee_body="link6",
        ee_site=None,
        reach=0.883,
        gripper_max=0.070,
        # link7/link8 origins are at z=0.13503 in link6, but the distal
        # fingertip collision pads are centered at z=0.12003.  The latter is
        # the physical pinch plane; do not use the finger body origins as the
        # TCP.
        tcp_pos_body=(0.0, 0.0, 0.12003),
        # The physical opening direction is link6 local y.  The TCP point
        # itself is on link6 local z, so no additional rotation is needed.
        tcp_rot_body=(1.0, 0.0, 0.0, 0.0),
        visual_hide_keywords=("base_link",),
    ),
    "yam": RobotSpec(
        name="yam",
        model_dir="i2rt_yam",
        xml_name="yam.xml",
        base_body="arm",
        arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
        finger_joints=("left_finger", "right_finger"),
        ee_body=None,
        ee_site="grasp_site",
        reach=0.866,
        gripper_max=0.075,
        visual_hide_keywords=("arm",),
    ),
    # These menagerie arm XMLs are bare arms.  They are composed at runtime
    # with compatible end-effector assets.  FR3 uses the native Franka Hand
    # description; Sawyer uses its native Intera Electric Gripper model below.
    "fr3": RobotSpec(
        name="fr3", model_dir="franka_fr3", xml_name="fr3.xml",
        base_body="base", arm_joints=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        finger_joints=("grip_finger_joint1", "grip_finger_joint2"),
        ee_body="grip_hand", ee_site=None, reach=1.272, gripper_max=0.080,
        tcp_pos_body=(0.0, 0.0, 0.1034),
        gripper_model_path_override=str(
            Path(config.MENAGERIE_DIR) / "franka_emika_panda/hand.xml"
        ),
        gripper_attach_body="fr3_link7",
        gripper_mode="franka_hand",
        gripper_attach_pos=(0.0, 0.0, 0.107),
        gripper_mount_body="hand",
        # The composed native hand bodies are prefixed with ``grip_``.
        visual_keywords=("fr3_link", "grip"),
        visual_hide_keywords=("base",),
    ),
    "ur5e": RobotSpec(
        name="ur5e", model_dir="universal_robots_ur5e", xml_name="ur5e.xml",
        base_body="base",
        arm_joints=("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        finger_joints=("grip_left_driver_joint", "grip_right_driver_joint"),
        ee_body=None, ee_site="grip_pinch", reach=1.236, gripper_max=0.085,
        gripper_model_dir="robotiq_2f85", gripper_xml_name="2f85.xml",
        gripper_attach_body="wrist_3_link",
        gripper_mode="robotiq_driver",
        gripper_all_joints=(
            "grip_right_driver_joint", "grip_right_coupler_joint",
            "grip_right_spring_link_joint", "grip_right_follower_joint",
            "grip_left_driver_joint", "grip_left_coupler_joint",
            "grip_left_spring_link_joint", "grip_left_follower_joint",
        ),
        gripper_attach_pos=(0.0, 0.1, 0.0),
        gripper_attach_quat=(-1.0, 1.0, 0.0, 0.0),
        # The Robotiq mount and base are prefixed as ``grip_base_mount`` and
        # ``grip_base``.  Keep them visible so the UR flange-to-gripper
        # connection is rendered; the unprefixed arm ``base`` remains hidden
        # because it is outside this visual whitelist.
        visual_keywords=("link", "grip"),
        visual_hide_keywords=(),
    ),
    "ur10e": RobotSpec(
        name="ur10e", model_dir="universal_robots_ur10e", xml_name="ur10e.xml",
        base_body="base",
        arm_joints=("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        finger_joints=("grip_left_driver_joint", "grip_right_driver_joint"),
        ee_body=None, ee_site="grip_pinch", reach=1.627, gripper_max=0.085,
        gripper_model_dir="robotiq_2f85", gripper_xml_name="2f85.xml",
        gripper_attach_body="wrist_3_link",
        gripper_mode="robotiq_driver",
        gripper_all_joints=(
            "grip_right_driver_joint", "grip_right_coupler_joint",
            "grip_right_spring_link_joint", "grip_right_follower_joint",
            "grip_left_driver_joint", "grip_left_coupler_joint",
            "grip_left_spring_link_joint", "grip_left_follower_joint",
        ),
        gripper_attach_pos=(0.0, 0.1, 0.0),
        gripper_attach_quat=(-1.0, 1.0, 0.0, 0.0),
        # Keep the prefixed Robotiq base/mount visible as the physical
        # connection between the UR tool flange and the gripper.
        visual_keywords=("link", "grip"),
        visual_hide_keywords=(),
    ),
    "kinova_gen3": RobotSpec(
        name="kinova_gen3", model_dir="kinova_gen3", xml_name="gen3.xml",
        base_body="base_link", arm_joints=tuple(f"joint_{i}" for i in range(1, 8)),
        finger_joints=("grip_left_driver_joint", "grip_right_driver_joint"),
        ee_body=None, ee_site="grip_pinch", reach=1.337, gripper_max=0.085,
        gripper_model_dir="robotiq_2f85", gripper_xml_name="2f85.xml",
        gripper_attach_body="bracelet_link",
        gripper_mode="robotiq_driver",
        gripper_all_joints=(
            "grip_right_driver_joint", "grip_right_coupler_joint",
            "grip_right_spring_link_joint", "grip_right_follower_joint",
            "grip_left_driver_joint", "grip_left_coupler_joint",
            "grip_left_spring_link_joint", "grip_left_follower_joint",
        ),
        gripper_mount_body="base",
        gripper_attach_pos=(0.0, 0.0, -0.06149039),
        gripper_attach_quat=(0.0, -1.0, 1.0, 0.0),
        # The composed Robotiq bodies are prefixed with ``grip_``.  Keeping
        # this keyword makes the base and linkage visible together with the
        # Kinova arm instead of leaving only the parts whose names contain
        # ``link`` visible.
        visual_keywords=("link", "grip"),
        visual_hide_keywords=("base_link",),
    ),
    "sawyer": RobotSpec(
        name="sawyer", model_dir="rethink_robotics_sawyer", xml_name="sawyer.xml",
        base_body="base",
        arm_joints=tuple(f"right_j{i}" for i in range(7)),
        finger_joints=("grip_l_finger_joint", "grip_r_finger_joint"),
        ee_body=None, ee_site="grip_pinch", reach=1.420, gripper_max=0.079,
        gripper_model_path_override=str(
            Path(__file__).resolve().parents[1]
            / "models/sawyer_gripper/sawyer_electric_gripper.xml"
        ),
        gripper_attach_body="right_l6",
        gripper_mode="sawyer_electric",
        gripper_attach_pos=(0.0, 0.0, 0.0195),
        gripper_attach_quat=(0.70710678, 0.0, 0.0, 0.70710678),
        # Sawyer names its arm bodies right_l0...right_l6 rather than link0...
        # link7; keep both the arm chain and the composed grip_* bodies visible.
        visual_keywords=("right_l", "grip"),
        # Do not force-hide ``grip_base_mount`` or
        # ``grip_electric_gripper_base``: they are the native connector and
        # gripper body.  The unprefixed Sawyer arm ``base`` remains hidden
        # because it is outside the visual whitelist.
        visual_hide_keywords=(),
    ),
    "iiwa": RobotSpec(
        name="iiwa", model_dir="kuka_iiwa_14", xml_name="iiwa14.xml",
        base_body="base", arm_joints=tuple(f"joint{i}" for i in range(1, 8)),
        finger_joints=("grip_left_driver_joint", "grip_right_driver_joint"),
        ee_body=None, ee_site="grip_pinch", reach=1.411, gripper_max=0.085,
        gripper_model_dir="robotiq_2f85", gripper_xml_name="2f85.xml",
        gripper_attach_body="link7",
        gripper_mode="robotiq_driver",
        gripper_all_joints=(
            "grip_right_driver_joint", "grip_right_coupler_joint",
            "grip_right_spring_link_joint", "grip_right_follower_joint",
            "grip_left_driver_joint", "grip_left_coupler_joint",
            "grip_left_spring_link_joint", "grip_left_follower_joint",
        ),
        gripper_attach_pos=(0.0, 0.0, 0.045),
        # ``base`` is the iiwa arm base, but it is also part of the
        # prefixed Robotiq names (``grip_base``/``grip_base_mount``).  Do not
        # use it as a force-hide keyword here: the arm base is already
        # excluded by the keep list while the gripper base must remain visible
        # as the flange-to-gripper connection.
        visual_keywords=("link", "grip"),
        visual_hide_keywords=(),
    ),
    "jaco": RobotSpec(
        name="jaco", model_dir="", xml_name="",
        model_path_override=str(Path(__file__).resolve().parents[1] / "models/jaco/jaco_arm.xml"),
        gripper_model_path_override=str(Path(__file__).resolve().parents[1] / "models/jaco/jaco_hand.xml"),
        base_body="b_base",
        arm_joints=tuple(f"joint_{i}" for i in range(1, 7)),
        finger_joints=("grip_finger_1", "grip_finger_2", "grip_finger_3"),
        ee_body=None, ee_site="grip_pinchsite", reach=1.200, gripper_max=0.125,
        gripper_mode="jaco",
        gripper_all_joints=("grip_finger_1", "grip_finger_2", "grip_finger_3"),
        gripper_attach_body="b_6", gripper_mount_body="hand",
        gripper_attach_quat=(0.0, 0.70710678, 0.70710678, 0.0),
        # Jaco arm bodies are named b_1...b_6, while the composed hand bodies
        # are prefixed with grip_.  Keep the arm chain and three-finger hand
        # visible, but omit the fixed b_base pedestal.
        visual_keywords=("b_", "grip"),
        visual_hide_keywords=("b_base",),
    ),
    "viperx": RobotSpec(
        name="viperx", model_dir="trossen_vx300s", xml_name="vx300s.xml",
        base_body="base_link",
        arm_joints=("waist", "shoulder", "elbow", "forearm_roll",
                    "wrist_angle", "wrist_rotate"),
        finger_joints=("left_finger", "right_finger"),
        ee_body=None, ee_site="pinch", reach=0.948, gripper_max=0.114,
        # The named pinch site is 25.8 mm behind the front pad centers.
        tcp_pos_site=(0.0258, 0.0, 0.0),
        gripper_mode="viper_asymmetric", visual_hide_keywords=("base_link",),
    ),
    "widowx": RobotSpec(
        name="widowx", model_dir="trossen_wx250s", xml_name="wx250s.xml",
        base_body="wx250s/base_link",
        arm_joints=("waist", "shoulder", "elbow", "forearm_roll",
                    "wrist_angle", "wrist_rotate"),
        finger_joints=("left_finger", "right_finger"),
        ee_body="wx250s/gripper_link", ee_site=None, reach=0.700, gripper_max=0.074,
        # The finger-link origins are 66 mm in front of gripper_link, while
        # the actual fingertip/contact geometry is another 42 mm forward
        # (the XML fingertip collision markers are at x=0.108 m).  Use the
        # physical pinch center explicitly instead of auto-calibrating from
        # the finger body origins.
        tcp_pos_body=(0.108, 0.0, 0.0),
        # gripper_link local x is approach and local y is opening.  Map common
        # TCP z -> body x and common TCP y -> body y.
        tcp_rot_body=(0.70710678, 0.0, 0.70710678, 0.0),
        gripper_mode="widowx_asymmetric", visual_hide_keywords=("wx250s/base_link",),
    ),
    "aloha_agilex": RobotSpec(
        name="aloha_agilex", model_dir="aloha", xml_name="aloha.xml",
        base_body="left/base_link",
        arm_joints=("left/waist", "left/shoulder", "left/elbow", "left/forearm_roll",
                    "left/wrist_angle", "left/wrist_rotate"),
        finger_joints=("left/left_finger", "left/right_finger"),
        ee_body=None, ee_site="left/gripper", reach=0.853, gripper_max=0.082,
        # The site is behind and slightly below the physical pad-center.
        tcp_pos_site=(0.01537, 0.0, 0.00425),
        # ALOHA site axes are x=approach, y=opening, z=gripper-normal. In the
        # common TCP frame, +z maps to site +x and the signed opening axis +y
        # maps to site -y. The opening sign only selects an equivalent jaw
        # direction; retaining site +x as approach is essential for IK. Both
        # rendered arms are copies of this native left arm and use this same
        # site-to-TCP calibration.
        tcp_rot_site=(0.0, 0.70710678, 0.0, 0.70710678),
        # aloha_agilex is a full 6-DOF arm, so it can track the gripper
        # orientation as well as the pinch position. The RobotSpec default of
        # 0.05 is tuned for underactuated arms (e.g. so_arm101); against the
        # 10.0 position cost it makes the QP effectively ignore orientation,
        # leaving a ~40 deg wrist residual on both arms. Weighting orientation
        # at 0.3 halves that residual (rot ~14-21 deg) while keeping position
        # perfect (both_position_ok=100%, ~1.5 mm). Pushing it to 1.0 lets the
        # left arm reach ~8 deg but drives the duplicated right arm into a bad
        # local minimum (rot ~70 deg, position sacrificed), so 0.3 is the
        # balance that improves both arms without regressing the right.
        ik_orientation_cost=(0.3, 0.3, 0.3),
        gripper_mode="aloha",
        # Show the native ViperX base pedestal (vx300s_1_base) on both arms so
        # the rendered morphology matches the real dual-arm ALOHA instead of a
        # pair of floating arms. Only the visual base geom (group 2) is kept;
        # the base collision geom (group 3) is still hidden by hide_non_arm_geoms.
        visual_hide_keywords=(),
    ),
    # Official SO-ARM101 MuJoCo model downloaded from
    # TheRobotStudio/SO-ARM100.  The ``new_calib`` model uses the calibrated
    # virtual joint zeros shipped by the upstream project.
    "so_arm101": RobotSpec(
        name="so_arm101", model_dir="", xml_name="",
        model_path_override=str(
            Path(__file__).resolve().parents[1]
            / "models/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
        ),
        base_body="base",
        arm_joints=("shoulder_pan", "shoulder_lift", "elbow_flex",
                    "wrist_flex", "wrist_roll"),
        finger_joints=("gripper",),
        ee_body=None, ee_site="gripperframe",
        reach=0.420, gripper_max=0.080,
        # Official gripperframe axes are x=approach and z=jaw opening.  The
        # common TCP convention is z=approach and y=opening.
        # Site x is the gripper approach axis and site z is the jaw-opening
        # axis.  The common TCP uses z for approach and y for opening, so the
        # site-to-TCP rotation must map x -> TCP z and z -> TCP y.
        tcp_rot_site=(0.5, 0.5, 0.5, 0.5),
        # The fixed jaw inner edge is at site z=-6.98 mm. Since only the other
        # jaw moves, the physical two-jaw midpoint shifts by half the opening.
        tcp_pos_site=(0.0, 0.0, -0.00698),
        tcp_pos_site_width_gain=(0.0, 0.0, 0.5),
        # Fallback for episodes without 2D observations: align the site z
        # opening axis while leaving rotation about it free. Path B uses the
        # projection-plane task below, which does not invent axis depth.
        ik_orientation_cost=(0.2, 0.2, 0.0),
        ik_projection_axis=(0.0, 0.0, 1.0),
        ik_projection_cost=0.3,
        ik_secondary_projection_axis=(1.0, 0.0, 0.0),
        ik_secondary_projection_cost=0.1,
        ik_position_priority=True,
        ik_branch_jump_threshold=0.45,
        base_orientation_mode="upright",
        coplanar_bases=False,
        aligned_base_depths=False,
        # Keep the physical tabletop orientation, but let each short arm
        # choose its own height/depth and search on either side of the hand
        # trajectory instead of forcing both mounts behind it.
        base_forward_offsets=(-0.9, -0.7, -0.5, -0.3, -0.1, 0.0,
                              0.1, 0.3, 0.5, 0.7, 0.9),
        # The model reaches about 0.546 m from base to gripperframe. Keep a
        # small geometric margin without applying the generic
        # 0.9 * 0.42 = 0.378 m gate. Scene-supported candidates in the sample
        # set require up to 0.519 m before the actual IK feasibility check.
        base_max_target_distance=0.53,
        scene_support_surface=True,
        gripper_mode="so101",
        visual_keywords=("base", "shoulder", "upper_arm", "lower_arm",
                         "wrist", "gripper", "moving_jaw"),
        visual_hide_keywords=(),
    ),
}


def get_robot_spec(name: str) -> RobotSpec:
    """Resolve a robot name or alias to a validated robot specification."""
    key = name.strip().lower()
    aliases = {
        "franka_panda": "panda", "franka": "panda", "xarm": "xarm7",
        "so-arm101": "so_arm101", "so-arm-101": "so_arm101",
        "so_arm-101": "so_arm101",
        "soarm101": "so_arm101", "so101": "so_arm101",
    }
    key = aliases.get(key, key)
    if key not in ROBOT_SPECS:
        supported = ", ".join(sorted(ROBOT_SPECS))
        raise ValueError(f"unsupported robot_type={name!r}; supported: {supported}")
    spec = ROBOT_SPECS[key]
    if not spec.model_path.is_file():
        raise FileNotFoundError(
            f"robot model for {key!r} was not found: {spec.model_path}. "
            "Set EGO2ROBOT_MENAGERIE_DIR to a populated mujoco_menagerie cache."
        )
    if spec.gripper_model_path is not None and not spec.gripper_model_path.is_file():
        raise FileNotFoundError(
            f"gripper model for {key!r} was not found: {spec.gripper_model_path}."
        )
    return spec


def _expand_mjcf_includes(root, source_dir):
    """Inline local MJCF includes so generated XML remains self-contained."""
    for parent in list(root.iter()):
        for include in list(parent.findall("include")):
            include_path = Path(include.get("file", ""))
            if not include_path.is_absolute():
                include_path = source_dir / include_path
            if not include_path.is_file():
                raise FileNotFoundError(
                    f"MJCF include {include.get('file')!r} was not found "
                    f"relative to {source_dir}"
                )
            included_root = ET.parse(include_path).getroot()
            _expand_mjcf_includes(included_root, include_path.parent)
            insert_at = list(parent).index(include)
            parent.remove(include)
            for child in list(included_root):
                parent.insert(insert_at, child)
                insert_at += 1


def _make_aloha_single_model(spec: RobotSpec) -> Path:
    """Build a base-at-origin single-arm view from the dual-arm ALOHA MJCF."""
    source = spec.model_path
    cache_key = repr(("aloha-single-v2", source.stat().st_mtime_ns)).encode()
    digest = hashlib.sha1(cache_key).hexdigest()[:12]
    out = Path("/tmp") / f"ego2robot_aloha_agilex_{digest}.xml"
    if out.exists():
        return out

    root = ET.parse(source).getroot()
    _expand_mjcf_includes(root, source.parent)
    for elem in root.iter():
        for attr in ("file",):
            filename = elem.get(attr)
            if not filename or Path(filename).is_absolute():
                continue
            elem.set(attr, str(source.parent / "assets" / filename))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"ALOHA model has no worldbody: {source}")

    left_base = next((body for body in worldbody.findall("body")
                      if body.get("name") == "left/base_link"), None)
    if left_base is None:
        raise ValueError("ALOHA model has no left/base_link body")
    left_base.set("pos", "0 0 0")

    # Remove the second arm and scene elements that target it.  The generated
    # model is used for kinematics/rendering only, so dual-arm actuators and
    # keyframes are intentionally not carried over.
    for child in list(worldbody):
        if child.tag == "body" and child.get("name", "").startswith("right/"):
            worldbody.remove(child)
        elif child.tag == "light" and child.get("target", "").startswith("right/"):
            worldbody.remove(child)
    for tag in ("actuator", "equality", "contact", "keyframe"):
        section = root.find(tag)
        if section is not None:
            root.remove(section)

    ET.ElementTree(root).write(out, encoding="unicode")
    return out


def _make_composed_model(spec: RobotSpec) -> Path:
    """Merge a bare arm and Robotiq model into one reusable MJCF file."""
    if spec.gripper_model_path is None:
        return spec.model_path

    # Include source mtimes and composition parameters in the cache key so a
    # changed mount/calibration cannot silently reuse an old temporary XML.
    cache_key = repr((
        "composition-v7",
        spec.model_path.stat().st_mtime_ns,
        spec.gripper_model_path.stat().st_mtime_ns,
        spec.gripper_attach_body,
        spec.gripper_mount_body,
        spec.gripper_attach_pos,
        spec.gripper_attach_quat,
    )).encode()
    digest = hashlib.sha1(cache_key).hexdigest()[:12]
    out = Path("/tmp") / f"ego2robot_composed_{spec.name}_{digest}.xml"
    if out.exists():
        return out

    arm_root = ET.parse(spec.model_path).getroot()
    grip_root = ET.parse(spec.gripper_model_path).getroot()

    _expand_mjcf_includes(arm_root, spec.model_path.parent)
    _expand_mjcf_includes(grip_root, spec.gripper_model_path.parent)

    # A gripper may include the same generic root defaults (solimp/site) as
    # the arm.  Keeping a second unnamed site default can override the arm's
    # material reference after <attach> prefixes the asset names.  The arm
    # already carries an equivalent generic default, so drop this include-only
    # default while retaining hand-specific joint/geom defaults.
    for default in list(grip_root.findall("default")):
        if default.find("joint") is None and not default.findall("default"):
            grip_root.remove(default)
    for root in (arm_root, grip_root):
        for default in root.findall("default"):
            for site_default in list(default.findall("site")):
                if not site_default.get("class"):
                    default.remove(site_default)

    def absolutize_meshes(root, source_dir):
        for elem in root.iter("mesh"):
            filename = elem.get("file")
            if not filename:
                continue
            if not elem.get("name"):
                elem.set("name", Path(filename).stem)
            if not Path(filename).is_absolute():
                elem.set("file", str(source_dir / "assets" / filename))

    absolutize_meshes(arm_root, spec.model_path.parent)
    absolutize_meshes(grip_root, spec.gripper_model_path.parent)

    # Prefix every gripper symbol so it cannot collide with arm symbols after
    # both descriptions are merged.  References in constraints and geoms use
    # the same XML attributes and are rewritten together.
    names = {e.get("name") for e in grip_root.iter() if e.get("name")}
    classes = {e.get("class") for e in grip_root.iter() if e.get("class")}
    name_map = {n: f"grip_{n}" for n in names}
    class_map = {c: f"grip_{c}" for c in classes}
    reference_attrs = (
        "mesh", "material", "texture", "joint", "joint1", "joint2",
        "tendon", "body1", "body2", "site", "pair",
    )
    for elem in grip_root.iter():
        if elem.get("name") in name_map:
            elem.set("name", name_map[elem.get("name")])
        if elem.get("class") in class_map:
            elem.set("class", class_map[elem.get("class")])
        if elem.get("childclass") in class_map:
            elem.set("childclass", class_map[elem.get("childclass")])
        for attr in reference_attrs:
            if elem.get(attr) in name_map:
                elem.set(attr, name_map[elem.get(attr)])

    if not spec.gripper_attach_body:
        raise ValueError(f"robot {spec.name!r} has a gripper but no attach body")
    target = next((e for e in arm_root.iter("body")
                   if e.get("name") == spec.gripper_attach_body), None)
    if target is None:
        raise ValueError(
            f"cannot find gripper attach body {spec.gripper_attach_body!r} "
            f"in {spec.model_path}"
        )
    mount_name = name_map.get(spec.gripper_mount_body, f"grip_{spec.gripper_mount_body}")
    mount = next((e for e in grip_root.iter("body") if e.get("name") == mount_name), None)
    if mount is None:
        raise ValueError(f"cannot find gripper mount body {spec.gripper_mount_body!r}")
    parent = next((e for e in grip_root.iter("body") if mount in list(e)), None)
    if parent is not None:
        parent.remove(mount)
    else:
        grip_root.find("worldbody").remove(mount)
    mount.set("pos", " ".join(str(x) for x in spec.gripper_attach_pos))
    mount.set("quat", " ".join(str(x) for x in spec.gripper_attach_quat))
    target.append(mount)

    arm_asset = arm_root.find("asset")
    if arm_asset is None:
        arm_asset = ET.SubElement(arm_root, "asset")
    for grip_asset in grip_root.findall("asset"):
        for elem in list(grip_asset):
            arm_asset.append(elem)
    # Keep each source's top-level defaults separate.  In particular, Jaco's
    # common include and hand include both define an unnamed <site> default;
    # flattening them into one <default> violates MJCF's unique-child schema.
    for grip_default in list(grip_root.findall("default")):
        arm_root.append(grip_default)
    for section_name in ("actuator", "tendon", "equality", "contact"):
        source = grip_root.find(section_name)
        if source is None:
            continue
        section = arm_root.find(section_name)
        if section is None:
            section = ET.SubElement(arm_root, section_name)
        for elem in list(source):
            section.append(elem)

    # Bare menagerie arms often provide a qpos/ctrl keyframe sized for the arm
    # only.  It is invalid after adding the gripper joints and is not used by
    # this pipeline, so omit it from the composed asset.
    keyframe = arm_root.find("keyframe")
    if keyframe is not None:
        arm_root.remove(keyframe)

    ET.ElementTree(arm_root).write(out, encoding="unicode")
    return out


BODY_TCP_SITE = "ego2robot_tcp"


def _make_body_tcp_model(spec: RobotSpec, source: Path) -> Path:
    """Add an IK site at the physical TCP of a body-referenced gripper.

    Tracking the body origin and compensating the TCP offset in the target is
    only exact when the orientation target is also reached exactly. With the
    deliberately soft orientation task used for continuous IK, a 10 cm Panda
    hand-to-TCP offset otherwise becomes a large visible position error.
    """
    if spec.ee_body is None or not np.any(np.asarray(spec.tcp_pos_body)):
        return source
    cache_key = repr((
        "body-tcp-site-v1", source.stat().st_mtime_ns, spec.ee_body,
        spec.tcp_pos_body, spec.tcp_rot_body,
    )).encode()
    digest = hashlib.sha1(cache_key).hexdigest()[:12]
    out = Path("/tmp") / f"ego2robot_tcp_{spec.name}_{digest}.xml"
    if out.exists():
        return out

    root = ET.parse(source).getroot()
    target = next((e for e in root.iter("body")
                   if e.get("name") == spec.ee_body), None)
    if target is None:
        raise ValueError(
            f"cannot add TCP site: body {spec.ee_body!r} not found in {source}")
    old_site = next((e for e in root.iter("site")
                     if e.get("name") == BODY_TCP_SITE), None)
    if old_site is not None:
        raise ValueError(f"reserved site name already exists: {BODY_TCP_SITE}")
    ET.SubElement(
        target, "site", name=BODY_TCP_SITE,
        pos=" ".join(str(x) for x in spec.tcp_pos_body),
        quat=" ".join(str(x) for x in spec.tcp_rot_body),
        size="0.003", rgba="0 0 0 0", group="3",
    )

    # Direct menagerie XMLs resolve assets relative to their source directory,
    # while this calibrated variant is cached in /tmp.
    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value and not Path(value).is_absolute():
                compiler.set(attr, str((source.parent / value).resolve()))
    ET.ElementTree(root).write(out, encoding="unicode")
    return out


def get_model_path(spec: RobotSpec) -> Path:
    """Return a composed MJCF path for the requested robot specification."""
    if spec.name == "aloha_agilex":
        source = _make_aloha_single_model(spec)
    else:
        source = (_make_composed_model(spec)
                  if spec.gripper_model_path else spec.model_path)
    return _make_body_tcp_model(spec, source)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    import mujoco

    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


def resolve_robot_spec(model, spec: RobotSpec) -> RobotSpec:
    """Resolve a finger-midpoint TCP for models without a named TCP site."""
    import mujoco

    tcp_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, BODY_TCP_SITE)
    if tcp_site_id >= 0:
        return RobotSpec(**{
            **spec.__dict__,
            "ee_site": BODY_TCP_SITE,
            "tcp_pos_site": (0.0, 0.0, 0.0),
            "tcp_rot_site": (1.0, 0.0, 0.0, 0.0),
        })
    if not spec.finger_bodies or spec.ee_body is None:
        return spec

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body)
    finger_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                  for n in spec.finger_bodies]
    if ee_id < 0 or any(i < 0 for i in finger_ids):
        raise ValueError(f"invalid TCP calibration bodies for robot {spec.name}")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    R = data.xmat[ee_id].reshape(3, 3)
    midpoint = np.mean([data.xpos[i] for i in finger_ids], axis=0)
    tcp_pos = R.T @ (midpoint - data.xpos[ee_id])
    return RobotSpec(**{**spec.__dict__, "tcp_pos_body": tuple(float(x) for x in tcp_pos)})


def resolve_ee_ref(model, spec: RobotSpec) -> tuple[str, int]:
    """Return (body|site, id) for the reference used by IK."""
    import mujoco

    if spec.ee_site:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.ee_site)
        if idx >= 0:
            return "site", idx
    if spec.ee_body:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body)
        if idx >= 0:
            return "body", idx
    raise ValueError(f"cannot resolve end-effector for robot {spec.name}")


def resolve_prefixed_ee_ref(model, spec: RobotSpec, prefix: str) -> tuple[str, int]:
    """Resolve the end-effector in an attached, prefixed dual-arm model."""
    import mujoco

    if spec.ee_site:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + spec.ee_site)
        if idx >= 0:
            return "site", idx
    if spec.ee_body:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + spec.ee_body)
        if idx >= 0:
            return "body", idx
    raise ValueError(f"cannot resolve prefixed end-effector for robot {spec.name}")
