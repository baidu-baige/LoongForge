# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Panda arms from Pipeline 2 with independently articulated Sharpa hands.

The mount is a simulation convention, not a measured hardware adapter.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .mount_contract import canonical_mounts, validate_mounts, LINK7_TO_FLANGE, BLACK_HEIGHT, SILVER_HEIGHT, HAND_OFFSET


def build(base_poses=None, mounts=None, include_floor=True):
    import mujoco
    from .backend import upstream_root
    from steps import config

    root = upstream_root()
    mounts = validate_mounts(mounts or canonical_mounts())
    src = Path(config.MENAGERIE_DIR) / "franka_emika_panda/panda.xml"
    tree = ET.parse(src)
    xml = tree.getroot()
    for compiler in xml.findall("compiler"):
        compiler.set("meshdir", str(src.parent / "assets"))
    for body in xml.iter("body"):
        for child in list(body):
            if child.tag == "body" and child.get("name") == "hand":
                body.remove(child)
    for tag in ["tendon", "equality", "keyframe"]:
        for x in xml.findall(tag):
            xml.remove(x)
    actuator = xml.find("actuator")
    for a in list(actuator):
        if a.get("tendon") is not None:
            actuator.remove(a)
    link7 = xml.find('.//body[@name="link7"]')
    ET.SubElement(link7, "site", name="attachment_site", pos=f"0 0 {LINK7_TO_FLANGE}", size="0.005")
    arm = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
    scene = mujoco.MjSpec()
    scene.modelname = "dual_panda_sharpa"
    scene.copy_during_attach = True
    scene.worldbody.add_light(name="top", pos=[0, 0, 2], dir=[0, 0, -1])
    if include_floor:
        scene.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[2, 2, 0.1],
            pos=[0, 0, -0.015],
            rgba=[0.85, 0.85, 0.9, 1],
        )
    handroot = root / "deployment/mujoco_replay/assets/sharpa_assets/SharpaWave_URDF_XML_USD_V3.0.0/src"
    for side, y in [("left", 0.4), ("right", -0.4)]:
        pose = base_poses[side] if base_poses else dict(pos=[0, y, 0], quat=[1, 0, 0, 0])
        frame = scene.worldbody.add_frame(pos=pose["pos"], quat=pose["quat"])
        scene.attach(arm, prefix=side + "_", frame=frame)
        # Bridge the same30.75mm mount offset used in the wrist transform.
        # Dimensions follow the upstream adapter; this remains a simulation mount.
        wrist = scene.body(f"{side}_link7")
        wrist.add_geom(
            name=f"{side}_coupler_black",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.034, BLACK_HEIGHT / 2, 0],
            pos=[0, 0, LINK7_TO_FLANGE + BLACK_HEIGHT / 2],
            mass=0.04,
            rgba=[0.05, 0.05, 0.05, 1],
            group=1,
        )
        wrist.add_geom(
            name=f"{side}_coupler_silver",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.021, SILVER_HEIGHT / 2, 0],
            pos=[0, 0, LINK7_TO_FLANGE + BLACK_HEIGHT + SILVER_HEIGHT / 2],
            mass=0.02,
            rgba=[0.65, 0.65, 0.7, 1],
            group=1,
        )
        hand = mujoco.MjSpec.from_file(str(handroot / f"{side}_sharpa_wave/{side}_sharpa_wave.xml"))
        # Deployment assets omit IK sites. Use the exact distal-body offsets
        # declared by the pinned retargeting Sharpa assets.
        for digit in ["thumb", "index", "middle", "ring", "pinky"]:
            hand.body(f"{side}_{digit}_DP").add_site(
                name=f"{side}_{digit}_tip",
                pos=[0.015 if digit == "thumb" else 0.014, 0, 0],
                size=[0.002, 0, 0],
                group=3,
            )
        hand.body(f"{side}_hand_C_MC").pos = [0, 0, HAND_OFFSET]
        hand.body(f"{side}_hand_C_MC").quat = mounts[side]["quaternion_link7_wxyz"]
        scene.attach(hand, prefix=f"{side}_hand_", site=scene.site(f"{side}_attachment_site"))
    m = scene.compile()
    q = m.qpos0.copy()
    ctrl = np.zeros(m.nu)
    home = [0, -0.4, 0, -2, 0, 1.6, 0.8]
    for side in ["left", "right"]:
        for i, value in enumerate(home, 1):
            j = m.joint(f"{side}_joint{i}").id
            q[int(m.jnt_qposadr[j])] = value
            ctrl[m.actuator(f"{side}_actuator{i}").id] = value
    scene.add_key(name="home", qpos=q, ctrl=ctrl)
    return scene


def indices(model, side):
    names = [f"{side}_joint{i}" for i in range(1, 8)]
    arm = np.array([int(model.jnt_qposadr[model.joint(n).id]) for n in names])
    # Preserve actuator joint order used by the upstream floating Sharpa trajectory.
    suffix = [
        "thumb_CMC_FE",
        "thumb_CMC_AA",
        "thumb_MCP_FE",
        "thumb_MCP_AA",
        "thumb_IP",
        "index_MCP_FE",
        "index_MCP_AA",
        "index_PIP",
        "index_DIP",
        "middle_MCP_FE",
        "middle_MCP_AA",
        "middle_PIP",
        "middle_DIP",
        "ring_MCP_FE",
        "ring_MCP_AA",
        "ring_PIP",
        "ring_DIP",
        "pinky_CMC",
        "pinky_MCP_FE",
        "pinky_MCP_AA",
        "pinky_PIP",
        "pinky_DIP",
    ]
    fn = [f"{side}_hand_{side}_{s}" for s in suffix]
    fingers = np.array([int(model.jnt_qposadr[model.joint(n).id]) for n in fn])
    return dict(
        arm_qadr=arm,
        finger_qadr=fingers,
        site_id=model.site(f"{side}_attachment_site").id,
        arm_joint_names=names,
        finger_joint_names=fn,
    )
