# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Assemble vendor hand geometry downstream of the actual Panda joint7."""

import functools
import xml.etree.ElementTree as E
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .registry import profile, selected

FLANGE = 0.107
ADAPTER = 0.03075
# Hands whose palm body origin is far from the palm itself (LEAP left: 115 mm). Their extended-hand
# axis starts at the palm-mesh centroid and lies in the palm plane instead of starting at the origin.
PALM_CENTROID = ("leap",)


def fmt(v):
    return " ".join(format(float(x), ".17g") for x in v)


@functools.lru_cache(None)
def mounting(name, side):
    """Fixed simulation adapter from model anatomy/geometry, never clip fitting."""

    p = profile(side, name)
    spec = mujoco.MjSpec.from_file(p["normalized"])
    m = spec.compile()
    d = mujoco.MjData(m)
    # Evaluate the published extended posture, respecting all affine linkage.
    names = p["joints"]
    memo = {}

    def value(n):
        if n in memo:
            return memo[n]
        if n in p["mimic"]:
            c = p["mimic"][n]
            x = c["offset"] + c["multiplier"] * value(c["joint"])
        else:
            j = m.joint(n).id
            x = float(np.clip(0, *m.jnt_range[j]))
        memo[n] = x
        return x

    for n in names:
        d.qpos[int(m.joint(n).qposadr[0])] = value(n)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    mujoco.mj_forward(m, d)
    palm = d.xpos[m.body(p["palm"]).id]
    tip = d.xpos[[m.body(t).id for t in p["tips"]]]
    if name in PALM_CENTROID:
        v = []
        pb = m.body(p["palm"]).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != pb or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            k = int(m.geom_dataid[g])
            a = int(m.mesh_vertadr[k])
            c = int(m.mesh_vertnum[k])
            v.append(m.mesh_vert[a : a + c] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        v = np.concatenate(v)
        c = v.mean(0)
        nrm = np.linalg.eigh((v - c).T @ (v - c))[1][:, 0]
        z = 0.5 * (tip[1] + tip[2]) - c
        z -= nrm * (z @ nrm)
        z /= np.linalg.norm(z)
    else:
        z = 0.5 * (tip[1] + tip[2]) - palm
        z /= np.linalg.norm(z)
    x = tip[1] - tip[3]
    x -= z * (x @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if name == "shadow":
        # Shadow's forearm mount face is the native root XY plane. Its wrist
        # axes are explicitly rooted along +Z in the vendor description;
        # the off-center neutral fingers must not tilt that mounting face.
        root_axes = d.xmat[m.body(p["root"]).id].reshape(3, 3)
        z = root_axes[:, 2].copy()
        x = root_axes[:, 0].copy() * (1 if x @ root_axes[:, 0] >= 0 else -1)
        y = np.cross(z, x)
    anatomy = np.stack([x, y, z], axis=1)
    R = Rotation.from_euler("z", -45, degrees=True).as_matrix() @ anatomy.T
    root = m.body(p["root"]).id
    fixed = []
    for g in range(m.ngeom):
        bid = int(m.geom_bodyid[g])
        chain = []
        b = bid
        while b > 0 and b != root:
            chain.append(b)
            b = int(m.body_parentid[b])
        if b != root or any(m.body_jntnum[b] for b in [root, *chain]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(m.geom_dataid[g])
            a = int(m.mesh_vertadr[mesh])
            n = int(m.mesh_vertnum[mesh])
            v = m.mesh_vert[a : a + n].copy()
        else:
            import itertools

            v = np.array(list(itertools.product(*zip(-m.geom_size[g], m.geom_size[g]))))
        fixed.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) @ R.T)
    if not fixed:
        raise ValueError("No physical root housing from which to define adapter plane")
    vertices = np.concatenate(fixed)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    mount_point = np.r_[0.5 * (low[:2] + high[:2]), low[2]]
    translation = np.array([0.0, 0.0, FLANGE + ADAPTER]) - mount_point
    rootR = R @ d.xmat[root].reshape(3, 3)
    rootP = R @ d.xpos[root] + translation
    return dict(
        position=rootP.tolist(),
        quaternion=Rotation.from_matrix(rootR).as_quat()[[3, 0, 1, 2]].tolist(),
        frame_position=translation.tolist(),
        frame_quaternion=Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]].tolist(),
        interface_native=(d.xmat[root].reshape(3, 3).T @ (R.T @ mount_point - d.xpos[root])).tolist(),
        anatomy_rotation=anatomy.tolist(),
        palm_anatomy_rotation=(d.xmat[m.body(p["palm"]).id].reshape(3, 3).T @ anatomy).tolist(),
        method="fixed model-derived extended-hand axis"
        + (" from palm-mesh centroid in palm plane" if name in PALM_CENTROID else "")
        + ", housing bounding-plane center, -45deg clocking",
        hardware_calibration_validated=False,
    )


def build(base_poses=None, mounts=None, include_floor=True):
    from steps import config

    name = selected()
    src = Path(config.MENAGERIE_DIR) / "franka_emika_panda/panda.xml"
    r = E.parse(src).getroot()
    r.find("compiler").set("meshdir", str(src.parent / "assets"))
    for body in r.iter("body"):
        for c in list(body):
            if c.tag == "body" and c.get("name") == "hand":
                body.remove(c)
    for tag in ["tendon", "equality", "keyframe"]:
        for c in r.findall(tag):
            r.remove(c)
    for a in list(r.find("actuator")):
        if a.get("tendon") is not None:
            r.find("actuator").remove(a)
    E.SubElement(r.find('.//body[@name="link7"]'), "site", name="attachment_site", pos=f"0 0 {FLANGE}", size=".005")
    arm = mujoco.MjSpec.from_string(E.tostring(r, encoding="unicode"))
    scene = mujoco.MjSpec()
    scene.modelname = "dual_panda_" + name
    scene.copy_during_attach = True
    scene.worldbody.add_light(name="top", pos=[0, 0, 2], dir=[0, 0, -1])
    if include_floor:
        scene.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0, 0, -0.015])
    for side, y in [("left", 0.4), ("right", -0.4)]:
        p = profile(side)
        pose = base_poses[side] if base_poses else dict(pos=[0, y, 0], quat=[1, 0, 0, 0])
        frame = scene.worldbody.add_frame(pos=pose["pos"], quat=pose["quat"])
        scene.attach(arm, prefix=side + "_", frame=frame)
        wrist = scene.body(side + "_link7")
        wrist.add_geom(
            name=side + "_coupler",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.025, ADAPTER / 2, 0],
            pos=[0, 0, FLANGE + ADAPTER / 2],
            mass=0.06,
            rgba=[0.2, 0.2, 0.22, 1],
            group=1,
        )
        hand = mujoco.MjSpec.from_file(p["normalized"])
        palm = hand.body(p["palm"])
        # Semantic aliases contain no geometry or joints. The real palm and
        # any vendor wrist joints are preserved, not welded or duplicated.
        palm.add_body(name=f"{side}_hand_C_MC")
        for digit, tip in zip(p["digits"], p["tips"]):
            hand.body(tip).add_site(name=f"{side}_{digit}_tip", size=[0.002, 0, 0], group=3)
        if len(p["digits"]) == 4:
            hand.body(p["tips"][-1]).add_site(name=f"{side}_pinky_tip", size=[0.002, 0, 0], group=3)
        mt = mounting(name, side)
        hand.body(p["root"]).add_site(
            name="assembly_interface", pos=mt["interface_native"], size=[0.002, 0, 0], group=3
        )
        hf = wrist.add_frame(pos=mt["frame_position"], quat=mt["frame_quaternion"])
        scene.attach(hand, prefix=side + "_hand_", frame=hf)
    m = scene.compile()
    # Enabling additional robot categories cannot silently remove a native
    # collision pair. Vendor visual geoms remain noncolliding.
    for geom in scene.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype = 1
            geom.conaffinity = 1
    return scene


def indices(model, side):
    arm_names = [f"{side}_joint{i}" for i in range(1, 8)]
    names = [f"{side}_hand_{n}" for n in profile(side)["joints"]]
    return dict(
        arm_qadr=np.array([int(model.joint(n).qposadr[0]) for n in arm_names]),
        finger_qadr=np.array([int(model.joint(n).qposadr[0]) for n in names]),
        site_id=model.site(side + "_attachment_site").id,
        arm_joint_names=arm_names,
        finger_joint_names=names,
    )


def standalone(spec, *unused):
    r = E.fromstring(spec.to_xml())
    for k in r.findall("keyframe"):
        r.remove(k)
    compiled = spec.compile()
    parents = {c: p for p in r.iter() for c in p}
    for side in ["left", "right"]:
        node = r.find(f'.//body[@name="{side}_link0"]')
        bid = compiled.body(side + "_link0").id
        assert int(compiled.body_parentid[bid]) == 0 and parents[node].tag == "worldbody"
        node.set("pos", fmt(compiled.body_pos[bid]))
        node.set("quat", fmt(compiled.body_quat[bid]))
    from steps import config

    panda = Path(config.MENAGERIE_DIR) / "franka_emika_panda/assets"
    for asset in r.findall("./asset/*"):
        f = asset.get("file")
        if not f or (Path(f).is_absolute() and Path(f).is_file()):
            continue
        name = asset.get("name", "")
        resolved = None
        for side in ["left", "right"]:
            prefix = side + "_hand_"
            if name.startswith(prefix):
                source = E.parse(profile(side)["normalized"]).getroot()
                match = source.find("./asset/" + asset.tag + '[@name="' + name[len(prefix) :] + '"]')
                if match is not None:
                    resolved = Path(match.get("file"))
        if resolved is None:
            candidates = list(panda.rglob(Path(f).name))
            if len(candidates) != 1:
                raise FileNotFoundError((name, f, candidates))
            resolved = candidates[0]
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        asset.set("file", str(resolved))
    # Restore assembly constants at full precision after MuJoCo XML output.
    for side in ["left", "right"]:
        p = profile(side)
        mt = mounting(selected(), side)
        # MjSpec writes attached frames explicitly; find the root's nearest
        # enclosing frame and stamp the declared transform without changing
        # any intrinsic palm/wrist articulation.
        root = r.find(f'.//body[@name="{side}_hand_{p["root"]}"]')
        parents = {c: p for p in r.iter() for c in p}
        frame = parents[root]
        if frame.tag == "frame":
            frame.set("pos", fmt(mt["frame_position"]))
            frame.set("quat", fmt(mt["frame_quaternion"]))
        else:
            root.set("pos", fmt(mt["position"]))
            root.set("quat", fmt(mt["quaternion"]))
        interface = r.find(f'.//site[@name="{side}_hand_assembly_interface"]')
        interface.set("pos", fmt(mt["interface_native"]))
        for joint, c in p["mimic"].items():
            eq = r.find(f'./equality/joint[@joint1="{side}_hand_{joint}"]')
            assert eq is not None, joint
            eq.set("polycoef", fmt([c["offset"], c["multiplier"], 0, 0, 0]))
    for sensor in r.findall("./sensor/user"):
        if sensor.get("objtype") is None:
            sensor.attrib.pop("objname", None)
    return r
