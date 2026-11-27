# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned public hand descriptions and build the normalized MuJoCo
models used by steps/hands (allegro, leap, wuji, inspire, shadow, svh, ability,
dexhand021).

    python tools/fetch_hand_assets.py [--out hand_assets]

Vendor geometry is never modified. The output directory holds:
  vendor/      pinned upstream checkouts (each keeps its own license),
               including mujoco_menagerie/franka_emika_panda
  converted/   GLB meshes converted to OBJ (dexhand021)
  meshes/      copies of multi-object OBJ meshes with o/g lines removed
  normalized/  {hand}_{side}.xml / .json used at run time

Generated files contain absolute paths. Re-run after moving the directory.
Sharpa is not included: obtain it from the official Do-As-I-Do release and
set EGO2ROBOT_SHARPA_UPSTREAM (see README).
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as E
from pathlib import Path

import mujoco

SOURCES = [
    (
        "https://github.com/dexsuite/dex-urdf",
        "f5e7132f22108164577fea4c25ef99b5cc0e1900",
        [
            "robots/hands/" + n
            for n in ["allegro_hand", "leap_hand", "shadow_hand", "inspire_hand", "schunk_hand", "ability_hand"]
        ],
    ),
    ("https://github.com/wuji-technology/wuji-hand-description", "bbc2f64eec834aae616eacc2db1f69016f860c2d", None),
    ("https://github.com/DexRobot/dexrobot_mujoco", "9ab142309aa2392d6923601064d3de8c58ba5ee3", None),
    (
        "https://github.com/fzi-forschungszentrum-informatik/schunk_svh_ros_driver",
        "121dbf8743f2ddfaa1fc8c08fb4c474937391a14",
        None,
    ),
    # Panda arm used by every hand scene (default EGO2ROBOT_MENAGERIE_DIR).
    (
        "https://github.com/google-deepmind/mujoco_menagerie",
        "71f066ad0be9cd271f7ed58c030243ef157af9f4",
        ["franka_emika_panda"],
    ),
]

# name: (URDF under dex-urdf/robots/hands, palm body, fingertip bodies thumb..pinky)
MODELS = {
    "allegro": (
        "allegro_hand/allegro_hand_{side}.urdf",
        "base_link",
        ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"],
    ),
    "leap": (
        "leap_hand/leap_hand_{side}.urdf",
        "palm_lower",
        ["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"],
    ),
    "inspire": (
        "inspire_hand/inspire_hand_{side}.urdf",
        "hand_base_link",
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
    ),
    "shadow": ("shadow_hand/shadow_hand_{side}.urdf", "palm", ["thtip", "fftip", "mftip", "rftip", "lftip"]),
    "svh": (
        "schunk_hand/schunk_svh_hand_{side}.urdf",
        "{side}_hand_base_link",
        ["thtip", "fftip", "mftip", "rftip", "lftip"],
    ),
    "ability": (
        "ability_hand/ability_hand_{side}.urdf",
        "thumb_base",
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
    ),
}
# The left Allegro digits are mirrored: index is link_11 and ring is link_3.
LEFT_TIPS = {"allegro": ["link_15.0_tip", "link_11.0_tip", "link_7.0_tip", "link_3.0_tip"]}
HANDS = [*MODELS, "wuji", "dexhand021"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, capture_output=True).stdout.strip()


def fetch(vendor):
    vendor.mkdir(parents=True, exist_ok=True)
    for url, commit, sparse in SOURCES:
        root = vendor / url.rstrip("/").split("/")[-1]
        if not (root / ".git").exists():
            root.mkdir(exist_ok=True)
            git(root, "init", "-q")
            git(root, "remote", "add", "origin", url + ".git")
        elif git(root, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"Refusing to overwrite edited vendor checkout: {root}")
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], text=True, capture_output=True)
        if head.returncode or head.stdout.strip() != commit:
            git(root, "fetch", "-q", "--depth", "1", "origin", commit)
            if sparse:
                git(root, "sparse-checkout", "set", *sparse)
            git(root, "checkout", "-q", "--detach", "FETCH_HEAD")
        assert git(root, "rev-parse", "HEAD") == commit, root
        print("pinned", root.name, commit, flush=True)


def absolute_meshes(tree, src, converted):
    for mesh in tree.findall(".//mesh"):
        f = mesh.get("filename")
        if not f:
            continue
        if f.startswith("package://"):
            candidate = src.parent / f[len("package://") :].split("/", 1)[-1]
        else:
            candidate = src.parent / f
        if not candidate.is_file():
            choices = list(src.parent.rglob(Path(f).name))
            if len(choices) != 1:
                raise FileNotFoundError((f, choices))
            candidate = choices[0]
        mesh.set("filename", str(candidate.resolve()))
        if candidate.suffix.lower() == ".glb":
            import trimesh

            obj = converted / (candidate.stem + ".obj")
            if not obj.exists():
                trimesh.load(candidate, force="scene").dump(concatenate=True).export(obj)
            mesh.set("filename", str(obj))


def normalize(name, side, vendor, stage, converted):
    """Self-contained MJCF with explicit mimic equalities; writes {name}_{side}.xml/.json into stage."""
    out = stage / f"{name}_{side}.xml"
    mimics, velocity, patches = {}, {}, []
    if name in MODELS:
        rel, palm, tips = MODELS[name]
        if side == "left":
            tips = LEFT_TIPS.get(name, tips)
        src = vendor / "dex-urdf/robots/hands" / rel.format(side=side)
        r = E.parse(src).getroot()
        absolute_meshes(r, src, converted)
        children = {j.find("child").get("link") for j in r.findall("joint")}
        roots = [x.get("name") for x in r.findall("link") if x.get("name") not in children]
        assert len(roots) == 1
        root = roots[0]
        if name == "shadow" and root == "world":
            r.remove(r.find('./link[@name="world"]'))
            r.remove(r.find('./joint[@name="world_joint"]'))
            root = "forearm"
        if name == "leap" and side == "left":
            palm = "palm_lower_left"
        if name == "svh":
            # The official ROS description fixes J5 (defective DH mechanism); the
            # community left-hand URDF has a positive mimic against a negative
            # slave range. Follow the official fixed-J5 model on both sides.
            official = vendor / "schunk_svh_ros_driver/schunk_svh_description/urdf/svh.urdf.xacro"
            assert "J5 JOINTS is set to fixed temporarily due to defective DH algo" in official.read_text()
            j = r.find(f'./joint[@name="{side}_hand_j5"]')
            j.set("type", "fixed")
            for tag in ["mimic", "limit", "axis"]:
                child = j.find(tag)
                if child is not None:
                    j.remove(child)
            patches.append(
                dict(
                    joint=side + "_hand_j5",
                    policy="fixed per official ROS model",
                    reference="schunk_svh_ros_driver/schunk_svh_description/urdf/svh.urdf.xacro",
                    reference_sha256=sha(official),
                )
            )
        ext = r.find("mujoco")
        if ext is None:
            ext = E.SubElement(r, "mujoco")
        compiler = ext.find("compiler")
        if compiler is None:
            compiler = E.SubElement(ext, "compiler")
        compiler.set("discardvisual", "false")
        compiler.set("fusestatic", "false")
        compiler.set("balanceinertia", "true")
        for j in r.findall("joint"):
            if j.get("type") == "fixed":
                continue
            velocity[j.get("name")] = min(3.0, float(j.find("limit").get("velocity", "3")))
            mimic = j.find("mimic")
            if mimic is not None:
                mimics[j.get("name")] = dict(
                    joint=mimic.get("joint"),
                    multiplier=float(mimic.get("multiplier", "1")),
                    offset=float(mimic.get("offset", "0")),
                )
        if name == "shadow":
            # Declared equal-split command synergy for the coupled distal pairs
            # (kinematic policy, not a tendon simulation).
            for finger in ["FF", "MF", "RF", "LF"]:
                mimics[finger + "J1"] = dict(joint=finger + "J2", multiplier=1.0, offset=0.0)
        temp = stage / f"{name}_{side}.urdf"
        temp.write_text(E.tostring(r, encoding="unicode"))
        spec = mujoco.MjSpec.from_file(str(temp))
        spec.compile()
        r = E.fromstring(spec.to_xml())
        palm = palm.format(side=side)
    else:
        if name == "wuji":
            src = vendor / "wuji-hand-description/mjcf" / f"{side}.xml"
        else:
            src = vendor / "dexrobot_mujoco/dexrobot_mujoco/models" / f"dexhand021_{side}.xml"
        r = E.parse(src).getroot()
        compiler = r.find("compiler")
        meshdir = (src.parent / compiler.get("meshdir", "")).resolve()
        for mesh in r.findall("./asset/mesh"):
            if mesh.get("file"):
                mesh.set("file", str((meshdir / mesh.get("file")).resolve()))
        compiler.attrib.pop("meshdir", None)
        root = r.find("./worldbody/body").get("name")
        palm = root
        if name == "wuji":
            tips = [f"{side}_finger{i}_tip_link" for i in range(1, 6)]
            urdf = E.parse(vendor / "wuji-hand-description/urdf" / f"{side}.urdf").getroot()
            for tip in tips:
                j = next(j for j in urdf.findall("joint") if j.find("child").get("link") == tip)
                parent = r.find(f'.//body[@name="{j.find("parent").get("link")}"]')
                E.SubElement(parent, "body", name=tip, pos=j.find("origin").get("xyz", "0 0 0"))
        else:
            tips = [f"{side[0]}_f_link{i}_tip" for i in range(1, 6)]
        for include in r.findall(".//include"):
            include.set("file", str((src.parent / include.get("file")).resolve()))
        for j in r.findall(".//joint"):
            if j.get("name"):
                velocity[j.get("name")] = 3.0
        if r.find("tendon") is not None:
            raise ValueError("Native tendon mechanism requires an explicit reduced-coordinate contract")
        for eq in r.findall("./equality/joint"):
            co = [float(x) for x in eq.get("polycoef", "0 1 0 0 0").split()]
            if any(abs(x) > 1e-12 for x in co[2:]):
                raise ValueError("Nonlinear native equality requires an explicit mechanism")
            mimics[eq.get("joint1")] = dict(joint=eq.get("joint2"), offset=co[0], multiplier=co[1])
    # Imported URDFs omit mimic constraints: declare them explicitly in MJCF.
    eq = r.find("equality")
    if eq is None:
        eq = E.SubElement(r, "equality")
    existing = {x.get("joint1") for x in eq.findall("joint")}
    for joint, c in mimics.items():
        if joint not in existing:
            E.SubElement(
                eq,
                "joint",
                name="mimic_" + joint,
                joint1=joint,
                joint2=c["joint"],
                polycoef=f"{c['offset']:.17g} {c['multiplier']:.17g} 0 0 0",
            )
    for key in r.findall("keyframe"):
        r.remove(key)
    r.set("model", name + "_" + side)
    out.write_text(E.tostring(r, encoding="unicode"))
    spec = mujoco.MjSpec.from_file(str(out))
    spec.compile()
    # Asset paths emitted by MuJoCo must resolve after another serialization.
    saved = E.fromstring(spec.to_xml())
    for asset in saved.findall("./asset/*"):
        f = asset.get("file")
        if f and not Path(f).is_absolute():
            candidates = list(src.parent.rglob(Path(f).name)) or list(src.parent.parent.rglob(Path(f).name))
            if len(candidates) != 1:
                raise FileNotFoundError((f, candidates))
            asset.set("file", str(candidates[0]))
    out.write_text(E.tostring(saved, encoding="unicode"))
    m = mujoco.MjModel.from_xml_path(str(out))
    names = [m.joint(i).name for i in range(m.njnt)]
    assert all(m.body(x).id > 0 for x in [root, palm, *tips]), (root, palm, tips)
    assert m.nq == m.nv == m.njnt, "Only bounded scalar mechanical joints are supported"
    row = dict(
        name=name,
        side=side,
        source=str(src.relative_to(vendor)),
        source_sha256=sha(src),
        normalized=str(out),
        root=root,
        palm=palm,
        tips=tips,
        digits=["thumb", "index", "middle", "ring", "pinky"][: len(tips)],
        joints=names,
        physical_joint_count=m.nq,
        motor_names=[n for n in names if n not in mimics],
        mimic=mimics,
        velocity_limits=velocity,
        geoms=m.ngeom,
        equalities=m.neq,
        actuators=m.nu,
        collision_category_normalization="positive vendor masks become 1/1 only in assembled model; visual 0/0 preserved",
        explicit_source_patches=patches,
        hardware_calibration_validated=False,
    )
    (stage / f"{name}_{side}.json").write_text(json.dumps(row, indent=2))
    return row


def objects_with_faces(path):
    per, cur = {}, "<none>"
    for line in open(path, errors="ignore"):
        if line.startswith(("o ", "g ")):
            cur = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            per[cur] = per.get(cur, 0) + 1
    return per


def merge_objects(stage, meshes, normalized):
    """MuJoCo 3.11 loads only the first object of a multi-object OBJ (LEAP
    fingertips, Shadow forearm). Copy such meshes with the o/g lines removed
    (vertex, normal, texcoord and face lines unchanged) and point the model at
    the copy."""
    changed = {}
    for x in sorted(stage.glob("*.xml")):
        tree = E.parse(x)
        for me in tree.getroot().iter("mesh"):
            f = me.get("file", "")
            if not f.lower().endswith(".obj") or len(objects_with_faces(f)) <= 1:
                continue
            dst = meshes / f.split("/robots/hands/", 1)[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with open(f, errors="ignore") as a, open(dst, "w") as b:
                    b.writelines(line for line in a if not line.startswith(("o ", "g ")))
            me.set("file", str(dst))
            changed.setdefault(x.name, []).append(me.get("name"))
        tree.write(normalized / x.name)
        profile = json.loads(x.with_suffix(".json").read_text())
        profile["normalized"] = str(normalized / x.name)
        (normalized / x.with_suffix(".json").name).write_text(json.dumps(profile, indent=1))
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "hand_assets")
    ap.add_argument("--skip_fetch", action="store_true", help="Reuse existing vendor/ checkouts")
    args = ap.parse_args()
    out = args.out.resolve()
    vendor = out / "vendor"
    if not args.skip_fetch:
        fetch(vendor)
    stage, converted, meshes, normalized = out / "_stage", out / "converted", out / "meshes", out / "normalized"
    for d in (stage, meshes, normalized):
        shutil.rmtree(d, ignore_errors=True)
    for d in (stage, converted, meshes, normalized):
        d.mkdir(parents=True, exist_ok=True)
    for name in HANDS:
        for side in ["left", "right"]:
            row = normalize(name, side, vendor, stage, converted)
            print(
                "normalized",
                name,
                side,
                row["physical_joint_count"],
                "joints",
                len(row["motor_names"]),
                "motors",
                flush=True,
            )
    changed = merge_objects(stage, meshes, normalized)
    shutil.rmtree(stage)
    print("merged multi-object OBJ meshes:", json.dumps(changed))
    print("hand assets ready:", normalized)


if __name__ == "__main__":
    main()
