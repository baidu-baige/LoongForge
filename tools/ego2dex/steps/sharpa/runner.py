# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Unified Sharpa entry with provenance, dual-target audits and motor export."""

from copy import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def _capture(output, cache, reference, parameters):
    """Archive actual source bytes and freeze original Orientation provenance."""
    from .backend import upstream_root
    from steps import config

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache, reference = Path(cache).resolve(), Path(reference).resolve()
    app = Path(__file__).resolve().parents[2]
    files = {
        str(p.relative_to(app)): p.read_bytes()
        for p in app.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.startswith("._")
    }
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    archive_root = Path(os.environ.get("EGO2ROBOT_SOURCE_ARCHIVES", str(output / "source_archives")))
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / (digest + ".zip")
    if not archive.exists():
        with tempfile.NamedTemporaryFile(dir=archive_root, suffix=".zip.tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as package:
                for name, value in files.items():
                    package.writestr(name, value)
            os.replace(temporary, archive)
        finally:
            if temporary.exists():
                temporary.unlink()
    manifest = json.loads((reference / "manifest.json").read_text())
    episode = Path(manifest["episode"]).name
    seed = cache / "06_retarget" / (episode + "_joint_pose_seed.npz")
    inputs = [
        seed,
        seed.with_name(episode + "_joint_pose.json"),
        reference / "manifest.json",
        reference / "hand_reference.npz",
        cache / "04_inpaint" / episode / "bg.mp4",
        cache / "03_mask" / episode / "hand_arm_masks.npz",
    ]
    original_ik = cache / "06_retarget" / (episode + "_ik.npz")
    if original_ik.exists():
        inputs.append(original_ik)
    original_status_path = cache.parent / "run_status.json"
    original_verification_path = cache.parent / "verification.json"
    original_status = json.loads(original_status_path.read_text()) if original_status_path.exists() else {}
    original_verification = (
        json.loads(original_verification_path.read_text()) if original_verification_path.exists() else {}
    )
    inputs.extend(path for path in (original_status_path, original_verification_path) if path.exists())
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(p)
    modelroot = upstream_root() / "deployment/mujoco_replay/assets/sharpa_assets/SharpaWave_URDF_XML_USD_V3.0.0/src"
    assets = {
        str(modelroot / f"{side}_sharpa_wave/{side}_sharpa_wave.xml"): _sha(
            modelroot / f"{side}_sharpa_wave/{side}_sharpa_wave.xml"
        )
        for side in ("left", "right")
    }
    panda = Path(config.MENAGERIE_DIR) / "franka_emika_panda/panda.xml"
    assets[str(panda)] = _sha(panda)
    record = dict(
        hand="sharpa",
        episode=episode,
        source_digest=digest,
        source_archive=str(archive),
        source_hashes=hashes,
        original_orientation_cache=str(cache),
        reference=str(reference),
        original_orientation_run_status=original_status.get("status", "unknown_no_run_status"),
        original_arm_accepted=bool(original_ik.exists() and original_verification.get("accepted", False)),
        original_input_sha256={str(p): _sha(p) for p in inputs},
        model_xml_sha256=assets,
        parameters=parameters,
        initializer_policy=os.environ.get("EGO2ROBOT_HAND_INITIALIZER", "v28-v29"),
        acceptance_profile=os.environ.get("EGO2ROBOT_ACCEPTANCE_PROFILE", "mild"),
        configured_speed_cap_rad_s=3.0,
        vendor_velocity_rating_declared=False,
    )
    _write(output / "run_provenance.json", record)
    return record


def _check_original_inputs(provenance):
    for path, expected in provenance["original_input_sha256"].items():
        if _sha(path) != expected:
            raise ValueError("Original Orientation input/cache changed: " + path)


def _status(output, passed, **fields):
    output = Path(output)
    path = output / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(fields, independent_kinematic_and_video_passed=bool(passed))
    # Preserve native quality_passed: its broad contact/dynamics scope is
    # distinct from this independent kinematic/video acceptance.
    _write(path, status)


def _validate_targets(output, reference, revision, report, provenance):
    from .validate_result import validate

    output, reference = Path(output), Path(reference)
    raw = validate(
        output,
        "sharpa",
        reference.name,
        reference_dir=reference,
        seed_source=Path(revision["raw_seed"]),
        report_path=output / "original_reference_validation.json",
        write_artifacts=False,
        require_goal_match=False,
    )
    raw.update(target_profile="original", source_rotation_changed_frames=[])
    _write(output / "original_reference_validation.json", raw)
    active = validate(
        output,
        "sharpa",
        reference.name,
        reference_dir=reference,
        seed_source=Path(revision["seed"]),
        require_goal_match=True,
        write_artifacts=True,
    )
    active.update(
        target_profile=revision["target_profile"],
        goal_sha256=revision["goal_sha256"],
        source_rotation_changed_frames=revision["changed_frames"],
    )
    _write(output / "independent_validation.json", active)
    _write(output / "stable_reference_validation.json", active)
    _check_original_inputs(provenance)
    report.update(
        independent_validation=active,
        original_reference_passed=bool(raw["passed"]),
        target_profile=revision["target_profile"],
        source_rotation_revision=revision,
        independent_kinematic_and_video_passed=bool(active["passed"]),
        source_digest=provenance["source_digest"],
        original_inputs_unchanged=True,
        original_orientation_run_status=provenance["original_orientation_run_status"],
        original_arm_accepted=provenance["original_arm_accepted"],
        original_rejection_preserved=not provenance["original_arm_accepted"],
    )
    _write(output / "replacement_report.json", report)
    _status(
        output,
        active["passed"],
        stage="independent_validation_passed" if active["passed"] else "independent_validation_failed",
        generation_completed=True,
        result_video="robot_on_bg.mp4",
        target_profile=revision["target_profile"],
        original_reference_passed=bool(raw["passed"]),
    )
    if not active["passed"]:
        failed = ", ".join(key for key, passed in active["checks"].items() if not passed)
        raise RuntimeError("Independent Sharpa final-reference acceptance failed: " + failed)
    _write_arm_seed(output)
    return active


def _write_arm_seed(output):
    """Save the static Panda bases and arm joints as arm_seed.npz, the
    EGO2ROBOT_ARM_INITIALIZER format, so other hands can reuse this result
    without loading the Sharpa meshes."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    qpos = np.load(output / "trajectory.npz")["qpos"]
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    seed = {}
    for side in ("left", "right"):
        body = model.body(side + "_link0").id
        seed[side + "_base_pos"] = data.xpos[body].copy()
        seed[side + "_base_quat"] = data.xquat[body].copy()
        seed[side + "_arm_qpos"] = np.stack(
            [qpos[:, int(model.joint(f"{side}_joint{j}").qposadr[0])] for j in range(1, 8)], axis=1
        )
    np.savez_compressed(output / "arm_seed.npz", **seed)


def _refresh_artifacts(output):
    output = Path(output)
    _write(
        output / "artifact_sha256.json",
        {
            str(p.relative_to(output)): _sha(p)
            for p in output.rglob("*")
            if p.is_file() and p.name not in ("artifact_sha256.json", "status.json")
        },
    )


def run_cached(args):
    from . import _native_runner_base
    from steps.hands.rotation_targets import prepare

    if args.robot_type != "panda" or args.end_effector != "sharpa":
        raise ValueError("Native Sharpa currently supports Panda and the Sharpa hand")
    os.environ["EGO2ROBOT_END_EFFECTOR"] = "sharpa"
    output, reference = Path(args.output_dir), Path(args.hand_reference_dir)
    provenance = _capture(
        output,
        args.orientation_cache,
        reference,
        dict(
            entrypoint="cli.py retarget",
            orientation_cache=str(args.orientation_cache),
            reference=str(reference),
            hand_rotation_policy=getattr(args, "hand_rotation_policy", "stabilize"),
        ),
    )
    revision = prepare(
        args.orientation_cache,
        reference,
        output / "_target_reference",
        getattr(args, "hand_rotation_policy", "stabilize"),
    )
    forwarded = copy(args)
    forwarded.orientation_cache = revision["orientation_cache"]
    report = _native_runner_base.run_cached(forwarded)
    _validate_targets(output, reference, revision, report, provenance)
    _refresh_artifacts(output)
    return report


def replace_pipeline_outputs(args, root, original_retarget):
    from . import _native_runner_base
    from .hand_swap import swap_grippers
    from steps.hands.rotation_targets import prepare, acceptance_fields

    if args.robot_type != "panda" or args.end_effector != "sharpa":
        raise ValueError("Native Sharpa currently supports Panda and the Sharpa hand")
    os.environ["EGO2ROBOT_END_EFFECTOR"] = "sharpa"

    def replacement(cache, reference, result, **kwargs):
        result, reference = Path(result), Path(reference)
        provenance = _capture(
            result,
            cache,
            reference,
            dict(
                entrypoint="cli.py run-all",
                orientation_cache=str(cache),
                reference=str(reference),
                hand_rotation_policy=getattr(args, "hand_rotation_policy", "stabilize"),
            ),
        )
        revision = prepare(
            cache, reference, result / "_target_reference", getattr(args, "hand_rotation_policy", "stabilize")
        )
        report = swap_grippers(Path(revision["orientation_cache"]), reference, result, **kwargs)
        _validate_targets(result, reference, revision, report, provenance)
        _refresh_artifacts(result)
        return report

    output = _native_runner_base.replace_pipeline_outputs(args, root, original_retarget, replacement_fn=replacement)
    for path in output.glob("*_ik.npz"):
        episode = path.name[: -len("_ik.npz")]
        if args.episodes and episode not in args.episodes:
            # Left over from an earlier run with another episode selection.
            continue
        result = output / "_render" / episode
        validation = json.loads((result / "independent_validation.json").read_text())
        if not validation["passed"]:
            raise RuntimeError("Sharpa cannot export a rejected trajectory")
        original = np.load(path)
        metadata = {key: original[key] for key in original.files}
        motor = np.load(result / "motor_actions.npz")
        quality = np.load(result / "ik_quality.npz")
        trajectory = np.load(result / "trajectory.npz")
        metadata.update(
            state=motor["motor_state"].astype(np.float32),
            state_names=motor["motor_names"],
            physical_state_names=motor["physical_state_names"],
            physical_from_motor_matrix=motor["physical_from_motor_matrix"],
            physical_from_motor_offset=motor["physical_from_motor_offset"],
            actuator_names=motor["actuator_names"],
            robot_type=np.array("panda_sharpa_independent_commands"),
            independent_kinematic_and_video_passed=np.array(True),
            **{
                key: trajectory[key]
                for key in ("timestamp", "timestamps_ns", "source_frame_indices", "camera_K", "obs_head_pose")
            },
            **{key: quality[key] for key in ("ik_ok", "ik_err_pos", "ik_err_rot")},
            **acceptance_fields(result),
        )
        np.savez_compressed(path, **metadata)
    return output
