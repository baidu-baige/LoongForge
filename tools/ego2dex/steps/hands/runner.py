# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Formal end-effector entrypoints with independent acceptance and motor export."""

import json
import os
from copy import copy
from pathlib import Path
import numpy as np


def _check(result, hand, cache, reference):
    from .validate_result import validate

    manifest = json.loads((reference / "manifest.json").read_text())
    ep = Path(manifest["episode"]).name
    validation = validate(
        result,
        hand,
        reference.name,
        reference_dir=reference,
        seed_source=cache / "06_retarget" / (ep + "_joint_pose_seed.npz"),
    )
    if not validation["passed"]:
        raise RuntimeError("Independent registered-hand acceptance failed")
    return validation


def run_cached(args):
    from . import _runner_base

    os.environ["EGO2ROBOT_END_EFFECTOR"] = args.end_effector
    from .provenance import capture

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    capture(
        out,
        args.end_effector,
        dict(orientation_cache=args.orientation_cache, reference=args.hand_reference_dir, entrypoint="cli.py retarget"),
    )
    from .rotation_targets import prepare
    from .validate_result import validate

    revision = prepare(
        args.orientation_cache,
        args.hand_reference_dir,
        out / "_target_reference",
        getattr(args, "hand_rotation_policy", "original"),
    )
    forwarded = copy(args)
    forwarded.orientation_cache = revision["orientation_cache"]
    report = _runner_base.run_cached(forwarded)
    raw = validate(
        out,
        args.end_effector,
        Path(args.hand_reference_dir).name,
        reference_dir=Path(args.hand_reference_dir),
        seed_source=Path(revision["raw_seed"]),
    )
    raw.update(target_profile="original")
    (out / "original_reference_validation.json").write_text(json.dumps(raw, indent=2))
    validation = _check(out, args.end_effector, Path(forwarded.orientation_cache), Path(args.hand_reference_dir))
    validation.update(
        target_profile=revision["target_profile"],
        goal_sha256=revision["goal_sha256"],
        source_rotation_changed_frames=revision["changed_frames"],
    )
    (out / "independent_validation.json").write_text(json.dumps(validation, indent=2))
    (out / "stable_reference_validation.json").write_text(json.dumps(validation, indent=2))
    report["independent_kinematic_and_video_passed"] = True
    report.update(
        independent_validation=validation,
        target_profile=revision["target_profile"],
        source_rotation_revision=revision,
        original_reference_passed=raw["passed"],
    )
    (out / "replacement_report.json").write_text(json.dumps(report, indent=2))
    status = json.loads((out / "status.json").read_text())
    status["independent_kinematic_and_video_passed"] = True
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return report


def replace_pipeline_outputs(args, root, original_retarget):
    from . import _runner_base

    os.environ["EGO2ROBOT_END_EFFECTOR"] = args.end_effector
    from .rotation_targets import prepare
    from .hand_swap import swap_grippers
    from .validate_result import validate

    def replacement(cache, reference, result, **kwargs):
        revision = prepare(
            cache, reference, Path(result) / "_target_reference", getattr(args, "hand_rotation_policy", "original")
        )
        report = swap_grippers(Path(revision["orientation_cache"]), reference, result, **kwargs)
        raw = validate(
            result,
            args.end_effector,
            Path(reference).name,
            reference_dir=Path(reference),
            seed_source=Path(revision["raw_seed"]),
        )
        raw.update(target_profile="original")
        (Path(result) / "original_reference_validation.json").write_text(json.dumps(raw, indent=2))
        stable = validate(
            result,
            args.end_effector,
            Path(reference).name,
            reference_dir=Path(reference),
            seed_source=Path(revision["seed"]),
        )
        stable.update(
            target_profile=revision["target_profile"],
            goal_sha256=revision["goal_sha256"],
            source_rotation_changed_frames=revision["changed_frames"],
        )
        (Path(result) / "independent_validation.json").write_text(json.dumps(stable, indent=2))
        (Path(result) / "stable_reference_validation.json").write_text(json.dumps(stable, indent=2))
        if not stable["passed"]:
            raise RuntimeError("Registered hand failed independent final-reference acceptance")
        report.update(
            target_profile=revision["target_profile"],
            source_rotation_revision=revision,
            original_reference_passed=raw["passed"],
            independent_kinematic_and_video_passed=True,
        )
        (Path(result) / "replacement_report.json").write_text(json.dumps(report, indent=2))
        return report

    output = _runner_base.replace_pipeline_outputs(args, root, original_retarget, replacement_fn=replacement)
    for path in output.glob("*_ik.npz"):
        ep = path.name[: -len("_ik.npz")]
        result = output / "_render" / ep
        reference = output / "_references" / ep
        stable = json.loads((result / "stable_reference_validation.json").read_text())
        assert stable["passed"]
        raw = np.load(path)
        metadata = {k: raw[k] for k in raw.files}
        motor = np.load(result / "motor_actions.npz")
        quality = np.load(result / "ik_quality.npz")
        metadata.update(
            state=motor["motor_state"].astype(np.float32),
            state_names=motor["motor_names"],
            physical_state_names=motor["physical_state_names"],
            physical_from_motor_matrix=motor["physical_from_motor_matrix"],
            physical_from_motor_offset=motor["physical_from_motor_offset"],
            **{k: quality[k] for k in ["ik_ok", "ik_err_pos", "ik_err_rot"]},
        )
        np.savez_compressed(path, **metadata)
    return output
