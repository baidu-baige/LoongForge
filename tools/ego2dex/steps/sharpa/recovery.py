# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real registered-hand global v28/v29 recovery on the native Sharpa backend.

The algorithms are hands.global_initializer_v28/v29.initialize. Old Sharpa
coupled_mapping_v28/v29 are not substitutes and are never called here. Native
v53/v54 may refine a rejected initialization, but only fresh native/model,
command, camera/workspace and raster checks may select any result.
"""

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np


def policy():
    mode = os.environ.get("EGO2ROBOT_HAND_INITIALIZER", "v28-v29")
    if mode not in ("v28-v29", "v28", "off"):
        raise ValueError("EGO2ROBOT_HAND_INITIALIZER must be v28-v29, v28 or off")
    return mode


def _write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _source(module):
    path = Path(module.__file__).resolve()
    return dict(
        module=module.__name__, source_file=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def new_session(output):
    """One task owns one session, including a later view-rejection fallback.

    Sharing this object between both task-mapping recovery hooks guarantees
    that each actual global initializer is executed at most once per task.
    Skipped/disabled paths still emit explicit receipts with algorithm hashes.
    """
    from . import initializer_backend

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mode = policy()
    reports, modules = {}, {}
    for label in ("v28", "v29"):
        module = importlib.import_module("steps.hands.global_initializer_" + label)
        modules[label] = module
        disabled = mode == "off" or (label == "v29" and mode != "v28-v29")
        reports[label] = dict(
            policy=mode,
            attempted=False,
            recovery_attempted=False,
            initializer_completed=False,
            initializer_invocations=0,
            initializer_output_validated=False,
            native_and_pixel_candidate_passed=False,
            selected="baseline",
            reason="disabled" if disabled else "not_needed",
            algorithm=_source(module),
            algorithm_callable=module.__name__ + ".initialize",
            model_backend=_source(initializer_backend),
            acceptance_gates_unchanged=True,
        )
        _write(output / (label + "_recovery.json"), reports[label])
    return dict(output=str(output.resolve()), mode=mode, attempted=set(), reports=reports, modules=modules)


def _protected_copy(candidate):
    """Post-solvers may change their inputs; never share baseline arrays/model."""
    clone = dict(candidate)
    clone["model"] = copy.copy(candidate["model"])
    clone["qpos"] = np.array(candidate["qpos"], copy=True)
    clone["record"] = dict(candidate.get("record", {}))
    return clone


def _snapshot(result):
    model = result["model"]
    return dict(
        qpos=np.array(result["qpos"], copy=True),
        ranges=np.array(model.jnt_range, copy=True),
        commands=np.array(model.actuator_ctrlrange, copy=True),
        flags=int(model.opt.disableflags),
    )


def _assert_unchanged(result, saved):
    # A mutation is an integration bug, not a numerical recovery rejection.
    np.testing.assert_array_equal(result["qpos"], saved["qpos"])
    np.testing.assert_array_equal(result["model"].jnt_range, saved["ranges"])
    np.testing.assert_array_equal(result["model"].actuator_ctrlrange, saved["commands"])
    assert int(result["model"].opt.disableflags) == saved["flags"], "Recovery mutated baseline collision flags"


def recover(
    result,
    times,
    P,
    R,
    masks,
    ref,
    manifest,
    points,
    tips,
    wristP,
    wristR,
    before,
    output,
    stage=None,
    force=False,
    *,
    session=None,
):
    """Retain baseline unless a fresh all-frame native and pixel audit passes.

    A physically passed baseline normally bypasses recovery. ``force`` is for
    the native scene/raster rejection hook only. It never bypasses candidate
    acceptance and never permits a second execution in a task session.
    """
    from . import initializer_backend as backend
    from .candidate_validation import check

    output = Path(output)
    session = new_session(output) if session is None else session
    if session["output"] != str(output.resolve()) or session["mode"] != policy():
        raise ValueError("Recovery session/output/policy identity changed during one task")
    mode = session["mode"]
    if mode == "off" or result.get("record", {}).get("native_and_pixel_acceptance_passed"):
        return result
    if result.get("record", {}).get("passed") and not force:
        return result
    times = np.asarray(times, dtype=float)
    if len(times) < 4:
        for label in ("v28", "v29"):
            if session["reports"][label]["reason"] != "disabled":
                session["reports"][label]["reason"] = "fewer_than_four_frames_baseline_retained"
                _write(output / (label + "_recovery.json"), session["reports"][label])
        return result
    saved = _snapshot(result)

    def attempt(label):
        if label in session["attempted"]:
            return None
        session["attempted"].add(label)
        module = session["modules"][label]
        report = session["reports"][label]
        report.update(
            recovery_attempted=True,
            started=time.time(),
            reason="baseline_requires_recovery",
            algorithm=_source(module),
            initialization_checks=[],
            post_refinement_checks=[],
        )
        path = output / (label + "_recovery.json")
        root = output / (label + "_recovery")
        root.mkdir(parents=True, exist_ok=True)
        _write(path, report)

        def assess(candidate, folder, candidate_label, *, post=False):
            checked, audit = check(
                candidate, times, P, R, masks, ref, manifest, points, before, folder, label=candidate_label, stage=stage
            )
            summary = dict(
                stage=candidate_label,
                passed=bool(audit["passed"]),
                checks=audit["checks"],
                acceptance_record=checked["record"],
                proof=str(Path(folder) / "native_acceptance.json"),
            )
            report["post_refinement_checks" if post else "initialization_checks"].append(summary)
            if audit["passed"]:
                report["native_and_pixel_candidate_passed"] = True
                report["accepted_candidate_stage"] = candidate_label
            _write(path, report)
            return checked

        def select(candidate, candidate_label):
            if not (
                candidate["record"].get("passed") and candidate["record"].get("native_and_pixel_acceptance_passed")
            ):
                raise AssertionError("Recovery selection lacks fresh native/pixel acceptance")
            _assert_unchanged(result, saved)
            report.update(
                selected=label,
                reason="fresh_native_and_all_frame_pixel_gates_passed",
                selected_candidate_stage=candidate_label,
                candidate_record=candidate["record"],
            )
            _write(path, report)
            return candidate

        try:
            if stage:
                stage(label + "_registered_global_initializer_with_sharpa_backend")
            source = _protected_copy(result)
            projected_seed = backend.project_seed(source["model"], source["qpos"], times)
            # This is the real registered-hand initializer, with its unchanged
            # sparse residual/Jacobian/LSMR algorithm and default iteration budget.
            report.update(attempted=True, initializer_invocations=1)
            _write(path, report)
            spec, model, q = module.initialize(
                source["model"], projected_seed, times, P, R, masks, root / "initializer", backend=backend
            )
            report["initializer_completed"] = True
            _write(path, report)
            if np.asarray(q).shape != np.asarray(result["qpos"]).shape or not np.isfinite(q).all():
                raise ValueError("Global initializer returned a malformed Sharpa trajectory")
            if label == "v28":
                before_ids = np.r_[
                    backend.indices(source["model"], "left")["finger_qadr"],
                    backend.indices(source["model"], "right")["finger_qadr"],
                ]
                after_ids = [
                    int(model.joint(source["model"].joint(int(address)).name).qposadr[0]) for address in before_ids
                ]
                if not np.array_equal(np.asarray(q)[:, after_ids], projected_seed[:, before_ids]):
                    raise RuntimeError("v28 changed a finger coordinate after canonical feasible seed projection")
                report["all_44_projected_finger_coordinates_preserved_exactly"] = True
            report.update(
                initializer_output_validated=True,
                input_seed_projection_changed_values=int(np.count_nonzero(projected_seed != result["qpos"])),
                input_seed_projection_changed_columns=int(
                    np.count_nonzero(np.any(projected_seed != result["qpos"], axis=0))
                ),
            )
            initial = dict(
                spec=spec,
                model=model,
                qpos=q,
                record=dict(label=label + "_global_initializer", passed=False, initializer_only=True),
            )
            checked = assess(initial, root / "initializer_acceptance", label + "_initializer")
            if checked["record"]["passed"]:
                report["post_refinement_reason"] = "not_needed_already_feasible_initializer_retained"
                return select(checked, label + "_initializer")

            # Native model-specific post optimizers are explicit refinements,
            # not renamed v28/v29 algorithms. Their legacy strict/proxy pass
            # flags do not control this recovery's fresh measured-view gates.
            from . import coupled_mapping_v53 as native_place
            from . import coupled_mapping_v54 as native_polish

            report["post_refinement_available_modules"] = [_source(native_place), _source(native_polish)]
            report["post_refinement_invocations"] = []
            sample = np.unique(np.r_[np.arange(0, len(times), 3), len(times) - 1]).astype(int)
            if len(sample) < 4:
                sample = np.arange(len(times))
            common = dict(
                initialization="existing",
                optimize_fingers=True,
                angular_baseline=before,
                camera_K=ref["intrinsics"],
                image_shape=(manifest["height"], manifest["width"]),
            )
            private = _protected_copy(initial)
            if stage:
                stage(label + "_native_sharpa_post_placement")
            report["post_refinement_invocations"].append(_source(native_place))
            _write(path, report)
            placed = native_place.remap_arms(
                backend.prepare_model(private["model"]),
                private["qpos"][sample].copy(),
                times[sample],
                root / "native_post_placement",
                target_positions=P[sample],
                target_rotations=R[sample],
                finger_targets=tips[sample],
                palm_reference_positions=wristP[sample],
                palm_reference_rotations=wristR[sample],
                target_masks=masks[sample],
                head_poses=ref["obs_head_pose"][sample],
                **common,
            )
            full_q = np.stack(
                [np.interp(times, times[sample], placed["qpos"][:, column]) for column in range(placed["model"].nq)],
                axis=1,
            )
            full = dict(placed, qpos=full_q)
            placed_check = assess(full, root / "post_placement_acceptance", label + "_native_post_placement", post=True)
            if placed_check["record"]["passed"]:
                report["post_refinement_reason"] = "feasible_full_frame_placement_retained_before_optional_polish"
                return select(placed_check, label + "_native_post_placement")
            if stage:
                stage(label + "_native_sharpa_post_polish")
            report["post_refinement_invocations"].append(_source(native_polish))
            _write(path, report)
            private = _protected_copy(full)
            polished = native_polish.remap_arms(
                backend.prepare_model(private["model"]),
                private["qpos"],
                times,
                root / "native_post_polish",
                target_positions=P,
                target_rotations=R,
                finger_targets=tips,
                palm_reference_positions=wristP,
                palm_reference_rotations=wristR,
                target_masks=masks,
                head_poses=ref["obs_head_pose"],
                **common,
            )
            polished_check = assess(polished, root / "post_polish_acceptance", label + "_native_post_polish", post=True)
            if polished_check["record"]["passed"]:
                return select(polished_check, label + "_native_post_polish")
            report.update(
                reason="fresh_native_or_pixel_gate_rejected_baseline_retained",
                candidate_record=polished_check["record"],
            )
            return None
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            # Do not swallow assertion/programming bugs, missing names, I/O
            # failures or arbitrary exceptions as ordinary numerical rejection.
            report.update(
                reason="numerical_or_model_recovery_failure_baseline_retained",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None
        finally:
            _assert_unchanged(result, saved)
            report["finished"] = time.time()
            _write(path, report)

    for label in ("v28", "v29") if mode == "v28-v29" else ("v28",):
        candidate = attempt(label)
        if candidate is not None:
            _assert_unchanged(result, saved)
            return candidate
    _assert_unchanged(result, saved)
    return result
