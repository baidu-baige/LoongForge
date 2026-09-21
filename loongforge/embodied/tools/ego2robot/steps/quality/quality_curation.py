#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ego2Robot L1/L2/L3 quality curation.

This step reads the per-frame quality artifacts written by
``steps.robot_retarget`` and produces a curation manifest without modifying
the original retarget output.  L1 is pipeline-internal validation, L2 is a
dataset-level robust action/dynamics filter, and L3 consumes cached VLM
judgements.

Example (with an SGLang server already running):
    python cli.py quality \
        --retarget_dir out/06_retarget \
        --state_dir out/02_align \
        --output_dir out/quality_curation

L3 results can be supplied as a JSON object keyed by episode name:
    {"episode_a": {"is_consistent": true, "confidence": 0.92,
                    "reasoning": "..."}}

The output ``manifest.parquet`` and ``*_frame_mask.npz`` files are intended to
be consumed by the LeRobot writer.  No source video or IK file is overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "ego2robot-quality-v1"
DEFAULT_VLM_MAX_FRAMES = 32
VLM_MODEL_ENV = "EGO2ROBOT_VLM_MODEL"
DEFAULT_MIN_VALID_RUN_FRAMES = 9  # floor(0.3 * 30 fps), as in Ego2Robot.
ALOHA_BASELINE_SELF_CONTACTS = 8
ALOHA_BASELINE_SELF_PENETRATION = 0.02163086


def _scalar(value: Any, default: Any = None) -> Any:
    """Convert a numpy scalar/0-d array to a Python value."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value
    return value


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _fit_array(value: Any, n: int, dtype: Any, default: Any) -> np.ndarray:
    """Return a length-n array, padding missing quality fields conservatively."""
    if value is None:
        return np.full(n, default, dtype=dtype)
    arr = np.asarray(value).reshape(-1)
    out = np.full(n, default, dtype=dtype)
    out[: min(n, len(arr))] = arr[:n]
    return out


def _fit_matrix(value: Any, n: int, width: int, default: Any) -> np.ndarray:
    """Return an n-by-width matrix without repeating missing columns."""
    out = np.full((n, width), default, dtype=np.asarray(default).dtype if np.asarray(default).ndim else None)
    if value is None:
        return out
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return out
    rows, cols = min(n, arr.shape[0]), min(width, arr.shape[1])
    out[:rows, :cols] = arr[:rows, :cols]
    return out


def _episode_names(retarget_dir: Path, requested: list[str] | None) -> list[str]:
    names = sorted(p.name[:-7] for p in retarget_dir.glob("*_ik.npz")
                   if (retarget_dir / f"{p.name[:-7]}_quality.npz").exists())
    if requested is None:
        return names
    requested_set = set(requested)
    missing = sorted(requested_set - set(names))
    if missing:
        raise FileNotFoundError(
            f"episodes missing *_ik.npz or *_quality.npz in {retarget_dir}: {missing}")
    return [name for name in names if name in requested_set]


def _task_text(state_dir: Path | None, episode: str) -> str:
    if state_dir is None:
        return "manipulation"
    path = state_dir / f"{episode}.npz"
    if not path.exists():
        return "manipulation"
    try:
        data = _read_npz(path)
        raw = _scalar(data.get("annotations"), [])
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if raw and isinstance(raw[0], dict):
            return str(raw[0].get("text") or raw[0].get("description") or "manipulation")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return "manipulation"


def _state_action(ik: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(ik.get("state", ik.get("qpos")), dtype=np.float64)
    if state.ndim != 2:
        raise ValueError(f"IK state must be a 2-D array, got {state.shape}")
    action = np.zeros_like(state)
    if len(state) >= 2:
        action[:-1] = np.diff(state, axis=0)
        action[-1] = action[-2]
    return state, action


def _interior_short_runs(valid: np.ndarray, min_length: int) -> np.ndarray:
    """Erode short valid runs that are surrounded by invalid regions."""
    result = valid.copy()
    n = len(valid)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        start = i
        while i < n and valid[i]:
            i += 1
        end = i
        if start > 0 and end < n and end - start < min_length:
            result[start:end] = False
    return result


def _l1(ik: dict[str, np.ndarray], quality: dict[str, np.ndarray],
        min_valid_run_frames: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Compute paper L1 validity and stability erosion masks."""
    lengths = []
    for key in ("state", "qpos"):
        if key in ik:
            lengths.append(len(ik[key]))
            break
    lengths.extend(len(quality[key]) for key in quality
                   if np.asarray(quality[key]).ndim > 0)
    n = min(lengths) if lengths else 0
    if n <= 0:
        raise ValueError("episode has no frame-aligned IK/quality arrays")

    err_pos = _fit_matrix(ik.get("ik_err_pos"), n, 2, np.inf)

    hand = _fit_array(quality.get("hand_detected"), n, bool, False)
    rendered = _fit_array(quality.get("rendered"), n, bool, False)
    pixels = _fit_array(quality.get("robot_pixel_count"), n, np.int64, 0)
    mask_ratio = _fit_array(quality.get("robot_mask_ratio"), n, np.float64, 1.0)
    self_collision = _fit_array(quality.get("self_collision"), n, bool, True)
    self_contact_count = _fit_array(quality.get("self_contact_count"), n, np.int64, 0)
    self_penetration = _fit_array(quality.get("self_penetration"), n, np.float64, 0.0)
    cross_contacts = _fit_array(
        quality.get("cross_arm_contact_count"), n, np.int64, np.iinfo(np.int64).max)

    # Older retarget artifacts counted fixed source-model mesh overlaps as
    # self-collisions. Remove only a verified deterministic baseline; contacts
    # or penetration beyond that baseline remain quality failures.
    robot_type = str(_scalar(quality.get("robot_type"), ""))
    baseline_collision = np.zeros(n, dtype=bool)
    if robot_type == "aloha_agilex" and n:
        baseline_only = (
            (self_contact_count == ALOHA_BASELINE_SELF_CONTACTS)
            & np.isclose(
                self_penetration, ALOHA_BASELINE_SELF_PENETRATION,
                rtol=0.0, atol=1e-6)
        )
        # Require at least one exact baseline-only frame before correcting a
        # legacy episode. This keeps already-corrected artifacts unchanged.
        if baseline_only.any():
            baseline_collision = (
                (self_contact_count >= ALOHA_BASELINE_SELF_CONTACTS)
                & (self_penetration >=
                   ALOHA_BASELINE_SELF_PENETRATION - 1e-6)
            )
            remaining_contacts = (
                self_contact_count - ALOHA_BASELINE_SELF_CONTACTS)
            remaining_penetration = (
                self_penetration - ALOHA_BASELINE_SELF_PENETRATION)
            self_collision[baseline_collision] = (
                (remaining_contacts[baseline_collision] > 0)
                | (remaining_penetration[baseline_collision] > 1e-6)
            )
    elif robot_type == "kinova_gen3" and n and np.all(self_collision):
        baseline_collision = (
            np.ptp(self_contact_count) == 0
            and self_contact_count[0] > 0
            and np.ptp(self_penetration) <= 1e-6
            and self_penetration[0] > 0.0
        )
        if baseline_collision:
            self_collision = np.zeros(n, dtype=bool)

    # Appendix A.5 defines L1 IK validity solely as finite positional tracking
    # error below 0.05 m. ``ik_ok`` is a full-pose flag in current retarget
    # artifacts and would add an undocumented orientation-error gate.
    position_ok = np.isfinite(err_pos).all(axis=1) & (err_pos.max(axis=1) < 0.05)
    valid = (hand & position_ok
             & rendered & (pixels > 0) & (~self_collision)
             & (cross_contacts <= 1) & np.isfinite(mask_ratio)
             & (mask_ratio <= 0.70))
    eroded = _interior_short_runs(valid, max(int(min_valid_run_frames), 1))
    reasons = {
        "hand_missing": int((~hand).sum()),
        "ik_position_error": int((~position_ok).sum()),
        "not_rendered_or_empty": int((~(rendered & (pixels > 0))).sum()),
        "self_collision": int(self_collision.sum()),
        "baseline_self_collision_ignored": int(baseline_collision.sum()),
        "cross_arm_contact": int((cross_contacts > 1).sum()),
        "mask_too_large": int((mask_ratio > 0.70).sum()),
        "short_valid_run_eroded": int((valid & ~eroded).sum()),
    }
    return valid, eroded, reasons


def _quantile_fence(values: np.ndarray, valid: np.ndarray, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper robust fences for each action dimension."""
    d = values.shape[1]
    q1 = np.zeros(d, dtype=np.float64)
    q99 = np.zeros(d, dtype=np.float64)
    for j in range(d):
        column = values[valid, j]
        column = column[np.isfinite(column)]
        if len(column) == 0:
            q1[j], q99[j] = -np.inf, np.inf
        else:
            q1[j], q99[j] = np.quantile(column, [0.01, 0.99])
    spread = q99 - q1
    # A constant action dimension should not reject tiny floating point noise.
    tol = np.maximum(spread * multiplier, 1e-8)
    return q1 - tol, q99 + tol


def _dynamic_values(state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return residual, acceleration and jerk arrays aligned to frames."""
    residual = action - np.median(action, axis=0, keepdims=True)
    acceleration = np.zeros_like(action)
    jerk = np.zeros_like(action)
    if len(action) >= 2:
        acceleration[1:] = np.diff(action, axis=0)
    if len(action) >= 3:
        jerk[2:] = np.diff(acceleration, axis=0)[1:]
    return residual, acceleration, jerk


def _dynamic_threshold(values: list[np.ndarray], masks: list[np.ndarray], quantile: float) -> np.ndarray:
    d = values[0].shape[1]
    threshold = np.full(d, np.inf, dtype=np.float64)
    for j in range(d):
        merged = np.concatenate([v[m, j] for v, m in zip(values, masks)])
        merged = np.abs(merged[np.isfinite(merged)])
        if len(merged):
            threshold[j] = max(float(np.quantile(merged, quantile)), 1e-8)
    return threshold


def _l2(records: list[dict[str, Any]], multiplier: float, sudden_quantile: float) -> dict[str, Any]:
    """Apply dataset-level Q1/Q99 and sudden-change filters by morphology/dim."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["robot_type"], record["action"].shape[1])
        groups.setdefault(key, []).append(record)

    for group_records in groups.values():
        values = [r["action"] for r in group_records]
        masks = [r["l1_valid"] for r in group_records]
        merged = np.concatenate(values, axis=0)
        merged_mask = np.concatenate(masks, axis=0)
        lower, upper = _quantile_fence(merged, merged_mask, multiplier)
        residuals, accels, jerks = [], [], []
        for r in group_records:
            residual, accel, jerk = _dynamic_values(r["state"], r["action"])
            residuals.append(residual)
            accels.append(accel)
            jerks.append(jerk)
        thresholds = {
            "residual": _dynamic_threshold(residuals, masks, sudden_quantile),
            "acceleration": _dynamic_threshold(accels, masks, sudden_quantile),
            "jerk": _dynamic_threshold(jerks, masks, sudden_quantile),
        }
        for r, residual, accel, jerk in zip(group_records, residuals, accels, jerks):
            q_outlier = ((r["action"] < lower) | (r["action"] > upper)).any(axis=1)
            sudden = ((np.abs(residual) > thresholds["residual"]) |
                      (np.abs(accel) > thresholds["acceleration"]) |
                      (np.abs(jerk) > thresholds["jerk"])).any(axis=1)
            r["l2_q_outlier"] = q_outlier
            r["l2_sudden_change"] = sudden
            r["l2_valid"] = r["l1_valid"] & ~q_outlier & ~sudden
            r["l2_thresholds"] = {
                "q01": lower.tolist(), "q99": upper.tolist(),
                "residual": thresholds["residual"].tolist(),
                "acceleration": thresholds["acceleration"].tolist(),
                "jerk": thresholds["jerk"].tolist(),
            }
    return {"groups": len(groups)}


def _load_vlm_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("--vlm_results must contain a JSON object keyed by episode")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def _prepare_records(retarget: Path, state_path: Path | None,
                     episodes: list[str] | None, min_valid_run_frames: int,
                     max_invalid_ratio: float, q_fence_multiplier: float,
                     sudden_quantile: float) -> list[dict[str, Any]]:
    """Compute L1/L2 once and return records ready for an L3 decision."""
    if not 0.0 <= max_invalid_ratio < 1.0:
        raise ValueError("max_invalid_ratio must be in [0, 1)")
    episode_names = _episode_names(retarget, episodes)
    if not episode_names:
        raise FileNotFoundError(
            f"no matching *_ik.npz and *_quality.npz files in {retarget}")

    records = []
    for episode in episode_names:
        ik = _read_npz(retarget / f"{episode}_ik.npz")
        quality = _read_npz(retarget / f"{episode}_quality.npz")
        state, action = _state_action(ik)
        l1, l1_eroded, reasons = _l1(ik, quality, min_valid_run_frames)
        # L1 uses the shortest frame-aligned artifact, so keep L2 aligned to it.
        state, action = state[:len(l1_eroded)], action[:len(l1_eroded)]
        robot_type = str(_scalar(
            ik.get("robot_type"), _scalar(quality.get("robot_type"), "unknown")))
        records.append({
            "episode": episode, "robot_type": robot_type,
            "task": _task_text(state_path, episode),
            "state": state, "action": action,
            "l1_valid_raw": l1, "l1_valid": l1_eroded,
            "l1_valid_eroded": l1_eroded,
            "l2_q_outlier": np.zeros(len(state), dtype=bool),
            "l2_sudden_change": np.zeros(len(state), dtype=bool),
            "l2_valid": l1_eroded.copy(), "l1_reasons": reasons,
            "vlm": None, "max_invalid_ratio": max_invalid_ratio,
            "l3_keep": False,
        })
    _l2(records, q_fence_multiplier, sudden_quantile)
    return records


def _apply_l3(records: list[dict[str, Any]],
              vlm: dict[str, dict[str, Any]], require_vlm: bool) -> None:
    """Attach L3 decisions and compute the final per-episode keep flags."""
    for record in records:
        result = vlm.get(record["episode"])
        if result is not None and "is_consistent" not in result:
            raise ValueError(
                f"VLM result for {record['episode']} lacks is_consistent")
        record["vlm"] = result
        record["l3_keep"] = result is not None and bool(result["is_consistent"])

        invalid_ratio = float((~record["l2_valid"]).mean())
        l2_keep = (invalid_ratio <= record["max_invalid_ratio"]
                   and bool(record["l2_valid"].any()))
        has_l3_gate = result is not None or require_vlm
        record["keep_episode"] = bool(
            l2_keep and record["l3_keep"] if has_l3_gate else l2_keep)
        reasons = []
        if not l2_keep:
            reasons.append("invalid_ratio>threshold_or_no_valid_frames")
        if result is None and require_vlm:
            reasons.append("vlm_result_missing")
        elif result is not None and not record["l3_keep"]:
            reasons.append("vlm_inconsistent")
        record["drop_reason"] = ",".join(reasons)


def _write_manifest(records: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for index, r in enumerate(records):
        vlm = r["vlm"]
        invalid = ~r["l2_valid"]
        vlm_status = "pending" if vlm is None else ("consistent" if vlm["is_consistent"] else "inconsistent")
        rows.append({
            "episode": r["episode"],
            "episode_index_original": index,
            "robot_type": r["robot_type"],
            "task": r["task"],
            "num_frames": int(len(r["l2_valid"])),
            "l1_valid_frames": int(r["l1_valid"].sum()),
            "l1_valid_frames_raw": int(r["l1_valid_raw"].sum()),
            "l2_invalid_frames": int(invalid.sum()),
            "invalid_ratio": float(invalid.mean()),
            "l1_keep": bool(r["l1_valid"].any()),
            "l2_keep": bool(invalid.mean() <= r["max_invalid_ratio"] and r["l2_valid"].any()),
            "vlm_status": vlm_status,
            "vlm_is_consistent": None if vlm is None else bool(vlm["is_consistent"]),
            "vlm_confidence": None if vlm is None else float(vlm.get("confidence", float("nan"))),
            "vlm_reasoning": "" if vlm is None else str(vlm.get("reasoning", "")),
            "vlm_backend": "" if vlm is None else str(vlm.get("backend", "unknown")),
            "vlm_model_name": "" if vlm is None else str(vlm.get("model_name", "unknown")),
            "vlm_sample_fps": None if vlm is None else float(vlm.get("sample_fps", 4.0)),
            "vlm_sampled_frames": None if vlm is None else int(vlm.get("sampled_frames", 0)),
            "keep_episode": bool(r["keep_episode"]),
            "drop_reason": r["drop_reason"],
        })
        np.savez_compressed(
            output_dir / f"{r['episode']}_frame_mask.npz",
            l1_valid=r["l1_valid"],
            l1_valid_raw=r["l1_valid_raw"],
            l1_valid_eroded=r["l1_valid_eroded"],
            l2_q_outlier=r["l2_q_outlier"],
            l2_sudden_change=r["l2_sudden_change"],
            valid=r["l2_valid"],
        )
        with (output_dir / f"{r['episode']}_vlm.json").open("w", encoding="utf-8") as handle:
            json.dump(vlm or {"status": "pending"}, handle, indent=2, ensure_ascii=False)

    table = pa.table({key: [row[key] for row in rows] for key in rows[0]}) if rows else pa.table({})
    pq.write_table(table, output_dir / "manifest.parquet")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "episodes": len(rows),
        "kept_episodes": sum(row["keep_episode"] for row in rows),
        "dropped_episodes": sum(not row["keep_episode"] for row in rows),
        "pending_vlm": sum(row["vlm_status"] == "pending" for row in rows),
        "kept_frames": int(sum(np.load(output_dir / f"{r['episode']}_frame_mask.npz")["valid"].sum()
                                for r in records)),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def _write_vlm_requests(records: list[dict[str, Any]], output_dir: Path,
                        retarget_dir: Path) -> None:
    """Write deterministic requests for an external VLM batch runner."""
    prompt = (
        "You are evaluating whether a manipulation video matches its text description.\n"
        "This is a ROBOT manipulation dataset. Hand refers to the robot gripper/end-effector.\n"
        "Many tasks use FAKE or SIMULATED objects as stand-ins for real objects. This is EXPECTED.\n"
        "Flag MAJOR MISMATCHES: wrong action type, wrong object category, wrong target location, "
        "or failed execution. Be tolerant of fake/toy objects, minor appearance variations, "
        "small spatial deviations, and different grasping approaches.\n"
        "Respond only as JSON: {\"is_consistent\": true/false, \"confidence\": 0.0-1.0, "
        "\"reasoning\": \"...\"}."
    )
    requests = []
    for r in records:
        requests.append({
            "episode": r["episode"],
            "video": str(retarget_dir / f"{r['episode']}_robot_on_bg.mp4"),
            "task_description": r["task"],
            "sample_fps": 4,
            "prompt": f"{prompt}\nTask Description: {r['task']}",
        })
    with (output_dir / "vlm_requests.jsonl").open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")


def curate(retarget_dir: str, output_dir: str, state_dir: str | None = None,
           episodes: list[str] | None = None, vlm_results: str | None = None,
           require_vlm: bool = False, min_valid_run_frames: int = DEFAULT_MIN_VALID_RUN_FRAMES,
           max_invalid_ratio: float = 0.60, q_fence_multiplier: float = 3.0,
           sudden_quantile: float = 0.999) -> dict[str, Any]:
    """Run L1/L2/L3 curation and return the summary dictionary."""
    retarget = Path(retarget_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = Path(state_dir) if state_dir else None
    vlm = _load_vlm_results(Path(vlm_results) if vlm_results else None)
    records = _prepare_records(
        retarget, state_path, episodes, min_valid_run_frames,
        max_invalid_ratio, q_fence_multiplier, sudden_quantile)
    _apply_l3(records, vlm, require_vlm)
    _write_manifest(records, output)
    _write_vlm_requests(records, output, retarget)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary.update({
        "l1_min_valid_run_frames": int(min_valid_run_frames),
        "l2_q_fence_multiplier": float(q_fence_multiplier),
        "l2_sudden_quantile": float(sudden_quantile),
        "l3_require_vlm": bool(require_vlm),
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _curation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "retarget_dir": args.retarget_dir,
        "state_dir": args.state_dir,
        "output_dir": args.output_dir,
        "episodes": args.episodes,
        "min_valid_run_frames": args.min_valid_run_frames,
        "max_invalid_ratio": args.max_invalid_ratio,
        "q_fence_multiplier": args.q_fence_multiplier,
        "sudden_quantile": args.sudden_quantile,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entry point: run L1/L2, invoke SGLang L3, then write final output."""
    kwargs = _curation_kwargs(args)
    if args.skip_vlm:
        if args.vlm_results:
            raise ValueError("--skip_vlm and --vlm_results cannot be used together")
        summary = curate(**kwargs, require_vlm=args.require_vlm)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    if args.vlm_results:
        summary = curate(
            **kwargs, vlm_results=args.vlm_results,
            require_vlm=args.require_vlm)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    # Build L1/L2 and requests once. L3 then mutates only the in-memory records.
    retarget = Path(args.retarget_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = _prepare_records(
        retarget, Path(args.state_dir) if args.state_dir else None,
        args.episodes, args.min_valid_run_frames, args.max_invalid_ratio,
        args.q_fence_multiplier, args.sudden_quantile)
    requests_path = output / "vlm_requests.jsonl"
    results_path = output / "vlm_results.json"
    _write_vlm_requests(records, output, retarget)

    from . import vlm_runner
    vlm_runner.run(
        requests_path=str(requests_path), output_path=str(results_path),
        server_url=args.server_url, model=args.vlm_model,
        timeout=args.vlm_timeout, max_tokens=args.vlm_max_tokens,
        concurrency=args.vlm_concurrency, retries=args.vlm_retries,
        max_frames=args.vlm_max_frames, max_edge=args.vlm_max_edge,
        jpeg_quality=args.vlm_jpeg_quality,
        overwrite=args.overwrite_vlm)

    vlm = _load_vlm_results(results_path)
    _apply_l3(records, vlm, require_vlm=True)
    _write_manifest(records, output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary.update({
        "l1_min_valid_run_frames": int(args.min_valid_run_frames),
        "l2_q_fence_multiplier": float(args.q_fence_multiplier),
        "l2_sudden_quantile": float(args.sudden_quantile),
        "l3_require_vlm": True,
        "l3_results": str(results_path),
    })
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    missing = [r["episode"] for r in records if r["vlm"] is None]
    if missing:
        error_path = results_path.with_name(results_path.stem + ".errors.json")
        raise RuntimeError(
            f"L3 has no result for {len(missing)} episode(s); rerun the same "
            f"command to resume. Details: {error_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ego2Robot L1/L2/L3 quality curation")
    parser.add_argument("--retarget_dir", required=True,
                        help="06_retarget directory containing *_ik.npz and *_quality.npz")
    parser.add_argument("--state_dir", default=None,
                        help="02_align directory used to recover task text")
    parser.add_argument("--output_dir", required=True,
                        help="quality_curation output directory")
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--vlm_results", default=None,
                        help="Use an existing L3 result JSON instead of calling SGLang")
    parser.add_argument("--skip_vlm", action="store_true",
                        help="Run only L1/L2 and leave L3 pending")
    parser.add_argument("--require_vlm", action="store_true",
                        help="With --skip_vlm/--vlm_results, fail closed on missing L3 results")
    parser.add_argument("--server_url", default="http://127.0.0.1:8000/v1")
    env_model = os.environ.get(VLM_MODEL_ENV) or None
    parser.add_argument(
        "--vlm_model", default=env_model,
        help=("SGLang model id (required for automatic L3; "
              f"or set {VLM_MODEL_ENV})"))
    parser.add_argument("--vlm_timeout", type=float, default=600.0)
    parser.add_argument("--vlm_max_tokens", type=int, default=512)
    parser.add_argument("--vlm_concurrency", type=int, default=4)
    parser.add_argument("--vlm_retries", type=int, default=2)
    parser.add_argument("--vlm_max_frames", type=int,
                        default=DEFAULT_VLM_MAX_FRAMES,
                        help=("Maximum frames uniformly sampled across each video "
                              f"(default: {DEFAULT_VLM_MAX_FRAMES})"))
    parser.add_argument("--vlm_max_edge", type=int, default=448)
    parser.add_argument("--vlm_jpeg_quality", type=int, default=85)
    parser.add_argument("--overwrite_vlm", action="store_true",
                        help="Discard cached L3 decisions and review every episode")
    parser.add_argument("--min_valid_run_frames", type=int, default=DEFAULT_MIN_VALID_RUN_FRAMES)
    parser.add_argument("--max_invalid_ratio", type=float, default=0.60)
    parser.add_argument("--q_fence_multiplier", type=float, default=3.0)
    parser.add_argument("--sudden_quantile", type=float, default=0.999)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
