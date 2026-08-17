# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""baseline read/write and tolerance comparison.

baseline file: ${EMBODIED_BASELINE_ROOT}/<chip>/<model_name>.json

The default value of EMBODIED_BASELINE_ROOT resolves to the in-repo
`tests/embodied/baseline` directory (computed from this file's location); export
EMBODIED_BASELINE_ROOT explicitly (typically via tests/embodied/config/env.sh)
to override — useful for shared out-of-repo baseline collections.

Format:
    {
      "<training_type>": [
        {"iteration": 1, "loss": ..., "grad_norm": ..., "elapsed_time_ms": ...},
        ...
      ],
      ...
    }

Here <training_type> is the top-level key, used to save baselines separately by training form (train / pretrain /
sft / lora etc.) under the same model. Callers currently all pass BASELINE_KEY = "train"; the
parameter is kept as an extension point for coexisting multiple training forms later.

Comparison strategy (aligned with the main framework's tests):
- loss-type metrics: per-iteration relative-error hard check (failure if it exceeds accuracy_relative_tolerance)
- grad_norm: hard check when check_loss_only=False
- elapsed_time_ms / throughput: soft check (only warn, no failure, if it exceeds performance_relative_tolerance)
- when the regression passes and performance overall improves beyond tolerance, automatically update the baseline's performance metrics to this run's better values
  (see update_perf_baseline; accuracy metrics are never auto-updated)
"""

import json
import math
import os
import shutil

# Skip the first few iterations for performance metrics (large compile/warmup jitter)
_PERF_WARMUP_ITERS = 3

# (metric name, whether larger is worse)
_PERF_KEYS = (("elapsed_time_ms", True), ("throughput", False))

# baseline.py lives at tests/embodied/reporting/baseline.py; two dirname() hops -> tests/embodied,
# then baseline. This keeps the embodied suite self-contained (its own baseline/ dir).
_IN_REPO_BASELINE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "baseline",
)


def _baseline_root():
    return os.environ.get("EMBODIED_BASELINE_ROOT") or _IN_REPO_BASELINE_ROOT


def baseline_path(chip, model_name):
    """Build the baseline JSON file path: ${EMBODIED_BASELINE_ROOT}/<chip>/<model_name>.json."""
    return os.path.join(_baseline_root(), chip, f"{model_name}.json")


def load_baseline(chip, model_name, training_type):
    """Read the baseline records list for the given model.

    Args:
        chip: chip model (e.g. A800 / P6K), which determines the baseline subdirectory.
        model_name: model name, corresponding to the baseline file name (without .json).
        training_type: the top-level key of the baseline JSON, used to distinguish records of
            different training forms (e.g. train / pretrain / sft / lora) within the same file.
            Actual callers currently all pass execution.executor.BASELINE_KEY = "train";
            the parameter is kept as an extension point for coexisting multiple training forms later.

    Returns:
        the records list under the corresponding training_type; returns None if the file does not exist or the key is missing.
    """
    path = baseline_path(chip, model_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(training_type)


def save_baseline(chip, model_name, training_type, records):
    """Write records into the training_type partition of the baseline file.

    Args:
        chip: chip model, which determines the baseline subdirectory.
        model_name: model name, corresponding to the baseline file name.
        training_type: the top-level key of the baseline JSON, used to distinguish training forms;
            the semantics match load_baseline, currently unified as "train".
        records: the complete records list to overwrite under this training_type.

    Behavior: read the existing JSON (if any), overwrite only data[training_type], and keep other keys.

    Returns:
        the written file path.
    """
    path = baseline_path(chip, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[training_type] = records
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _is_loss_key(key):
    return "loss" in key and key != "loss_scale"


def _rel_diff(actual, expected):
    return abs(actual - expected) / max(abs(expected), 1e-12)


def compare(actual_records, baseline_records, accuracy_tol, performance_tol,
            check_loss_only=False, max_report=10):
    """Return (failed_metrics, warnings). A non-empty failed_metrics means failure."""
    failed, warnings = [], []

    base_by_iter = {r["iteration"]: r for r in baseline_records}
    act_by_iter = {r["iteration"]: r for r in actual_records}
    common_iters = sorted(set(base_by_iter) & set(act_by_iter))
    if not common_iters:
        return ["No alignable iteration between baseline and actual result"], warnings
    if len(act_by_iter) < len(base_by_iter):
        failed.append(
            f"Not enough iterations: actual={len(act_by_iter)} < baseline={len(base_by_iter)}"
        )

    # Set of hard-check metrics: take the union over all baseline records, to avoid a metric never being
    # checked in later iterations because one iteration lacks a field (e.g. grad_norm missing during warmup).
    hard_keys = sorted(
        {k for r in baseline_records
         for k in r if _is_loss_key(k) or (not check_loss_only and k == "grad_norm")}
    )
    for key in hard_keys:
        for it in common_iters:
            expected = base_by_iter[it].get(key)
            actual = act_by_iter[it].get(key)
            if expected is None:
                continue
            if actual is None:
                failed.append(f"{key}@iter{it}: this metric is missing from the actual log")
            elif not math.isfinite(actual):
                # NaN/Inf slip past `rel > tol` (NaN comparisons are always False), so guard explicitly.
                failed.append(f"{key}@iter{it}: actual is non-finite ({actual!r})")
            elif not math.isfinite(expected):
                failed.append(f"{key}@iter{it}: baseline is non-finite ({expected!r})")
            else:
                rel = _rel_diff(actual, expected)
                if rel > accuracy_tol:
                    failed.append(
                        f"{key}@iter{it}: actual={actual:.6g} expected={expected:.6g} "
                        f"rel_diff={rel:.4f} > {accuracy_tol}"
                    )
            if len(failed) >= max_report:
                failed.append("... (more failures omitted)")
                return failed, warnings

    # Trainer-reported numerical instability signals (from metrics.jsonl): any iter with
    # nan_iterations>0 or skipped_iterations>0 means the trainer already flagged this step as
    # numerically bad; treat it as hard failure regardless of loss-diff comparison.
    for it in common_iters:
        for key in ("nan_iterations", "skipped_iterations"):
            val = act_by_iter[it].get(key)
            if val is not None and val > 0:
                failed.append(f"{key}@iter{it}: trainer reported {val} (>0)")
                if len(failed) >= max_report:
                    failed.append("... (more failures omitted)")
                    return failed, warnings

    # Performance soft check: average per-iteration time
    def _mean(records_by_iter, key):
        vals = [
            records_by_iter[it][key] for it in common_iters[_PERF_WARMUP_ITERS:]
            if key in records_by_iter[it]
        ]
        return sum(vals) / len(vals) if vals else None

    for key, worse_is_larger in _PERF_KEYS:
        base_mean = _mean(base_by_iter, key)
        act_mean = _mean(act_by_iter, key)
        if base_mean is None or act_mean is None:
            continue
        rel = (act_mean - base_mean) / max(abs(base_mean), 1e-12)
        if not worse_is_larger:
            rel = -rel
        if rel > performance_tol:
            warnings.append(
                f"{key}: actual_mean={act_mean:.2f} baseline_mean={base_mean:.2f} "
                f"degraded {rel * 100:.1f}% > {performance_tol * 100:.0f}% (soft check, warning only)"
            )

    return failed, warnings


def update_perf_baseline(chip, model_name, training_type,
                         actual_records, baseline_records, performance_tol):
    """When the regression passes and performance improves, write the baseline's performance metrics back to this run's better values.

    Args:
        chip / model_name / training_type: same semantics as load_baseline;
            training_type determines which top-level partition of the baseline JSON is written back.
        actual_records: the records produced by this run.
        baseline_records: the current baseline's records (modified in place and then flushed to disk).
        performance_tol: the performance relative tolerance, which determines the "significant improvement" threshold.

    Only performance metrics (_PERF_KEYS) are overwritten; accuracy metrics (loss / grad_norm etc.) stay unchanged.
    Update conditions (to avoid noise causing the baseline to jitter repeatedly):
    - all comparable performance metrics show no degradation;
    - and at least one improvement exceeds performance_tol.
    If satisfied, write to disk and return the baseline path, otherwise return None.
    """
    base_by_iter = {r["iteration"]: r for r in baseline_records}
    act_by_iter = {r["iteration"]: r for r in actual_records}
    common_iters = sorted(set(base_by_iter) & set(act_by_iter))

    def _mean(records_by_iter, key):
        vals = [
            records_by_iter[it][key] for it in common_iters[_PERF_WARMUP_ITERS:]
            if key in records_by_iter[it]
        ]
        return sum(vals) / len(vals) if vals else None

    improved = False
    for key, worse_is_larger in _PERF_KEYS:
        base_mean = _mean(base_by_iter, key)
        act_mean = _mean(act_by_iter, key)
        if base_mean is None or act_mean is None:
            continue
        # Never write back the baseline from non-finite means — comparisons with NaN are all False,
        # so relying on the rel checks below would silently allow polluted values through.
        if not math.isfinite(base_mean) or not math.isfinite(act_mean):
            return None
        rel = (act_mean - base_mean) / max(abs(base_mean), 1e-12)
        if not worse_is_larger:
            rel = -rel
        if rel > 0:  # any performance metric degraded -> do not update
            return None
        if -rel > performance_tol:
            improved = True
    if not improved:
        return None

    # Back up a copy before auto-writing (overwrite into .bak, keeping the most recent one for rollback).
    # Only rewrite the _PERF_KEYS outside the accuracy metrics; this mechanism prevents noise-induced
    # false improvements from polluting the comparison baseline for later runs.
    path = baseline_path(chip, model_name)
    if os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except OSError:
            pass
    for it in common_iters:
        for key, _ in _PERF_KEYS:
            if key in act_by_iter[it]:
                base_by_iter[it][key] = act_by_iter[it][key]
    return save_baseline(chip, model_name, training_type, baseline_records)
