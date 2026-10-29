#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed comparison of paired full-model GPU diagnostics, not paper acceptance."""

import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--transplant", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-single-run",
        action="store_true",
        help="Only for labelled >=100-step deterministic diagnostic; no band claim",
    )
    args = p.parse_args()
    a, b = args.base.resolve(), args.transplant.resolve()
    if a == b:
        raise ValueError("Cannot compare a run to itself")
    ma, mb = read(a / "manifest.json"), read(b / "manifest.json")
    if ma["side"] != "base" or mb["side"] != "loongforge":
        raise ValueError("Wrong run side identities")
    ma.setdefault("deterministic", False)
    mb.setdefault("deterministic", False)
    for k in [
        "steps",
        "repeats",
        "seed",
        "deterministic",
        "global_batch",
        "engine",
        "autocast",
        "grad_clip",
        "lr",
        "betas",
        "weight_decay",
        "warmup_steps",
        "schedule_steps",
        "dataset_roots",
        "dependencies",
    ]:
        if ma[k] != mb[k]:
            raise ValueError(f"Mismatched manifest {k}")
    single_diagnostic = (
        args.allow_single_run and ma["repeats"] == 1 and ma["steps"] >= 100 and ma["deterministic"]
    )
    if ma["repeats"] < 3 and not single_diagnostic:
        raise ValueError(
            "Requires >=3 repeats per side (or explicit deterministic >=100-step single diagnostic)"
        )
    checks = {}
    # LoongForge carries a logging-only sample_index key; numerical inputs must
    # still match by every other key, shape, dtype and complete bytes.
    inputs = []
    for root in [a, b]:
        rows = read(root / "input_fingerprints.json")
        for row in rows:
            row["tensors"]["tensors"].pop("sample_index", None)
            row["tensors"].pop("sha256", None)
        inputs.append(rows)
    checks["numerical_input_tensors_exact"] = inputs[0] == inputs[1]
    for name in [
        "asset_sha256.json",
        "initial_all_parameters.json",
        "frozen_before.json",
        "frozen_after.json",
        "module_modes.json",
        "placement.json",
    ]:
        checks[name] = read(a / name) == read(b / name)
    for run in [a, b]:
        v = read(run / "verification.json")
        if not v["frozen_unchanged"] or not v["full_frozen_hash"]:
            raise ValueError("Missing complete frozen verification")
    arrays = []
    for root in [a, b]:
        reps = []
        for rep in range(1, ma["repeats"] + 1):
            run = root / f"run{rep}"
            rows = [
                json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines() if s.strip()
            ]
            if [r["step"] for r in rows] != list(range(1, ma["steps"] + 1)):
                raise ValueError(f"Missing/duplicate/nonordered steps: {run}")
            for row in rows:
                if not all(
                    math.isfinite(row[k])
                    for k in ["loss", "action_loss", "video_loss", "grad_norm", "lr", "lr_used"]
                ):
                    raise ValueError("Nonfinite metric")
            if read(run / "parameter_delta.json")["changed"] <= 0:
                raise ValueError("No parameter updates")
            reps.append(rows)
        arrays.append(reps)
    # Bands below describe only these executed steps; never count them as the
    # plan's first50 band or 100/200-step integration acceptance.
    per_rep = []
    for rep in range(ma["repeats"]):
        ra, rb = a / f"run{rep + 1}", b / f"run{rep + 1}"
        diff = [
            abs(x["action_loss"] - y["action_loss"]) for x, y in zip(arrays[0][rep], arrays[1][rep])
        ]
        per_rep.append(
            {
                "repeat": rep + 1,
                "loss_mae": sum(diff) / len(diff),
                "loss_max_abs": max(diff),
                "final_parameters_exact": read(ra / "final_parameters.json")
                == read(rb / "final_parameters.json"),
                "first_grad_exact": read(ra / "grad_step000001.json")
                == read(rb / "grad_step000001.json"),
                "last_grad_exact": read(ra / f"grad_step{ma['steps']:06d}.json")
                == read(rb / f"grad_step{ma['steps']:06d}.json"),
                "grad_norm_max_abs": max(
                    abs(x["grad_norm"] - y["grad_norm"])
                    for x, y in zip(arrays[0][rep], arrays[1][rep])
                ),
                "lr_exact": all(
                    x["lr"] == y["lr"] and x["lr_used"] == y["lr_used"]
                    for x, y in zip(arrays[0][rep], arrays[1][rep])
                ),
                "timesteps_exact": all(
                    x["action_timestep"] == y["action_timestep"]
                    and x["action_weight"] == y["action_weight"]
                    for x, y in zip(arrays[0][rep], arrays[1][rep])
                ),
            }
        )
    bands = [
        max(
            max(rep[i]["action_loss"] for rep in reps) - min(rep[i]["action_loss"] for rep in reps)
            for i in range(ma["steps"])
        )
        for reps in arrays
    ]
    mean_deltas = [
        abs(
            sum(rep[i]["action_loss"] for rep in arrays[0]) / ma["repeats"]
            - sum(rep[i]["action_loss"] for rep in arrays[1]) / ma["repeats"]
        )
        for i in range(ma["steps"])
    ]
    observed_mean_within_band = all(d <= max(bands) for d in mean_deltas)
    result = {
        "observed_step_mean_abs_deltas": mean_deltas,
        "observed_short_mean_within_band": observed_mean_within_band
        if ma["repeats"] >= 3
        else None,
        "scope": ma["engine"],
        "steps": ma["steps"],
        "repeats_per_side": ma["repeats"],
        "global_batch": ma["global_batch"],
        "checks": checks,
        "per_repeat": per_rep,
        "observed_loss_range_base": bands[0] if ma["repeats"] >= 3 else None,
        "observed_loss_range_transplant": bands[1] if ma["repeats"] >= 3 else None,
        "run_to_run_band_measured": ma["repeats"] >= 3,
        "band_scope": "warm-process full trainable reset + RNG reset; only executed steps, NOT plan first50 unless >=50",
        "exact_diagnostic_pass": all(checks.values())
        and all(
            r["loss_max_abs"] == 0
            and r["final_parameters_exact"]
            and r["first_grad_exact"]
            and r["last_grad_exact"]
            and r["grad_norm_max_abs"] == 0
            and r["lr_exact"]
            and r["timesteps_exact"]
            for r in per_rep
        ),
        "canonical_zero2_pass": False,
        "full_plan_acceptance": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["exact_diagnostic_pass"] else 1)


if __name__ == "__main__":
    main()
