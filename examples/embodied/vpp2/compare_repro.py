#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed action-loss MAE gate. Does not certify the plan's full acceptance."""

import argparse
import json
import math
from pathlib import Path


def rows(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        row = record.get("train", record)
        step = int(row["iteration"] if "iteration" in row else row["step"])
        if "learning_rate" not in row:
            row["learning_rate"] = row["lr"]
        if step in result:
            raise ValueError(f"duplicate optimizer step {step}: {path}")
        for key in ["video_loss", "action_loss", "learning_rate", "grad_norm"]:
            if not math.isfinite(float(row[key])):
                raise ValueError(f"nonfinite {key} at step {step}: {path}")
        if row.get("nan_iterations", 0) or row.get("skipped_iterations", 0):
            raise ValueError(f"skipped/nonfinite update at step {step}")
        result[step] = row
    return result


def compare(base, trans, steps, threshold):
    if base.resolve() == trans.resolve():
        raise ValueError("BASE and transplant must be independent run directories")
    manifests = [json.loads((root / "run_manifest.json").read_text()) for root in [base, trans]]
    if [item.get("side") for item in manifests] != ["base", "loongforge"]:
        raise ValueError("Run side identities must be BASE then LoongForge")
    for item in manifests:
        item.setdefault("deterministic", False)
    for key in [
        "steps",
        "schedule_steps",
        "world_size",
        "per_device_batch",
        "ga",
        "global_batch",
        "seed",
        "num_workers",
        "internal_chunk",
        "dtype",
        "autocast",
        "deterministic",
        "engine",
        "dataset_roots",
        "stats_sha256",
    ]:
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"incomparable configuration: {key}")
    for filename in ["asset_sha256.json", "dataset_meta_sha256.json", "initial_trainable.json"]:
        a, b = [json.loads((root / filename).read_text()) for root in [base, trans]]
        if a != b:
            raise ValueError(f"incomparable inputs/weights: {filename}")
    # Pin numerical dependencies, not the two different project distributions.
    dependencies = [
        "torch",
        "numpy",
        "deepspeed",
        "accelerate",
        "diffusers",
        "transformers",
        "torchvision",
        "av",
    ]
    freezes = [
        json.loads((root / "dependency_versions.json").read_text()) for root in [base, trans]
    ]
    for name in dependencies:
        if freezes[0].get(name) != freezes[1].get(name) or name not in freezes[0]:
            raise ValueError(f"dependency mismatch or missing pin: {name}")
    effective = [
        json.loads((root / "effective_deepspeed.json").read_text()) for root in [base, trans]
    ]
    for key, default in [
        ("zero_optimization", None),
        ("bf16", {}),
        ("fp16", {}),
        ("gradient_clipping", 0.0),
        ("train_batch_size", None),
        ("train_micro_batch_size_per_gpu", None),
        ("gradient_accumulation_steps", None),
    ]:
        if effective[0].get(key, default) != effective[1].get(key, default):
            raise ValueError(f"effective DeepSpeed configuration differs: {key}")
    data = [rows(root / "metrics.jsonl") for root in [base, trans]]
    expected = set(range(1, steps + 1))
    for rowset in data:
        if not expected <= rowset.keys():
            raise ValueError(
                "missing optimizer steps; refusing partial/intersection-only comparison"
            )
    deltas = [
        abs(float(data[0][i]["action_loss"]) - float(data[1][i]["action_loss"]))
        for i in sorted(expected)
    ]
    video_deltas = [
        abs(float(data[0][i]["video_loss"]) - float(data[1][i]["video_loss"]))
        for i in sorted(expected)
    ]
    lr_delta = max(
        abs(float(data[0][i]["learning_rate"]) - float(data[1][i]["learning_rate"]))
        for i in expected
    )
    report = {
        "steps": steps,
        "action_loss_mae": sum(deltas) / steps,
        "action_loss_max_abs": max(deltas),
        "video_loss_max_abs": max(video_deltas),
        "lr_max_abs": lr_delta,
        "threshold": threshold,
        "pass_100_step_gate": sum(deltas) / steps <= threshold
        and max(video_deltas) <= threshold
        and lr_delta <= 1e-12,
        "full_plan_acceptance": False,
        "note": "Single-pair short-run gate only; >=3 runs/side, 200-step band, long-run/eval remain required.",
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--transplant", required=True, type=Path)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--mae-threshold", type=float, default=0.001)
    args = p.parse_args()
    if args.steps <= 0 or not math.isfinite(args.mae_threshold) or args.mae_threshold < 0:
        p.error("steps must be positive and threshold finite/nonnegative")
    result = compare(args.base, args.transplant, args.steps, args.mae_threshold)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass_100_step_gate"] else 1)


if __name__ == "__main__":
    main()
