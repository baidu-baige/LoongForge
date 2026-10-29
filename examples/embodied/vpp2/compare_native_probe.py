#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Strict comparison for TINY native-trainer regression, not full14B acceptance."""
import argparse
import json
import math
from pathlib import Path
import torch


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def compare(base, trans, steps, start=1):
    rows = [[r.get("train", r) for r in lines(root / "metrics.jsonl")] for root in (base, trans)]
    # BASE is full reference; a resumed run may start later.
    mappings = []
    for side in rows:
        normalized = {}
        for row in side:
            step = int(row.get("iteration", row.get("step", -1)))
            if step < 1 or step in normalized:
                raise ValueError("Invalid or duplicate optimizer step")
            if int(row.get("nan_iterations", 0)) or int(row.get("skipped_iterations", 0)):
                raise ValueError("NaN or skipped optimizer update")
            normalized[step] = dict(row, learning_rate=row.get("learning_rate", row.get("lr")))
        mappings.append(normalized)
    expected = set(range(start, steps + 1))
    if not expected <= mappings[0].keys() or mappings[1].keys() != expected:
        raise ValueError("Missing required optimizer steps")
    report = dict(steps=steps, compared_start=start, full14B=False, canonical_zero2_pass=False)
    for key in ("action_loss", "video_loss", "learning_rate", "grad_norm"):
        deltas = [abs(float(mappings[0][i][key]) - float(mappings[1][i][key])) for i in sorted(expected)]
        if not all(math.isfinite(v) for v in deltas):
            raise ValueError("Nonfinite metric")
        report[key + "_mae"] = sum(deltas) / len(deltas)
        report[key + "_max_abs"] = max(deltas)
    weights = [torch.load(root / "final.pt", map_location="cpu", weights_only=True) for root in (base, trans)]
    if weights[0].keys() != weights[1].keys():
        raise ValueError("Parameter coverage differs")
    report["final_parameters_exact"] = all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    report["final_parameters_max_abs"] = max(float((weights[0][k].float() - weights[1][k].float()).abs().max()) for k in weights[0])
    report["trainable_parameters"] = sum(v.numel() for v in weights[0].values())
    if start == 1:
        report["initial_exact"] = read(base / "initial_trainable.json") == read(trans / "initial_trainable.json")
    world = read(base / "scope.json")["world"]
    if read(trans / "scope.json")["world"] != world:
        raise ValueError("World sizes differ")
    report["input_exact_by_rank"] = {}
    for rank in range(world):
        x, y = [lines(root / f"inputs_rank{rank}.jsonl") for root in (base, trans)]
        # Reference may cover a full run when TRANS is resumed. Compare every
        # original tensor key; only TRANS-only logging metadata is permitted.
        x = x[start - 1:steps]
        report["input_exact_by_rank"][str(rank)] = len(x) == len(y) == steps - start + 1 and all(
            all(k in b and v == b[k] for k, v in a.items()) for a, b in zip(x, y)
        )
    report["exact_native_tiny_pass"] = all(report[k + "_max_abs"] == 0 for k in ("action_loss", "video_loss", "learning_rate", "grad_norm")) and report["final_parameters_exact"] and all(report["input_exact_by_rank"].values())
    if start == 1:
        report["exact_native_tiny_pass"] &= report["initial_exact"]
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--transplant", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--start", type=int, default=1)
    a = p.parse_args()
    result = compare(a.base, a.transplant, a.steps, a.start)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["exact_native_tiny_pass"] else 1)


if __name__ == "__main__":
    main()
