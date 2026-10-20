#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collect numeric release metrics and the baseline from the source checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


METRICS = {
    "iteration", "lm_loss", "action_loss", "loss", "grad_norm",
    "elapsed_time_ms", "throughput", "learning_rate",
    "mem_allocated_avg_MB", "mem_max_allocated_avg_MB",
    "nan_iterations", "skipped_iterations",
}


def numeric_records(records: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in row.items()
         if key in METRICS and isinstance(value, (int, float))
         and not isinstance(value, bool) and math.isfinite(value)}
        for row in records if isinstance(row, dict) and "iteration" in row
    ]


def baseline(source: Path, suite: str, chip: str, model: str, phase: str) -> list[dict]:
    roots = (
        [source / "tests/llm_vlm/baseline" / category / chip
         for category in ("default", "optional")]
        if suite == "llm_vlm"
        else [source / "tests/embodied/baseline" / chip]
    )
    for root in roots:
        path = root / f"{model}.json"
        if path.is_file():
            return numeric_records(json.loads(path.read_text()).get(phase, []))
    return []


def llm_phases(parser, path: Path) -> dict[str, list[dict]]:
    """Keep iterations from different training phases in separate series."""
    phases = {}
    phase, buffer = "unknown", ""

    def flush() -> None:
        record = parser._process_buffer(buffer)
        if record is not None:
            phases.setdefault(phase, []).append(record)

    with path.open(encoding="utf-8", errors="ignore") as log:
        for line in log:
            clean = parser.strip_ansi(line)
            marker = parser.phase_pattern.search(clean)
            if marker:
                flush()
                buffer = ""
                phase = marker.group(1)
            if parser.patterns["iteration"].search(clean):
                flush()
                buffer = line
            else:
                buffer += line
    flush()
    return phases


def collect(
    *, source: Path, llm_log: Path, llm_model: str, llm_chip: str,
    vla_results: Path, vla_chip: str, passed: bool | None,
) -> dict:
    results = []
    if llm_log.is_file():
        parser_path = source / "tests/llm_vlm/tools/log2json.py"
        spec = importlib.util.spec_from_file_location("release_log_parser", parser_path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        for phase, records in llm_phases(parser, llm_log).items():
            actual = numeric_records(records)
            if actual:
                results.append({
                    "suite": "llm_vlm", "model": llm_model, "training_type": phase,
                    "baseline_chip": llm_chip,
                    "passed": passed, "grad_norm_checked": False,
                    "actual": actual,
                    "baseline": baseline(source, "llm_vlm", llm_chip, llm_model, phase),
                })
    if vla_results.is_file():
        payload = json.loads(vla_results.read_text())
        for item in payload.get("results", []):
            model = item.get("model_name", "")
            if not model or Path(model).name != model:
                continue
            results.append({
                "suite": "embodied", "model": model, "training_type": "train",
                "baseline_chip": vla_chip,
                "passed": item.get("passed"), "grad_norm_checked": True,
                "warnings": item.get("warnings", []),
                "failed_metrics": item.get("failed_metrics", []),
                "actual": numeric_records(item.get("metrics", [])),
                "baseline": baseline(source, "embodied", vla_chip, model, "train"),
            })
    return {"suite": "release", "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--llm-log", type=Path, required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--llm-chip", required=True)
    parser.add_argument("--vla-results", type=Path, required=True)
    parser.add_argument("--vla-chip", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(collect(
            source=args.source, llm_log=args.llm_log, llm_model=args.llm_model,
            llm_chip=args.llm_chip, vla_results=args.vla_results,
            vla_chip=args.vla_chip, passed=args.exit_code == 0,
        ), allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
