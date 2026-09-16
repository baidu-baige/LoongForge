#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Render a regression suite results.json as a compact markdown table.

The Native regression framework writes per-model pass/fail state, failed
baseline comparisons, warnings, and per-iteration metrics to results.json.
This script turns that file into a short markdown summary suitable for a
pull request check run output. It never fails the job: any input problem
produces a fallback line instead of an error exit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

# Keep the rendered table compact: the Notes column condenses baseline
# findings to "metric degraded N%" form and clips anything else to a short
# excerpt. Full detail stays in the suite log artifact.
NOTE_EXCERPT_CHARS = 120

# Matches the framework's soft-check warning text, e.g.
# "throughput: actual_mean=4.40 baseline_mean=14.47 degraded 69.6% > 5%
#  (soft check, warning only)".
_METRIC_WARNING_RE = re.compile(
    r"^(?P<metric>\S+?): actual_mean=\S+ baseline_mean=\S+ "
    r"degraded (?P<pct>\d+(?:\.\d+)?)%"
)


def _clip(text: str, limit: int = NOTE_EXCERPT_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _condense_warning(note: str) -> str:
    match = _METRIC_WARNING_RE.match(note.strip())
    if match:
        # The percentage is a direction-normalised "how much worse" magnitude,
        # not a signed delta: baseline.py flips the sign for metrics where
        # smaller is worse. Rendering it as "-X%" therefore reads as an
        # improvement for elapsed_time_ms, so keep the framework's wording.
        return f"{match.group('metric')} degraded {match.group('pct')}%"
    return _clip(note)


def _condense(notes: list[str]) -> str:
    return "; ".join(_condense_warning(note) for note in notes if note.strip())


def _loss_series(metrics: list[dict[str, Any]]) -> list[tuple[int, str, float]]:
    losses: list[tuple[int, str, float]] = []
    for record in metrics or []:
        iteration = record.get("iteration", 0)
        for key, value in record.items():
            if "loss" not in key or key == "loss_scale":
                continue
            try:
                losses.append((int(iteration), key, float(value)))
            except (TypeError, ValueError):
                continue
    return losses


def _render_model_row(result: dict[str, Any]) -> str:
    name = str(result.get("model_name", "?"))
    passed = result.get("passed", False)
    verdict = "PASS" if passed else "FAIL"

    losses = _loss_series(result.get("metrics", []))
    if losses:
        first_key, first_value = losses[0][1], losses[0][2]
        last_key, last_value = losses[-1][1], losses[-1][2]
        if first_key == last_key:
            loss_text = f"{first_key}: {first_value:.6g} → {last_value:.6g}"
        else:
            loss_text = (
                f"{first_key}: {first_value:.6g}; {last_key}: {last_value:.6g}"
            )
    else:
        loss_text = "-"

    notes: list[str] = []
    failed = list(result.get("failed_metrics") or [])
    warnings = list(result.get("warnings") or [])
    error = str(result.get("error") or "").strip()
    if failed:
        notes.append(f"failed: {_clip('; '.join(failed))}")
    if warnings:
        notes.append(f"warn: {_condense(warnings)}")
    if error:
        notes.append(f"error: {_clip(error)}")
    notes_text = "<br>".join(notes) if notes else "-"

    return f"| {name} | {verdict} | {loss_text} | {notes_text} |"


def render(summary: dict[str, Any]) -> str:
    results = summary.get("results") or []
    lines = [
        "Regression results:",
        "",
        "| Model | Result | Loss (first → last) | Notes |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(_render_model_row(result) for result in results)
    passed = sum(1 for result in results if result.get("passed"))
    lines.append("")
    lines.append(f"{passed}/{len(results)} models passed baseline comparison.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_file", help="path to the suite results.json")
    args = parser.parse_args()

    try:
        with open(args.results_file, encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict) or not isinstance(
            summary.get("results"), list
        ):
            raise ValueError("unexpected results.json structure")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # Report the parsing problem without failing the job; the check run
        # still carries the suite-level pass/fail state.
        print(f"Suite detail unavailable ({type(exc).__name__}).")
        return 0

    print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
