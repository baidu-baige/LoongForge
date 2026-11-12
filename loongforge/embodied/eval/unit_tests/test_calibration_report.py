# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the calibration report builder."""

from __future__ import annotations

from loongforge.embodied.eval.orchestrator.calibration.report import build_report, render_markdown


def test_server_busy_fraction() -> None:
    """Run test_server_busy_fraction."""
    sweep = [{"N": 8, "wall_sec": 100.0, "requests": 1200}]
    report = build_report(sweep, {"mean": 50.0})
    assert report["sweep"][0]["server_busy_frac"] == round(1200 * 50.0 / 1000.0 / 100.0, 3)


def test_missing_requests_is_none() -> None:
    """Run test_missing_requests_is_none."""
    report = build_report([{"N": 1, "wall_sec": 40.0}], {"mean": 50.0})
    assert report["sweep"][0]["server_busy_frac"] is None


def test_render_markdown_contains_rows_and_headers() -> None:
    """Run test_render_markdown_contains_rows_and_headers."""
    sweep = [
        {"N": 1, "wall_sec": 40.0, "requests": 150, "throughput_ep_per_min": 1.5},
        {"N": 8, "wall_sec": 94.0, "requests": 1200, "throughput_ep_per_min": 5.1},
    ]
    markdown = render_markdown(build_report(sweep, {"mean": 49.2}))
    assert "| 1 |" in markdown and "| 8 |" in markdown
    assert "server busy frac" in markdown
    assert "utilization.gpu" in markdown  # the "do not use" note
