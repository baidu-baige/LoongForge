# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

from loongforge.embodied.eval.metrics.repro_compare import compare_results
from loongforge.embodied.eval.metrics.report import write_eval_report
from loongforge.embodied.eval.metrics.resource_monitor import sample_resources
from loongforge.embodied.eval.metrics.results import (
    load_jsonl,
    summarize_records,
    summarize_suites,
    validate_episode_floor,
    write_suite_summary_csv,
    write_summary_csv,
)
from loongforge.embodied.eval.metrics.trace_archive import archive_traces


def test_summary_includes_confidence_and_failures(tmp_path: Path) -> None:
    """Run test_summary_includes_confidence_and_failures."""
    records = [
        {"benchmark": "libero", "task_suite": "libero_goal", "task_id": 0, "episode_idx": 0, "success": 1, "steps": 10},
        {
            "benchmark": "libero",
            "task_suite": "libero_goal",
            "task_id": 0,
            "episode_idx": 1,
            "success": 0,
            "steps": 20,
            "failure_reason": "not_successful_within_max_steps",
        },
    ]

    rows = summarize_records(records)

    assert len(rows) == 1
    assert rows[0]["success_rate"] == 0.5
    assert rows[0]["successes"] == 1
    assert rows[0]["success_stderr"] == 0.3535533905932738
    assert 0.0 <= rows[0]["success_ci95_low"] <= rows[0]["success_ci95_high"] <= 1.0
    assert rows[0]["failure_reasons"] == "not_successful_within_max_steps:1"

    summary_path = tmp_path / "summary.csv"
    write_summary_csv(summary_path, records)
    content = summary_path.read_text(encoding="utf-8")
    assert "success_ci95_low" in content
    assert "failure_reasons" in content

    suite_rows = summarize_suites(records, min_episodes=3)
    assert suite_rows[0]["n_tasks"] == 1
    assert suite_rows[0]["min_episodes_per_task"] == 2
    assert suite_rows[0]["episode_floor_met"] is False
    assert validate_episode_floor(records, min_episodes=3)[0]["task_id"] == 0

    suite_summary_path = tmp_path / "suite_summary.csv"
    write_suite_summary_csv(suite_summary_path, records, min_episodes=3)
    assert "episode_floor_violations" in suite_summary_path.read_text(encoding="utf-8")


def test_write_eval_report_archives_failure_artifacts(tmp_path: Path) -> None:
    """Run test_write_eval_report_archives_failure_artifacts."""
    trace_path = tmp_path / "trace.json"
    trace_path.write_text("[]\n", encoding="utf-8")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "benchmark": "simplerenv",
                "task_suite": "widowx_put_eggplant_in_basket",
                "task_id": 0,
                "episode_idx": 0,
                "success": 0,
                "steps": 3,
                "failure_reason": "not_successful_within_max_steps",
                "trace_path": str(trace_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = write_eval_report(results_path, tmp_path / "report", title="Smoke Report")

    report_path = Path(outputs["report_path"])
    manifest_path = Path(outputs["failure_artifacts_manifest_path"])
    assert report_path.exists()
    assert manifest_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Smoke Report" in report_text
    assert "Per Suite" in report_text
    assert Path(outputs["resource_snapshot_path"]).exists()
    assert Path(outputs["episode_floor_violations_path"]).exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest[0]["failure_reason"] == "not_successful_within_max_steps"
    assert Path(manifest[0]["paths"][0]).exists()
    assert load_jsonl(results_path)[0]["success"] == 0

    trace_manifest = archive_traces(results_path, tmp_path / "trace_archive", run_id="run-a")
    assert len(trace_manifest) == 1
    assert Path(trace_manifest[0]["archive_path"]).exists()

    resource_result = sample_resources(tmp_path / "resources.jsonl", interval_sec=0.1, duration_sec=0.0)
    assert resource_result["samples"] == 1
    assert (tmp_path / "resources.jsonl").exists()


def test_compare_results_ignores_runtime_fields(tmp_path: Path) -> None:
    """Run test_compare_results_ignores_runtime_fields."""
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    base = {
        "benchmark": "libero",
        "task_suite": "libero_goal",
        "task_id": 0,
        "episode_idx": 0,
        "success": 1,
        "steps": 12,
    }
    left = dict(base, episode_time_sec=1.0, trace_path="/tmp/a.json")
    right = dict(base, episode_time_sec=2.0, trace_path="/tmp/b.json")
    left_path.write_text(json.dumps(left) + "\n", encoding="utf-8")
    right_path.write_text(json.dumps(right) + "\n", encoding="utf-8")

    result = compare_results(left_path, right_path)

    assert result["match"] is True
    assert result["changed_records"] == []
