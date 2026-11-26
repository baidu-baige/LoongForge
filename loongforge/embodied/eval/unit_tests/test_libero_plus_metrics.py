# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LIBERO-Plus perturbation-level aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loongforge.embodied.eval.metrics.libero_plus import (
    ALL_SUITES_LABEL,
    TOTAL_LABEL,
    UNMAPPED_CATEGORY,
    load_task_classification,
    summarize_libero_plus,
    write_libero_plus_summary_csv,
)


def _write_classification(path: Path, *, bad_id: bool = False) -> Path:
    data = {
        "libero_spatial": [
            {"id": 1, "name": "task_a", "category": "Camera Viewpoints", "difficulty_level": 1},
            {"id": 2, "name": "task_b", "category": "Camera Viewpoints", "difficulty_level": 2},
            {"id": 3, "name": "task_c", "category": "Light Conditions", "difficulty_level": 5},
        ],
        "libero_goal": [
            {"id": 1, "name": "task_d", "category": "Objects Layout", "difficulty_level": None},
        ],
    }
    if bad_id:
        data["libero_spatial"][1]["id"] = 7
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _record(task_suite: str, task_id: int, success: int, failure_reason=None) -> dict:
    return {
        "benchmark": "libero",
        "task_suite": task_suite,
        "task_id": task_id,
        "episode_idx": 0,
        "success": success,
        "failure_reason": failure_reason,
    }


def test_load_task_classification_maps_ids_to_zero_based(tmp_path: Path) -> None:
    """Ids are 1-based in file order; the loader must expose 0-based task ids."""
    classification = load_task_classification(_write_classification(tmp_path / "cls.json"))

    assert classification["libero_spatial"][0] == {"category": "Camera Viewpoints", "level": 1}
    assert classification["libero_spatial"][2] == {"category": "Light Conditions", "level": 5}
    assert classification["libero_goal"][0]["level"] is None


def test_load_task_classification_rejects_non_identity_order(tmp_path: Path) -> None:
    """A mismatch between id and file position means task_id mapping is unsafe."""
    with pytest.raises(ValueError, match="expected id == index \\+ 1"):
        load_task_classification(_write_classification(tmp_path / "cls.json", bad_id=True))


def test_summarize_libero_plus_groups_by_category_and_total(tmp_path: Path) -> None:
    """Category, suite Total and cross-suite Total rows, infra failures excluded."""
    classification = load_task_classification(_write_classification(tmp_path / "cls.json"))
    records = [
        _record("libero_spatial", 0, 1),
        _record("libero_spatial", 1, 0, "not_successful_within_max_steps"),
        # Infra failure: must stay out of the denominator but remain visible.
        _record("libero_spatial", 2, 0, "episode_timeout"),
        # Unknown task id -> explicit "unmapped" bucket instead of silent loss.
        _record("libero_spatial", 99, 1),
        _record("libero_goal", 0, 1),
    ]

    rows = summarize_libero_plus(records, classification)
    by_key = {(row["task_suite"], row["category"]): row for row in rows}

    camera = by_key[("libero_spatial", "Camera Viewpoints")]
    assert (camera["n_tasks"], camera["n_episodes"], camera["successes"], camera["infra_failed"]) == (2, 2, 1, 0)
    assert camera["success_rate"] == 0.5

    light = by_key[("libero_spatial", "Light Conditions")]
    assert (light["n_tasks"], light["n_episodes"], light["infra_failed"]) == (0, 0, 1)

    assert by_key[("libero_spatial", UNMAPPED_CATEGORY)]["success_rate"] == 1.0

    spatial_total = by_key[("libero_spatial", TOTAL_LABEL)]
    assert (spatial_total["n_tasks"], spatial_total["n_episodes"]) == (3, 3)
    assert (spatial_total["successes"], spatial_total["infra_failed"]) == (2, 1)

    # Same task number in two suites must count as two tasks in the global row.
    overall = by_key[(ALL_SUITES_LABEL, TOTAL_LABEL)]
    assert (overall["n_tasks"], overall["n_episodes"], overall["successes"]) == (4, 4, 3)

    # Rows are ordered per suite, categories before that suite's Total, global Total last.
    assert rows[-1]["task_suite"] == ALL_SUITES_LABEL
    assert rows[-1]["category"] == TOTAL_LABEL


def test_write_libero_plus_summary_csv(tmp_path: Path) -> None:
    """The CSV carries the documented columns."""
    classification = load_task_classification(_write_classification(tmp_path / "cls.json"))
    rows = summarize_libero_plus([_record("libero_goal", 0, 1)], classification)
    summary_path = tmp_path / "libero_plus_summary.csv"

    write_libero_plus_summary_csv(summary_path, rows)

    lines = summary_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == [
        "benchmark",
        "task_suite",
        "category",
        "n_tasks",
        "n_episodes",
        "successes",
        "success_rate",
        "success_stderr",
        "success_ci95_low",
        "success_ci95_high",
        "infra_failed",
    ]
    assert len(lines) == 1 + len(rows)
