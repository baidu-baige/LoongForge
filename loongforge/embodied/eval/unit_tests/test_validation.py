# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for coordinated-run coverage validation."""

from __future__ import annotations

from typing import Any, Dict

from loongforge.embodied.eval.orchestrator.validation import format_report, short_shards, validate


def _rec(episode_id: str, success: bool = True, reason: str | None = None) -> Dict[str, Any]:
    return {"episode_id": episode_id, "success": success, "failure_reason": reason}


def test_a_complete_run_is_ok() -> None:
    """Run test_a_complete_run_is_ok."""
    report = validate({"a": 2}, ["a", "a"], [[_rec("x/1")], [_rec("x/2")]])
    assert report["ok"]
    assert report["actual"] == 2
    assert report["missing"] == 0


def test_missing_episodes_are_counted() -> None:
    """Run test_missing_episodes_are_counted."""
    report = validate({"a": 3}, ["a"], [[_rec("x/1")]])
    assert not report["ok"]
    assert report["missing"] == 2


def test_a_rerun_of_the_same_shard_is_not_a_duplicate() -> None:
    """Re-running a shard overlaps on purpose; dedup by episode_id handles it."""
    records = [_rec("x/1"), _rec("x/2")]
    report = validate({"a": 2}, ["a", "a"], [records, records])
    assert report["ok"]
    assert report["duplicates"] == 0
    assert report["records"] == 4  # the overlap is visible, it just is not a fault


def test_two_shards_claiming_the_same_episode_is_a_duplicate() -> None:
    """Run test_two_shards_claiming_the_same_episode_is_a_duplicate."""
    report = validate({"a": 1, "b": 1}, ["a", "b"], [[_rec("x/1")], [_rec("x/1")]])
    assert report["duplicates"] == 1
    assert not report["ok"]


def test_extra_episodes_are_allowed_replacements() -> None:
    """RoboTwin skips unstable seeds; the replacement seed is extra, not a fault."""
    report = validate({"a": 1}, ["a"], [[_rec("x/1"), _rec("x/2")]])
    assert report["unexpected"] == 1
    assert report["ok"]


def test_short_shards_reports_only_the_short_ones() -> None:
    """Run test_short_shards_reports_only_the_short_ones."""
    assert short_shards({"a": 2, "b": 1}, ["a", "b"], [[_rec("x/1")], [_rec("y/1")]]) == ["a"]


def test_infra_failures_are_separated_from_model_failures() -> None:
    """Run test_infra_failures_are_separated_from_model_failures."""
    report = validate(
        {"a": 3},
        ["a"],
        [
            [
                _rec("x/1", success=False, reason="not_successful_within_max_steps"),
                _rec("x/2", success=False, reason="episode_error"),
                _rec("x/3"),
            ]
        ],
    )
    assert report["ok"]  # every planned episode produced a record
    assert report["successes"] == 1
    assert report["infra_failed"] == 1
    assert "success_rate=0.3333" in format_report(report)
