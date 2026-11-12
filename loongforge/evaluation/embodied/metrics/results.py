# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


PathLike = Union[str, Path]

# Failures caused by the evaluation infrastructure rather than by the policy.
# Such episodes must not count as "completed" for resume, must not enter the
# success-rate denominator, and must remain visible as missing coverage.
INFRA_FAILURE_REASONS = frozenset(
    {
        "policy_timeout",
        "env_timeout",
        "episode_timeout",
        "sequence_timeout",
        "server_unreachable",
        "episode_error",
        "sequence_error",
    }
)


def is_infra_failure(record: Dict[str, Any]) -> bool:
    """Return True when a failed episode carries an infrastructure failure reason."""
    if int(record.get("success", 0)):
        return False
    return str(record.get("failure_reason") or "") in INFRA_FAILURE_REASONS


def record_task_id(record: Dict[str, Any]) -> int:
    """Return the record's task id.

    CALVIN records have no ``task_id``; all sequences belong to one suite, so
    they are grouped under task 0 and distinguished by ``sequence_idx``.
    """
    if record.get("task_id") is not None:
        return int(record["task_id"])
    return 0


def record_episode_idx(record: Dict[str, Any]) -> int:
    """Return the record's episode index (CALVIN falls back to ``sequence_idx``)."""
    if record.get("episode_idx") is not None:
        return int(record["episode_idx"])
    if record.get("sequence_idx") is not None:
        return int(record["sequence_idx"])
    raise ValueError(f"record has neither episode_idx nor sequence_idx: {record!r}")


def append_jsonl(path: PathLike, record: Dict[str, Any]) -> None:
    """Run append_jsonl."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()


def load_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Run load_jsonl."""
    result_path = Path(path)
    if not result_path.exists():
        return []
    with result_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def completed_episode_keys(records: Iterable[Dict[str, Any]]) -> Set[Tuple[str, str, int, int]]:
    """Return episode keys that are genuinely complete.

    Infrastructure failures are retried by the runner; if the final record still
    carries an infra reason, the episode did not complete and must be retried on
    the next resume instead of being skipped forever.
    """
    keys: Set[Tuple[str, str, int, int]] = set()
    for record in records:
        if is_infra_failure(record):
            continue
        keys.add(
            (
                str(record.get("benchmark")),
                str(record.get("task_suite")),
                record_task_id(record),
                record_episode_idx(record),
            )
        )
    return keys


def write_summary_csv(path: PathLike, records: Iterable[Dict[str, Any]]) -> None:
    """Run write_summary_csv."""
    rows = summarize_records(records)
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _SUMMARY_FIELDNAMES
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_suite_summary_csv(
    path: PathLike,
    records: Iterable[Dict[str, Any]],
    min_episodes: int = 1,
    expected_counts: Optional[Dict[Any, int]] = None,
) -> None:
    """Run write_suite_summary_csv."""
    rows = summarize_suites(records, min_episodes=min_episodes, expected_counts=expected_counts)
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "task_suite",
        "n_tasks",
        "n_records",
        "n_episodes",
        "successes",
        "success_rate",
        "success_stderr",
        "success_ci95_low",
        "success_ci95_high",
        "infra_failed",
        "min_episodes_per_task",
        "episode_floor_met",
        "episode_floor_violations",
        "failure_reasons",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_SUMMARY_FIELDNAMES = [
    "benchmark",
    "task_suite",
    "task_id",
    "n_records",
    "n_episodes",
    "successes",
    "success_rate",
    "success_stderr",
    "success_ci95_low",
    "success_ci95_high",
    "infra_failed",
    "avg_steps",
    "avg_inference_latency_ms",
    "avg_e2e_latency_ms",
    "failure_reasons",
]


def summarize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run summarize_records."""
    return _summarize(records)


def summarize_suites(
    records: Iterable[Dict[str, Any]],
    min_episodes: int = 1,
    expected_counts: Optional[Dict[Any, int]] = None,
) -> List[Dict[str, Any]]:
    """Summarize per suite.

    ``expected_counts`` makes completeness independent of the recorded data:
    either a flat ``{task_id: episodes}`` mapping (single-suite runs) or a
    ``{(benchmark, task_suite): {task_id: episodes}}`` mapping. Missing tasks
    are reported as violations instead of silently disappearing.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("benchmark")), str(record.get("task_suite")))
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for (benchmark, task_suite), suite_records in sorted(grouped.items()):
        evaluated = [record for record in suite_records if not is_infra_failure(record)]
        infra_failed = len(suite_records) - len(evaluated)
        successes = sum(int(record.get("success", 0)) for record in evaluated)
        n_episodes = len(evaluated)
        success_rate = successes / n_episodes if n_episodes else 0.0
        ci_low, ci_high = _wilson_ci(successes, n_episodes)
        expected = _expected_for_suite(expected_counts, benchmark, task_suite)
        task_episode_counts = _task_episode_counts(evaluated, expected)
        violations = _floor_violations(task_episode_counts, min_episodes, expected)
        rows.append(
            {
                "benchmark": benchmark,
                "task_suite": task_suite,
                "n_tasks": len(task_episode_counts),
                "n_records": len(suite_records),
                "n_episodes": n_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "success_stderr": _binomial_stderr(success_rate, n_episodes),
                "success_ci95_low": ci_low,
                "success_ci95_high": ci_high,
                "infra_failed": infra_failed,
                "min_episodes_per_task": min(task_episode_counts.values()) if task_episode_counts else 0,
                "episode_floor_met": len(violations) == 0,
                "episode_floor_violations": ";".join(violations),
                "failure_reasons": _failure_reasons(suite_records),
            }
        )
    return rows


def validate_episode_floor(
    records: Iterable[Dict[str, Any]],
    min_episodes: int,
    expected_counts: Optional[Dict[Any, int]] = None,
) -> List[Dict[str, Any]]:
    """Return one violation per task whose valid episode count is too low.

    Infrastructure failures are not counted as episodes, so they surface as
    missing coverage. When ``expected_counts`` is provided it is authoritative:
    tasks absent from the records are reported with ``n_episodes=0``.
    """
    violations = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("benchmark")), str(record.get("task_suite")))
        grouped.setdefault(key, []).append(record)
    for (benchmark, task_suite), suite_records in sorted(grouped.items()):
        evaluated = [record for record in suite_records if not is_infra_failure(record)]
        expected = _expected_for_suite(expected_counts, benchmark, task_suite)
        counts = _task_episode_counts(evaluated, expected)
        for task_id, count in sorted(counts.items()):
            required = int(expected[task_id]) if expected and task_id in expected else int(min_episodes)
            if count < required:
                violations.append(
                    {
                        "benchmark": benchmark,
                        "task_suite": task_suite,
                        "task_id": task_id,
                        "n_episodes": count,
                        "min_required_episodes": required,
                    }
                )
    return violations


def _summarize(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run _summarize."""
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("benchmark")), str(record.get("task_suite")), record_task_id(record))
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for (benchmark, task_suite, task_id), task_records in sorted(grouped.items()):
        evaluated = [record for record in task_records if not is_infra_failure(record)]
        infra_failed = len(task_records) - len(evaluated)
        successes = sum(int(record.get("success", 0)) for record in evaluated)
        n_episodes = len(evaluated)
        success_rate = successes / n_episodes if n_episodes else 0.0
        ci_low, ci_high = _wilson_ci(successes, n_episodes)
        rows.append(
            {
                "benchmark": benchmark,
                "task_suite": task_suite,
                "task_id": task_id,
                "n_records": len(task_records),
                "n_episodes": n_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "success_stderr": _binomial_stderr(success_rate, n_episodes),
                "success_ci95_low": ci_low,
                "success_ci95_high": ci_high,
                "infra_failed": infra_failed,
                "avg_steps": _average(record.get("steps") for record in evaluated),
                "avg_inference_latency_ms": _average(
                    record.get("avg_inference_latency_ms") for record in evaluated
                ),
                "avg_e2e_latency_ms": _average(record.get("avg_e2e_latency_ms") for record in evaluated),
                "failure_reasons": _failure_reasons(task_records),
            }
        )
    return rows


def _average(values: Iterable[Any]) -> Optional[float]:
    """Run _average."""
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _binomial_stderr(success_rate: float, n_episodes: int) -> float:
    """Run _binomial_stderr."""
    if n_episodes <= 0:
        return 0.0
    return math.sqrt(success_rate * (1.0 - success_rate) / n_episodes)


def _wilson_ci(successes: int, n_episodes: int, z: float = 1.96) -> Tuple[float, float]:
    """Run _wilson_ci."""
    if n_episodes <= 0:
        return (0.0, 0.0)
    p = successes / n_episodes
    denom = 1.0 + z * z / n_episodes
    center = (p + z * z / (2.0 * n_episodes)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n_episodes)) / n_episodes) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _failure_reasons(records: Iterable[Dict[str, Any]]) -> str:
    """Run _failure_reasons."""
    counts: Dict[str, int] = {}
    for record in records:
        if int(record.get("success", 0)):
            continue
        reason = str(record.get("failure_reason") or "not_successful")
        counts[reason] = counts.get(reason, 0) + 1
    return ";".join(f"{reason}:{count}" for reason, count in sorted(counts.items()))


def _expected_for_suite(
    expected_counts: Optional[Dict[Any, int]], benchmark: str, task_suite: str
) -> Optional[Dict[int, int]]:
    """Normalize flat or per-suite expected counts to ``{task_id: episodes}``."""
    if not expected_counts:
        return None
    if all(isinstance(key, tuple) for key in expected_counts):
        suite_counts = expected_counts.get((benchmark, task_suite))
        return dict(suite_counts) if suite_counts else None
    return dict(expected_counts)


def _floor_violations(
    counts: Dict[int, int], min_episodes: int, expected: Optional[Dict[int, int]]
) -> List[str]:
    """Return ``task<id>:<count>`` for every task below its required count."""
    violations = []
    for task_id, count in sorted(counts.items()):
        required = int(expected[task_id]) if expected and task_id in expected else int(min_episodes)
        if count < required:
            violations.append(f"task{task_id}:{count}")
    return violations


def _task_episode_counts(
    records: Iterable[Dict[str, Any]], expected: Optional[Dict[int, int]] = None
) -> Dict[int, int]:
    """Count valid episodes per task, keeping expected-but-absent tasks at zero."""
    counts: Dict[int, int] = {}
    for record in records:
        task_id = record_task_id(record)
        counts[task_id] = counts.get(task_id, 0) + 1
    if expected:
        for task_id in expected:
            counts.setdefault(int(task_id), 0)
    return counts
