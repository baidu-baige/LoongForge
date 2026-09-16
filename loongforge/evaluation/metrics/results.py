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
    """Run completed_episode_keys."""
    keys: Set[Tuple[str, str, int, int]] = set()
    for record in records:
        keys.add(
            (
                str(record.get("benchmark")),
                str(record.get("task_suite")),
                int(record.get("task_id")),
                int(record.get("episode_idx")),
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


def write_suite_summary_csv(path: PathLike, records: Iterable[Dict[str, Any]], min_episodes: int = 1) -> None:
    """Run write_suite_summary_csv."""
    rows = summarize_suites(records, min_episodes=min_episodes)
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "task_suite",
        "n_tasks",
        "n_episodes",
        "successes",
        "success_rate",
        "success_stderr",
        "success_ci95_low",
        "success_ci95_high",
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
    "n_episodes",
    "successes",
    "success_rate",
    "success_stderr",
    "success_ci95_low",
    "success_ci95_high",
    "avg_steps",
    "avg_inference_latency_ms",
    "avg_e2e_latency_ms",
    "failure_reasons",
]


def summarize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run summarize_records."""
    return _summarize(records)


def summarize_suites(records: Iterable[Dict[str, Any]], min_episodes: int = 1) -> List[Dict[str, Any]]:
    """Run summarize_suites."""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("benchmark")), str(record.get("task_suite")))
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for (benchmark, task_suite), suite_records in sorted(grouped.items()):
        successes = sum(int(record.get("success", 0)) for record in suite_records)
        n_episodes = len(suite_records)
        success_rate = successes / n_episodes if n_episodes else 0.0
        ci_low, ci_high = _wilson_ci(successes, n_episodes)
        task_episode_counts = _task_episode_counts(suite_records)
        violations = [
            f"task{task_id}:{count}" for task_id, count in sorted(task_episode_counts.items()) if count < min_episodes
        ]
        rows.append(
            {
                "benchmark": benchmark,
                "task_suite": task_suite,
                "n_tasks": len(task_episode_counts),
                "n_episodes": n_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "success_stderr": _binomial_stderr(success_rate, n_episodes),
                "success_ci95_low": ci_low,
                "success_ci95_high": ci_high,
                "min_episodes_per_task": min(task_episode_counts.values()) if task_episode_counts else 0,
                "episode_floor_met": len(violations) == 0,
                "episode_floor_violations": ";".join(violations),
                "failure_reasons": _failure_reasons(suite_records),
            }
        )
    return rows


def validate_episode_floor(records: Iterable[Dict[str, Any]], min_episodes: int) -> List[Dict[str, Any]]:
    """Run validate_episode_floor."""
    violations = []
    grouped: Dict[Tuple[str, str, int], int] = {}
    for record in records:
        key = (str(record.get("benchmark")), str(record.get("task_suite")), int(record.get("task_id")))
        grouped[key] = grouped.get(key, 0) + 1
    for (benchmark, task_suite, task_id), count in sorted(grouped.items()):
        if count < min_episodes:
            violations.append(
                {
                    "benchmark": benchmark,
                    "task_suite": task_suite,
                    "task_id": task_id,
                    "n_episodes": count,
                    "min_required_episodes": min_episodes,
                }
            )
    return violations


def _summarize(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run _summarize."""
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("benchmark")), str(record.get("task_suite")), int(record.get("task_id")))
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, Any]] = []
    for (benchmark, task_suite, task_id), task_records in sorted(grouped.items()):
        successes = sum(int(record.get("success", 0)) for record in task_records)
        n_episodes = len(task_records)
        success_rate = successes / n_episodes if n_episodes else 0.0
        ci_low, ci_high = _wilson_ci(successes, n_episodes)
        rows.append(
            {
                "benchmark": benchmark,
                "task_suite": task_suite,
                "task_id": task_id,
                "n_episodes": n_episodes,
                "successes": successes,
                "success_rate": success_rate,
                "success_stderr": _binomial_stderr(success_rate, n_episodes),
                "success_ci95_low": ci_low,
                "success_ci95_high": ci_high,
                "avg_steps": _average(record.get("steps") for record in task_records),
                "avg_inference_latency_ms": _average(record.get("avg_inference_latency_ms") for record in task_records),
                "avg_e2e_latency_ms": _average(record.get("avg_e2e_latency_ms") for record in task_records),
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


def _task_episode_counts(records: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    """Run _task_episode_counts."""
    counts: Dict[int, int] = {}
    for record in records:
        task_id = int(record.get("task_id"))
        counts[task_id] = counts.get(task_id, 0) + 1
    return counts
