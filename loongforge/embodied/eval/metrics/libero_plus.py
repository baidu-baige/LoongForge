# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-Plus perturbation-level aggregation.

LIBERO-Plus reuses the standard LIBERO task interface (same ``libero`` package
name, same ``BENCHMARK_MAPPING``), but ships 10,030 perturbed tasks whose
``benchmark/task_classification.json`` labels each task with a perturbation
category and a difficulty level (both 1-based ids).  Standard ``summary.csv`` /
``suite_summary.csv`` aggregate per task and per suite only, which hides the
official leaderboard breakdown (7 perturbation categories + Total).  The helpers
here add that view without touching the runner, adapter or server.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .results import _binomial_stderr, _wilson_ci, is_infra_failure, record_task_id

PathLike = Union[str, Path]

LIBERO_PLUS_BENCHMARK = "libero_plus"
TOTAL_LABEL = "Total"
ALL_SUITES_LABEL = "all"
UNMAPPED_CATEGORY = "unmapped"

# Canonical ordering so merged reports read like the official leaderboard.
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORY_ORDER = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)

_FIELDNAMES = [
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


def default_task_classification_path() -> Path:
    """Return ``<libero benchmark_root>/benchmark/task_classification.json``."""
    from libero.libero import get_libero_path

    return Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"


def load_task_classification(path: Optional[PathLike] = None) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load LIBERO-Plus ``task_classification.json``.

    Returns ``{suite: {task_id(0-based): {"category": str, "level": int|None}}}``.

    ``task_classification.json`` stores 1-based ids in file order and LIBERO's
    ``task_order_index=0`` keeps the task enumeration in that same order, so
    ``task_id == id - 1``.  Any other layout would silently mislabel categories,
    hence the strict check.
    """
    classification_path = Path(path) if path else default_task_classification_path()
    data = json.loads(classification_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{classification_path}: expected a suite -> task list mapping")

    classification: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for suite, tasks in data.items():
        if not isinstance(tasks, list):
            raise ValueError(f"{classification_path}: suite {suite!r} is not a list")
        suite_tasks: Dict[int, Dict[str, Any]] = {}
        for index, task in enumerate(tasks):
            task_id = int(task["id"]) - 1
            if task_id != index:
                raise ValueError(
                    f"{classification_path}: suite {suite!r} task at index {index} has id "
                    f"{task['id']!r}; expected id == index + 1 (1-based, file order)"
                )
            suite_tasks[index] = {
                "category": str(task["category"]),
                "level": task.get("difficulty_level"),
            }
        classification[str(suite)] = suite_tasks
    return classification


def summarize_libero_plus(
    records: Iterable[Dict[str, Any]],
    classification: Dict[str, Dict[int, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Aggregate episode records per (suite, category), plus Total rows.

    Infrastructure failures are excluded from the success-rate denominator (they
    remain visible through ``infra_failed``), mirroring ``summarize_suites``.
    Rows are emitted for every category that has at least one record, then one
    ``Total`` row per suite, then one ``all``/``Total`` row.
    """
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    all_records: List[Dict[str, Any]] = []
    for record in records:
        suite = str(record.get("task_suite") or "")
        task_id = record_task_id(record)
        category = _category_for(classification, suite, task_id)
        grouped.setdefault(suite, {}).setdefault(category, []).append(record)
        all_records.append(record)

    rows: List[Dict[str, Any]] = []
    for suite in _sorted_keys(grouped, SUITE_ORDER):
        categories = grouped[suite]
        for category in _sorted_keys(categories, CATEGORY_ORDER):
            rows.append(_build_row(suite, category, categories[category]))
        rows.append(_build_row(suite, TOTAL_LABEL, [r for items in categories.values() for r in items]))
    if all_records:
        rows.append(_build_row(ALL_SUITES_LABEL, TOTAL_LABEL, all_records))
    return rows


def write_libero_plus_summary_csv(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    """Write the perturbation-level rows to ``libero_plus_summary.csv``."""
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _category_for(
    classification: Dict[str, Dict[int, Dict[str, Any]]], suite: str, task_id: int
) -> str:
    """Map a record's (suite, task_id) to its perturbation category."""
    task = classification.get(suite, {}).get(task_id)
    if task is not None:
        return str(task["category"])
    logging.warning(
        "LIBERO-Plus: no classification entry for suite=%s task_id=%s; using %r",
        suite,
        task_id,
        UNMAPPED_CATEGORY,
    )
    return UNMAPPED_CATEGORY


def _sorted_keys(mapping: Dict[str, Any], order: Iterable[str]) -> List[str]:
    """Sort known keys by the canonical order, unknown keys alphabetically last."""
    canonical = list(order)
    rank = {key: index for index, key in enumerate(canonical)}
    return sorted(mapping, key=lambda key: (rank.get(key, len(canonical)), key))


def _build_row(task_suite: str, category: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one (suite, category) bucket."""
    evaluated = [record for record in records if not is_infra_failure(record)]
    infra_failed = len(records) - len(evaluated)
    successes = sum(int(record.get("success", 0)) for record in evaluated)
    n_episodes = len(evaluated)
    # Task ids are only unique within a suite; key on (suite, task_id) so the
    # cross-suite Total row does not merge same-numbered tasks.
    n_tasks = len({(str(record.get("task_suite") or ""), record_task_id(record)) for record in evaluated})
    success_rate = successes / n_episodes if n_episodes else 0.0
    ci_low, ci_high = _wilson_ci(successes, n_episodes)
    return {
        "benchmark": LIBERO_PLUS_BENCHMARK,
        "task_suite": task_suite,
        "category": category,
        "n_tasks": n_tasks,
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "success_stderr": _binomial_stderr(success_rate, n_episodes),
        "success_ci95_low": ci_low,
        "success_ci95_high": ci_high,
        "infra_failed": infra_failed,
    }
