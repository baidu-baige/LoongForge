# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .results import PathLike, load_jsonl


def archive_traces(results_path: PathLike, output_dir: PathLike, run_id: str = "default") -> List[Dict[str, Any]]:
    """Run archive_traces."""
    records = load_jsonl(results_path)
    trace_root = Path(output_dir) / "traces" / _safe_name(run_id)
    manifest: List[Dict[str, Any]] = []
    for record in records:
        trace_path = record.get("trace_path")
        if not trace_path:
            continue
        source = Path(str(trace_path))
        if not source.exists():
            continue
        destination_dir = (
            trace_root / _safe_name(str(record.get("benchmark"))) / _safe_name(str(record.get("task_suite")))
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"task{record.get('task_id')}_ep{record.get('episode_idx')}{source.suffix}"
        shutil.copy2(source, destination)
        manifest.append(
            {
                "benchmark": record.get("benchmark"),
                "task_suite": record.get("task_suite"),
                "task_id": record.get("task_id"),
                "episode_idx": record.get("episode_idx"),
                "source": str(source),
                "archive_path": str(destination),
            }
        )
    manifest_path = trace_root / "manifest.json"
    trace_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _safe_name(value: str) -> str:
    """Run _safe_name."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
