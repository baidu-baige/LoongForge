# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .results import PathLike, load_jsonl, summarize_records, summarize_suites

_VOLATILE_KEYS = {
    "episode_time_sec",
    "reset_time_sec",
    "avg_inference_latency_ms",
    "avg_e2e_latency_ms",
    "trace_path",
    "replay_path",
    "scheduler_worker_id",
    "server_port",
    "server_host",
    "server_endpoint",
}


def compare_results(left_path: PathLike, right_path: PathLike) -> Dict[str, Any]:
    """Run compare_results."""
    left = _canonical_records(load_jsonl(left_path))
    right = _canonical_records(load_jsonl(right_path))
    left_map = {_record_key(record): record for record in left}
    right_map = {_record_key(record): record for record in right}
    left_keys = set(left_map)
    right_keys = set(right_map)
    changed = []
    for key in sorted(left_keys & right_keys):
        if left_map[key] != right_map[key]:
            changed.append({"key": key, "left": left_map[key], "right": right_map[key]})
    return {
        "match": not (left_keys - right_keys or right_keys - left_keys or changed),
        "left_records": len(left),
        "right_records": len(right),
        "missing_in_right": sorted(left_keys - right_keys),
        "missing_in_left": sorted(right_keys - left_keys),
        "changed_records": changed,
        "left_task_summary": summarize_records(left),
        "right_task_summary": summarize_records(right),
        "left_suite_summary": summarize_suites(left),
        "right_suite_summary": summarize_suites(right),
    }


def write_repro_comparison(left_path: PathLike, right_path: PathLike, output_path: PathLike) -> Dict[str, Any]:
    """Run write_repro_comparison."""
    result = compare_results(left_path, right_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _canonical_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run _canonical_records."""
    canonical = []
    for record in records:
        canonical.append({key: value for key, value in sorted(record.items()) if key not in _VOLATILE_KEYS})
    return sorted(canonical, key=_record_key)


def _record_key(record: Dict[str, Any]) -> Tuple[str, str, int, int]:
    """Run _record_key."""
    return (
        str(record.get("benchmark")),
        str(record.get("task_suite")),
        int(record.get("task_id")),
        int(record.get("episode_idx")),
    )


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    result = write_repro_comparison(args.left, args.right, args.output)
    print(
        json.dumps(
            {"match": result["match"], "changed_records": len(result["changed_records"])}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
