# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .results import PathLike, load_jsonl, summarize_records, summarize_suites, validate_episode_floor


def write_eval_report(
    results_path: PathLike,
    output_dir: PathLike,
    *,
    title: str = "VLA Eval Report",
    artifact_limit: int = 20,
    min_episodes_per_task: int = 1,
) -> Dict[str, str]:
    """Run write_eval_report."""
    records = load_jsonl(results_path)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = summarize_records(records)
    suite_rows = summarize_suites(records, min_episodes=min_episodes_per_task)
    floor_violations = validate_episode_floor(records, min_episodes_per_task)
    failure_rows = _failure_summary(records)
    artifact_rows = _archive_failure_artifacts(records, report_dir / "failure_artifacts", artifact_limit)
    repro_info = _collect_repro_info(results_path)
    resource_snapshot = collect_resource_snapshot()

    report_path = report_dir / "report.md"
    repro_path = report_dir / "repro_info.json"
    failure_manifest_path = report_dir / "failure_artifacts.json"
    resource_snapshot_path = report_dir / "resource_snapshot.json"
    episode_floor_path = report_dir / "episode_floor_violations.json"

    repro_path.write_text(json.dumps(repro_info, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failure_manifest_path.write_text(
        json.dumps(artifact_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resource_snapshot_path.write_text(
        json.dumps(resource_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_floor_path.write_text(
        json.dumps(floor_violations, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        _render_markdown_report(
            title,
            records,
            summary_rows,
            suite_rows,
            floor_violations,
            failure_rows,
            artifact_rows,
            repro_path,
            failure_manifest_path,
            resource_snapshot_path,
            episode_floor_path,
        ),
        encoding="utf-8",
    )
    return {
        "report_path": str(report_path),
        "repro_info_path": str(repro_path),
        "failure_artifacts_manifest_path": str(failure_manifest_path),
        "resource_snapshot_path": str(resource_snapshot_path),
        "episode_floor_violations_path": str(episode_floor_path),
    }


def _render_markdown_report(
    title: str,
    records: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    suite_rows: List[Dict[str, Any]],
    floor_violations: List[Dict[str, Any]],
    failure_rows: List[Dict[str, Any]],
    artifact_rows: List[Dict[str, Any]],
    repro_path: Path,
    failure_manifest_path: Path,
    resource_snapshot_path: Path,
    episode_floor_path: Path,
) -> str:
    """Run _render_markdown_report."""
    total = len(records)
    successes = sum(int(record.get("success", 0)) for record in records)
    success_rate = successes / total if total else 0.0
    ci_low, ci_high = _wilson_ci(successes, total)

    lines = [
        f"# {title}",
        "",
        "## Overall",
        "",
        "| records | successes | success_rate | ci95_low | ci95_high | episode_floor_violations |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {total} | {successes} | {_fmt_float(success_rate)} | {_fmt_float(ci_low)} | "
            f"{_fmt_float(ci_high)} | {len(floor_violations)} |"
        ),
        "",
        "## Per Suite",
        "",
        (
            "| benchmark | task_suite | n_tasks | n_episodes | successes | success_rate | stderr | "
            "ci95_low | ci95_high | min_episodes_per_task | episode_floor_met | failure_reasons |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in suite_rows:
        lines.append(
            (
                "| {benchmark} | {task_suite} | {n_tasks} | {n_episodes} | {successes} | "
                "{success_rate} | {success_stderr} | {success_ci95_low} | {success_ci95_high} | "
                "{min_episodes_per_task} | {episode_floor_met} | {failure_reasons} |"
            ).format(
                benchmark=row["benchmark"],
                task_suite=row["task_suite"],
                n_tasks=row["n_tasks"],
                n_episodes=row["n_episodes"],
                successes=row["successes"],
                success_rate=_fmt_float(row.get("success_rate")),
                success_stderr=_fmt_float(row.get("success_stderr")),
                success_ci95_low=_fmt_float(row.get("success_ci95_low")),
                success_ci95_high=_fmt_float(row.get("success_ci95_high")),
                min_episodes_per_task=row["min_episodes_per_task"],
                episode_floor_met=str(row["episode_floor_met"]).lower(),
                failure_reasons=row.get("failure_reasons") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Per Task",
            "",
            (
                "| benchmark | task_suite | task_id | n_episodes | successes | success_rate | stderr | "
                "ci95_low | ci95_high | avg_steps | avg_inference_latency_ms | avg_e2e_latency_ms |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            (
                "| {benchmark} | {task_suite} | {task_id} | {n_episodes} | {successes} | "
                "{success_rate} | {success_stderr} | {success_ci95_low} | {success_ci95_high} | "
                "{avg_steps} | {avg_inference_latency_ms} | {avg_e2e_latency_ms} |"
            ).format(
                benchmark=row["benchmark"],
                task_suite=row["task_suite"],
                task_id=row["task_id"],
                n_episodes=row["n_episodes"],
                successes=row["successes"],
                success_rate=_fmt_float(row.get("success_rate")),
                success_stderr=_fmt_float(row.get("success_stderr")),
                success_ci95_low=_fmt_float(row.get("success_ci95_low")),
                success_ci95_high=_fmt_float(row.get("success_ci95_high")),
                avg_steps=_fmt_float(row.get("avg_steps")),
                avg_inference_latency_ms=_fmt_float(row.get("avg_inference_latency_ms")),
                avg_e2e_latency_ms=_fmt_float(row.get("avg_e2e_latency_ms")),
            )
        )

    lines.extend(["", "## Failure Reasons", "", "| failure_reason | count |", "|---|---:|"])
    if failure_rows:
        for row in failure_rows:
            lines.append(f"| {row['failure_reason']} | {row['count']} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Episode Floor",
            "",
            f"- Violations manifest: `{episode_floor_path}`",
            f"- Violation count: {len(floor_violations)}",
            "",
            "## Artifacts",
            "",
            f"- Failure artifact manifest: `{failure_manifest_path}`",
            f"- Repro information: `{repro_path}`",
            f"- Resource snapshot: `{resource_snapshot_path}`",
            f"- Archived failure artifact groups: {len(artifact_rows)}",
            "",
        ]
    )
    return "\n".join(lines)


def _failure_summary(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run _failure_summary."""
    counter: Counter[str] = Counter()
    for record in records:
        if int(record.get("success", 0)):
            continue
        reason = record.get("failure_reason") or "not_successful"
        counter[str(reason)] += 1
    return [{"failure_reason": reason, "count": count} for reason, count in sorted(counter.items())]


def _archive_failure_artifacts(
    records: Iterable[Dict[str, Any]], artifact_dir: Path, artifact_limit: int
) -> List[Dict[str, Any]]:
    """Run _archive_failure_artifacts."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    archived: List[Dict[str, Any]] = []
    per_reason_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        if int(record.get("success", 0)):
            continue
        reason = str(record.get("failure_reason") or "not_successful")
        if per_reason_counts[reason] >= artifact_limit:
            continue
        copied_paths = []
        for key in ("replay_path", "trace_path"):
            source_value = record.get(key)
            if not source_value:
                continue
            source = Path(str(source_value))
            if not source.exists():
                continue
            destination_dir = artifact_dir / _safe_name(reason)
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{_episode_name(record)}_{key}{source.suffix}"
            shutil.copy2(source, destination)
            copied_paths.append(str(destination))
        if copied_paths:
            per_reason_counts[reason] += 1
            archived.append(
                {
                    "benchmark": record.get("benchmark"),
                    "task_suite": record.get("task_suite"),
                    "task_id": record.get("task_id"),
                    "episode_idx": record.get("episode_idx"),
                    "failure_reason": reason,
                    "paths": copied_paths,
                }
            )
    return archived


def _collect_repro_info(results_path: PathLike) -> Dict[str, Any]:
    """Run _collect_repro_info."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_path": str(Path(results_path)),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_info(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def collect_resource_snapshot() -> Dict[str, Any]:
    """Run collect_resource_snapshot."""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_count": os.cpu_count(),
        "loadavg": _loadavg(),
        "memory": _memory_info(),
        "disk": _disk_info(),
        "nvidia_smi": _nvidia_smi(),
    }


def _loadavg() -> Optional[List[float]]:
    """Run _loadavg."""
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def _memory_info() -> Dict[str, Optional[int]]:
    """Run _memory_info."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"mem_total_kb": None, "mem_available_kb": None}
    values: Dict[str, Optional[int]] = {"mem_total_kb": None, "mem_available_kb": None}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        if key == "MemTotal":
            values["mem_total_kb"] = int(rest.strip().split()[0])
        elif key == "MemAvailable":
            values["mem_available_kb"] = int(rest.strip().split()[0])
    return values


def _disk_info() -> Dict[str, Dict[str, int]]:
    """Run _disk_info."""
    paths = [Path("/workspace"), Path("/ssd1")]
    info: Dict[str, Dict[str, int]] = {}
    for path in paths:
        if not path.exists():
            continue
        usage = shutil.disk_usage(path)
        info[str(path)] = {"total": usage.total, "used": usage.used, "free": usage.free}
    return info


def _nvidia_smi() -> Optional[str]:
    """Run _nvidia_smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_info() -> Dict[str, Optional[str]]:
    """Run _git_info."""
    repo = Path(__file__).resolve().parents[3]
    return {
        "repo": str(repo),
        "sha": _run_git(repo, ["rev-parse", "HEAD"]),
        "status_short": _run_git(repo, ["status", "--short"]),
    }


def _run_git(repo: Path, args: List[str]) -> Optional[str]:
    """Run _run_git."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(repo), text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Run _wilson_ci."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _fmt_float(value: Any) -> str:
    """Run _fmt_float."""
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _safe_name(value: str) -> str:
    """Run _safe_name."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _episode_name(record: Dict[str, Any]) -> str:
    """Run _episode_name."""
    return _safe_name(
        "{benchmark}_{task_suite}_task{task_id}_ep{episode_idx}".format(
            benchmark=record.get("benchmark"),
            task_suite=record.get("task_suite"),
            task_id=record.get("task_id"),
            episode_idx=record.get("episode_idx"),
        )
    )
