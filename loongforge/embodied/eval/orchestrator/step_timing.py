# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Split an eval run's per-episode wall time into setup vs steady-state compute.

``episode_time_sec`` in ``results.jsonl`` covers the whole unit: scene build,
asset I/O, rollout and teardown. For short episodes that is dominated by setup,
so comparing configurations with it measures I/O rather than render/policy cost.

The per-step ``timestamp_sec`` in ``trace.json`` gives the steady-state step
interval, so this module reports both:

* ``compute_sec``  — time inside the rollout (first to last step timestamp)
* ``setup_sec``    — episode wall minus compute (scene build, I/O, teardown)
* ``inference_sec``— summed policy latency inside the rollout
* ``render_sec``   — compute minus inference (render + physics + client glue)

Usage:
    python -m loongforge.embodied.eval.orchestrator.step_timing <output_dir> [...]
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_trace(worker_dir: Path) -> Optional[List[Dict[str, Any]]]:
    matches = glob.glob(str(worker_dir / "artifacts/robotwin/*/*/trace.json"))
    if not matches:
        return None
    try:
        data = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    steps = data.get("steps")
    return steps if isinstance(steps, list) and len(steps) >= 2 else None


def _first_record(worker_dir: Path) -> Optional[Dict[str, Any]]:
    path = worker_dir / "results.jsonl"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def summarize_run(output_dir: str) -> Dict[str, Any]:
    """Aggregate setup / compute / inference split over a run's workers."""
    root = Path(output_dir)
    result = json.loads((root / "coordinator_result.json").read_text(encoding="utf-8"))
    sims = int(result.get("concurrency") or 0)

    episodes: List[Dict[str, Any]] = []
    for worker_dir in sorted(root.glob("worker_*")):
        record = _first_record(worker_dir)
        trace = _load_trace(worker_dir)
        if record is None or trace is None:
            continue
        stamps = [step["timestamp_sec"] for step in trace if step.get("timestamp_sec")]
        if len(stamps) < 2:
            continue
        compute = stamps[-1] - stamps[0]
        intervals = [b - a for a, b in zip(stamps, stamps[1:])]
        inference = sum(
            float(step.get("inference_latency_ms") or 0.0) for step in trace
        ) / 1000.0
        episode_wall = float(record.get("episode_time_sec") or 0.0)
        episodes.append(
            {
                "task": record.get("task_suite"),
                "success": int(record.get("success") or 0),
                "steps": len(trace),
                "episode_wall": episode_wall,
                "compute": compute,
                "setup": max(0.0, episode_wall - compute),
                "inference": inference,
                "render": max(0.0, compute - inference),
                "median_step": statistics.median(intervals) if intervals else 0.0,
            }
        )
    if not episodes:
        raise ValueError(f"no trace/results pairs under {output_dir}")

    def mean(key: str) -> float:
        return statistics.mean(ep[key] for ep in episodes)

    mean_wall, mean_compute = mean("episode_wall"), mean("compute")
    return {
        "output_dir": str(root),
        "episodes": len(episodes),
        "sims": sims,
        "success_rate": statistics.mean(ep["success"] for ep in episodes),
        "mean_steps": mean("steps"),
        "median_step_sec": mean("median_step"),
        "mean_episode_wall_sec": mean_wall,
        "mean_compute_sec": mean_compute,
        "mean_setup_sec": mean("setup"),
        "mean_inference_sec": mean("inference"),
        "mean_render_sec": mean("render"),
        "setup_share": mean("setup") / mean_wall if mean_wall else 0.0,
        # End-to-end (includes scene build) vs steady-state compute only.
        "throughput_wall_ep_per_min": sims * 60.0 / mean_wall if mean_wall else 0.0,
        "throughput_compute_ep_per_min": sims * 60.0 / mean_compute if mean_compute else 0.0,
        "throughput_render_ep_per_min": (
            sims * 60.0 / mean("render") if mean("render") else 0.0
        ),
    }


def format_row(row: Dict[str, Any]) -> str:
    """One aligned line per run."""
    return (
        f"{Path(row['output_dir']).parent.name:18} "
        f"n={row['episodes']:3d} sims={row['sims']:2d} "
        f"succ={row['success_rate']*100:5.1f}% "
        f"wall={row['mean_episode_wall_sec']:6.0f}s "
        f"setup={row['mean_setup_sec']:6.0f}s({row['setup_share']*100:4.1f}%) "
        f"compute={row['mean_compute_sec']:6.0f}s "
        f"render={row['mean_render_sec']:6.0f}s "
        f"infer={row['mean_inference_sec']:5.1f}s "
        f"step={row['median_step_sec']*1000:5.0f}ms | "
        f"thr wall={row['throughput_wall_ep_per_min']:5.1f} "
        f"compute={row['throughput_compute_ep_per_min']:5.1f} "
        f"render={row['throughput_render_ep_per_min']:5.1f} ep/min"
    )


def main() -> None:
    """Run main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dirs", nargs="+")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of rows")
    args = parser.parse_args()
    rows = [summarize_run(path) for path in args.output_dirs]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
