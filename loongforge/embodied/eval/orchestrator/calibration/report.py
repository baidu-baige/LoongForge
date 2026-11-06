# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build the calibration report for the embodied eval N-sweep.

The calibration sweep runs one policy server against N sims for several N and
records, per N, the throughput and resource occupancy. This module turns that
sweep plus the server service-time distribution into the report that drives the
parallel-config decision (report.json / report.md / env.json).

Design notes:
- The **primary** saturation signal is the throughput curve plus the server busy
  fraction (requests x mean service time / wall). ``nvidia-smi`` GPU utilization
  is reported only as a side signal: it is the fraction of time during which a
  kernel was executing, not how many SMs were busy.
- Input format is decoupled from any particular sweep harness: ``--sweep`` is a
  JSON list of per-N records, ``--timing`` is the policy server's timing JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_report(
    sweep: List[Dict[str, Any]],
    service_time: Dict[str, Any],
    env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach derived fields (server busy fraction) and assemble the report dict."""
    default_ms = float(service_time.get("mean") or 0.0)
    rows: List[Dict[str, Any]] = []
    for record in sweep:
        row = dict(record)
        requests = row.get("requests")
        wall = row.get("wall_sec") or 0.0
        # prefer the row's own measured service time (each N has its own timing.json)
        mean_ms = float(row.get("service_mean_ms") or default_ms)
        row["server_busy_frac"] = (
            round(requests * mean_ms / 1000.0 / wall, 3) if requests and wall else None
        )
        rows.append(row)
    return {
        "kind": "m0_calibration",
        "service_time_ms": service_time,
        "sweep": rows,
        "env": env or {},
    }


def _row_line(row: Dict[str, Any]) -> str:
    busy = row.get("server_busy_frac")
    busy_str = f"{busy:.0%}" if isinstance(busy, (int, float)) else "n/a"
    return (
        f"| {row.get('N')} | {row.get('wall_sec')} | {row.get('throughput_ep_per_min')} | "
        f"{row.get('gpu_util_mean')}/{row.get('gpu_util_max')} | "
        f"{row.get('mem_util_mean')}/{row.get('mem_util_max')} | "
        f"{row.get('cpu_busy_mean')}/{row.get('cpu_busy_max')} | {busy_str} |"
    )


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report as a human/agent-readable markdown summary."""
    st = report.get("service_time_ms", {})
    lines = [
        "# Calibration report (M0 N-sweep)",
        "",
        f"- service time: `mean={st.get('mean', 0):.1f}ms p50={st.get('p50', 0):.4f}ms "
        f"p95={st.get('p95', 0):.0f}ms p99={st.get('p99', 0):.0f}ms max={st.get('max', 0):.0f}ms`",
        "",
        "| N | wall(s) | throughput(ep/min) | util.gpu mean/max | util.memory mean/max | "
        "CPU busy mean/max | server busy frac |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_row_line(r) for r in report.get("sweep", [])]
    lines += [
        "",
        "## How to read",
        "",
        "- Primary signal: **throughput plateau** + **server busy fraction**.",
        "- Do NOT use `nvidia-smi utilization.gpu` as saturation: it is the fraction of time "
        "a kernel was executing, not SM occupancy.",
        "- The binding resource is whichever approaches 100% first as N grows.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run main."""
    parser = argparse.ArgumentParser(description="Build the M0 calibration report.")
    parser.add_argument("--sweep", required=True, help="JSON list of per-N sweep records")
    parser.add_argument("--timing", required=True, help="policy server timing JSON")
    parser.add_argument("--env", default="", help="optional env facts JSON")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    sweep = json.loads(Path(args.sweep).read_text())
    service_time = json.loads(Path(args.timing).read_text())
    env = json.loads(Path(args.env).read_text()) if args.env else {}

    report = build_report(sweep, service_time, env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(render_markdown(report))


if __name__ == "__main__":
    main()
