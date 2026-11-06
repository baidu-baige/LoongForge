# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Post-run coverage validation for a coordinated evaluation.

A coordinated run is only trustworthy if the episodes the plan asked for all ran
and no episode was produced twice. ``_aggregate`` merely dedups and counts
successes, so a dropped or re-run unit would silently skew the rate.

Extra episodes are allowed: benchmarks that skip unstable seeds (RoboTwin's
``UnStableError``) replace them with later seeds, which fall outside the fixed
seed range in the plan. They are reported as ``unexpected`` for visibility but
do not fail the run.

The check is deliberately benchmark-agnostic: it never expands a
``(suite, task, episode)`` grid (that only fits LIBERO). Instead each shard
declares how many episodes it must produce, and the runner's own ``episode_id``
is the identity. "Duplicate" means two *different* shards claimed the same
episode -- a re-run of the same shard is an expected overlap that dedup handles.

This module only *reports*; the coordinator turns a short shard into a repair
round rather than aborting the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from loongforge.embodied.eval.metrics.results import INFRA_FAILURE_REASONS

# Failure reasons that mean "the episode did not actually run" (see
# libero_runner._classify_exception + the retry set). Shared with the metrics
# layer so the coordinator and the per-run statistics cannot drift apart.


def read_records_by_unit(output_dir: Path, n_units: int) -> List[List[Dict[str, Any]]]:
    """Read each unit's ``results.jsonl``; index i corresponds to worker_i."""
    per_unit: List[List[Dict[str, Any]]] = []
    for i in range(n_units):
        path = output_dir / f"worker_{i}" / "results.jsonl"
        records: List[Dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "episode_id" in record:
                    records.append(record)
        per_unit.append(records)
    return per_unit


def short_shards(
    expected_by_shard: Dict[str, int],
    shard_ids: Sequence[str],
    records_by_unit: Sequence[Sequence[Dict[str, Any]]],
) -> List[str]:
    """Shard ids whose unit produced fewer distinct episodes than it declared."""
    produced: Dict[str, set] = {}
    for shard_id, records in zip(shard_ids, records_by_unit):
        produced.setdefault(shard_id, set()).update(str(r.get("episode_id")) for r in records)
    return [shard_id for shard_id, want in expected_by_shard.items() if len(produced.get(shard_id, ())) < want]


def validate(
    expected_by_shard: Dict[str, int],
    shard_ids: Sequence[str],
    records_by_unit: Sequence[Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compare what the plan asked for against what the run actually produced."""
    seen: Dict[str, Dict[str, Any]] = {}  # episode_id -> record (last write wins)
    owners: Dict[str, set] = {}  # episode_id -> shard ids that produced it
    unkeyed = 0
    n_records = 0
    for shard_id, records in zip(shard_ids, records_by_unit):
        for record in records:
            episode_id = record.get("episode_id")
            if episode_id is None:
                unkeyed += 1
                continue
            n_records += 1
            episode_id = str(episode_id)
            owners.setdefault(episode_id, set()).add(shard_id)
            seen[episode_id] = record

    duplicates = sorted(eid for eid, who in owners.items() if len(who) > 1)
    expected = sum(expected_by_shard.values())
    actual = len(seen)
    missing = max(0, expected - actual)
    unexpected = max(0, actual - expected)

    failures: Dict[str, int] = {}
    infra_failed = 0
    successes = 0
    for record in seen.values():
        if record.get("success"):
            successes += 1
            continue
        reason = str(record.get("failure_reason") or "unknown")
        failures[reason] = failures.get(reason, 0) + 1
        if reason in INFRA_FAILURE_REASONS:
            infra_failed += 1

    return {
        # Extra episodes are legitimate replacements for skipped unstable seeds
        # (e.g. RoboTwin), so they are reported but do not fail the run.
        "ok": missing == 0 and not duplicates and unkeyed == 0,
        "expected": expected,
        "actual": actual,
        "records": n_records,
        "missing": missing,
        "duplicates": len(duplicates),
        "unexpected": unexpected,
        "unkeyed": unkeyed,
        "successes": successes,
        "success_rate": (successes / actual) if actual else None,
        "failures": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        "infra_failed": infra_failed,
        "duplicate_examples": duplicates[:10],
    }


def format_report(report: Dict[str, Any]) -> str:
    """One-paragraph human summary of a validation report."""
    status = "OK" if report["ok"] else "INCOMPLETE"
    parts = [
        f"coverage {status}: {report['actual']}/{report['expected']} episodes",
        f"missing={report['missing']}",
        f"duplicates={report['duplicates']}",
        f"unexpected={report['unexpected']}",
    ]
    if report["unkeyed"]:
        parts.append(f"unkeyed={report['unkeyed']}")
    if report["infra_failed"]:
        parts.append(f"infra_failed={report['infra_failed']}")
    if report["success_rate"] is not None:
        parts.append(f"success_rate={report['success_rate']:.4f}")
    return " | ".join(parts)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate episode coverage of a coordinated eval run.")
    parser.add_argument("--output-dir", required=True, help="coordinator output dir (holds worker_*/results.jsonl)")
    parser.add_argument("--plan", required=True, help="plan.json written by the coordinator")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when coverage is incomplete")
    return parser


def main() -> None:
    """Run main."""
    args = _build_argparser().parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    expected_by_shard = {shard["id"]: int(shard["expected"]) for shard in plan["shards"]}
    launched = plan["launched"]
    report = validate(expected_by_shard, launched, read_records_by_unit(Path(args.output_dir), len(launched)))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
