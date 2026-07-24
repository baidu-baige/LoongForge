# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generate M4 statistics, artifacts manifest, and repro report from results.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loongforge.embodied.eval.metrics import write_eval_report, write_suite_summary_csv, write_summary_csv
from loongforge.embodied.eval.metrics.results import load_jsonl


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for report.md and manifests")
    parser.add_argument("--summary", default=None, help="Optional path for summary.csv")
    parser.add_argument("--suite-summary", default=None, help="Optional path for suite_summary.csv")
    parser.add_argument("--title", default="VLA Eval Report")
    parser.add_argument("--artifact-limit", type=int, default=20)
    parser.add_argument("--min-episodes-per-task", type=int, default=1)
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    records = load_jsonl(args.results)
    if args.summary:
        write_summary_csv(args.summary, records)
    else:
        output_dir = Path(args.output_dir)
        write_summary_csv(output_dir / "summary.csv", records)
    if args.suite_summary:
        write_suite_summary_csv(args.suite_summary, records, min_episodes=args.min_episodes_per_task)
    else:
        output_dir = Path(args.output_dir)
        write_suite_summary_csv(output_dir / "suite_summary.csv", records, min_episodes=args.min_episodes_per_task)
    report = write_eval_report(
        args.results,
        args.output_dir,
        title=args.title,
        artifact_limit=args.artifact_limit,
        min_episodes_per_task=args.min_episodes_per_task,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
