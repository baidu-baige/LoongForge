# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Archive episode traces under traces/<run_id>/... from results.jsonl."""

from __future__ import annotations

import argparse
import json

from loongforge.embodied.eval.metrics import archive_traces


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="default")
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    manifest = archive_traces(args.results, args.output_dir, run_id=args.run_id)
    print(json.dumps({"archived_traces": len(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
