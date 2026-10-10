# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the LoongForge VLA evaluation module."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from .report import collect_resource_snapshot


def sample_resources(output_path: str | Path, interval_sec: float = 5.0, duration_sec: float = 60.0) -> Dict[str, Any]:
    """Run sample_resources."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    samples = 0
    with path.open("w", encoding="utf-8") as file:
        while True:
            snapshot = collect_resource_snapshot()
            snapshot["elapsed_sec"] = time.time() - start
            file.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")
            file.flush()
            samples += 1
            if duration_sec <= 0 or time.time() - start >= duration_sec:
                break
            time.sleep(max(interval_sec, 0.1))
    return {"resource_timeseries_path": str(path), "samples": samples, "elapsed_sec": time.time() - start}


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    return parser


def main() -> None:
    """Run main."""
    args = build_argparser().parse_args()
    result = sample_resources(args.output, interval_sec=args.interval_sec, duration_sec=args.duration_sec)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
