#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build a RoboTwin dataset view whose ``meta/stats.json`` uses pi-space statistics.

When ``data.use_aloha_pi_delta_actions: true`` is enabled, ``AlohaPiDeltaActionTransform``
converts raw env-space 14D joint actions into pi-space deltas and rewrites
``observation.state`` into pi-space proprioception BEFORE ``ActionTransform`` and
``StateDiscretizationTransform`` run. Therefore, the dataset's ``meta/stats.json``
must hold the pi-space statistics for both ``action`` and ``observation.state``
(matching ``examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json``).

This script creates a lightweight destination directory that symlinks ``data/``,
``videos/``, and all ``meta/`` files except ``stats.json`` from the source dataset,
and writes a new ``meta/stats.json`` with ``action`` and ``observation.state``
replaced by the pi-space statistics.

Usage:
    python examples/embodied/pi05/build_pidelta_stats.py \\
        --src /path/to/robotwin_unified \\
        --dst /path/to/robotwin_unified_pidelta
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PI_STATS = REPO_ROOT / "examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json"


def _stat_block(src_block: Dict[str, Any], count_ref: Any) -> Dict[str, Any]:
    out = {k: list(src_block[k]) for k in ("mean", "std", "min", "max", "q01", "q99") if k in src_block}
    if count_ref is not None:
        out["count"] = copy.deepcopy(count_ref)
    return out


def build_pidelta_dataset_view(
    src_dir: str | os.PathLike,
    dst_dir: str | os.PathLike,
    pi_stats_path: str | os.PathLike = DEFAULT_PI_STATS,
) -> Path:
    """Create a symlinked dataset view at ``dst_dir`` with pi-space ``meta/stats.json``."""
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    pi_stats_file = Path(pi_stats_path).resolve()

    if not (src / "meta" / "stats.json").is_file():
        raise FileNotFoundError(f"Source stats.json not found: {src / 'meta' / 'stats.json'}")
    if not pi_stats_file.is_file():
        raise FileNotFoundError(f"Pi-space stats file not found: {pi_stats_file}")

    raw_stats = json.loads((src / "meta" / "stats.json").read_text(encoding="utf-8"))
    pi_blob = json.loads(pi_stats_file.read_text(encoding="utf-8"))
    pi_norm = pi_blob.get("norm_stats", pi_blob)
    if "actions" not in pi_norm or "state" not in pi_norm:
        raise KeyError(f"Expected 'actions' and 'state' in {pi_stats_file}")

    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    for top_entry in sorted(src.iterdir()):
        if top_entry.name == "meta":
            continue
        link = dst / top_entry.name
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                continue
        link.symlink_to(top_entry)

    for meta_entry in sorted((src / "meta").iterdir()):
        if meta_entry.name == "stats.json":
            continue
        link = dst_meta / meta_entry.name
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                continue
        link.symlink_to(meta_entry)

    new_stats = copy.deepcopy(raw_stats)
    count_ref = raw_stats.get("action", {}).get("count")
    new_stats["action"] = _stat_block(pi_norm["actions"], count_ref)
    state_count = raw_stats.get("observation.state", {}).get("count", count_ref)
    new_stats["observation.state"] = _stat_block(pi_norm["state"], state_count)

    stats_out = dst_meta / "stats.json"
    if stats_out.is_symlink():
        stats_out.unlink()
    stats_out.write_text(json.dumps(new_stats, indent=2) + "\n", encoding="utf-8")
    return stats_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pi-space dataset view for RoboTwin pi0.5 training.")
    parser.add_argument("--src", required=True, help="Source robotwin_unified dataset directory.")
    parser.add_argument("--dst", required=True, help="Destination directory for the pi-space dataset view.")
    parser.add_argument(
        "--pi-stats",
        default=str(DEFAULT_PI_STATS),
        help="Path to pi05_robotwin2_dataset_stats.json.",
    )
    args = parser.parse_args()

    stats_out = build_pidelta_dataset_view(args.src, args.dst, args.pi_stats)
    print(f"Wrote {stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
