#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Ported (minimal) from wall-x (scripts/compute_norm_stats.py). Only the code
# path required to produce ``libero_norm_stats.json`` is kept: read
# ``observation.state`` and ``action`` columns from a local LeRobot v2.1/v3
# parquet dataset and write per-dim mean/std/q01/q99 in the layout expected by
# ``wall_oss_0_5/transforms/wall_oss_0_5_utils.py::load_norm_stats``.
"""Compute LeRobot normalization stats (mean, std, q01, q99).

Output JSON layout::

    {"norm_stats": {
        "observation.state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
        "action":            {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
    }}

Usage::

    python -m loongforge.embodied.data.datasets.wall_oss_0_5.compute_norm_stats \\
        --data-root /path/to/lerobot_dataset \\
        --output-path /path/to/libero_norm_stats.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.dataset as pa_ds


def compute_vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    """Compute vector stats."""
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def load_state_action_arrays(
    data_root: Path,
    state_key: str,
    action_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read state/action columns from LeRobot parquet files (no video decode)."""
    root = data_root.expanduser().resolve()
    paths = sorted((root / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")

    logging.info(
        "Reading parquet columns %r, %r from %d file(s)",
        state_key,
        action_key,
        len(paths),
    )
    table = pa_ds.dataset([str(p) for p in paths], format="parquet").to_table(
        columns=[state_key, action_key]
    )
    states = np.asarray(table[state_key].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[action_key].to_pylist(), dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(-1, 1)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    logging.info(
        "  frames=%d state_dim=%d action_dim=%d",
        len(states),
        states.shape[1],
        actions.shape[1],
    )
    return states, actions


def compute_norm_stats(
    data_root: Path,
    output_path: Path,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> dict[str, dict]:
    """Compute norm stats."""
    states, actions = load_state_action_arrays(data_root, state_key, action_key)
    norm_stats = {
        state_key: compute_vector_stats(states),
        action_key: compute_vector_stats(actions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"norm_stats": norm_stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return norm_stats


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Compute norm stats for a local LeRobot v2.1/v3 dataset.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Local LeRobot dataset directory (containing data/*/*.parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="observation.state",
        help="Dataset column for proprioception (default: observation.state)",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="Dataset column for action (default: action)",
    )
    return parser.parse_args()


def main() -> None:
    """Main."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output_path)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found: {data_root}")

    logging.info("dataset: %s", data_root)
    logging.info("output:  %s", output_path)

    norm_stats = compute_norm_stats(
        data_root=data_root,
        output_path=output_path,
        state_key=args.state_key,
        action_key=args.action_key,
    )

    for key, stats in norm_stats.items():
        logging.info("  %s: dim=%d", key, len(stats["mean"]))

    logging.info("Saved norm stats to %s", output_path)


if __name__ == "__main__":
    main()
