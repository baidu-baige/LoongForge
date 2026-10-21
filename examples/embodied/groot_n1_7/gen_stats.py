#!/usr/bin/env python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
Generate meta/stats.json for LeRobot datasets (standalone version of gr00t/data/stats.py).

Usage:
    python gen_stats.py <dataset_path> [<dataset_path> ...]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_GLOB = "data/*/*.parquet"
INFO_FILENAME = "meta/info.json"
STATS_FILENAME = "meta/stats.json"


def compute_stats(dataset_path: Path, num_workers: int = 32) -> dict:
    with open(dataset_path / INFO_FILENAME, "r") as f:
        le_features = json.load(f)["features"]
    features = [f for f in le_features if "float" in le_features[f]["dtype"]]

    parquet_paths = sorted(dataset_path.glob(DATA_GLOB))
    # Files are small and reads are latency-bound on network storage, so read concurrently.
    # executor.map preserves input order, so the result matches a sequential read.
    with ThreadPoolExecutor(num_workers) as executor:
        dfs = list(
            tqdm(
                executor.map(lambda p: pd.read_parquet(p, columns=features), parquet_paths),
                total=len(parquet_paths),
                desc="Loading parquet",
            )
        )
    df = pd.concat(dfs, axis=0)

    stats = {}
    for feature in features:
        print(f"Computing statistics for {feature}...")
        data = np.vstack([np.asarray(x, dtype=np.float32) for x in df[feature]])
        stats[feature] = dict(
            mean=np.mean(data, axis=0).tolist(),
            std=np.std(data, axis=0).tolist(),
            min=np.min(data, axis=0).tolist(),
            max=np.max(data, axis=0).tolist(),
            q01=np.quantile(data, 0.01, axis=0).tolist(),
            q99=np.quantile(data, 0.99, axis=0).tolist(),
        )
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset_paths", nargs="+", type=Path)
    parser.add_argument("--num-workers", type=int, default=32, help="Threads for reading parquet files")
    args = parser.parse_args()

    for dataset_path in args.dataset_paths:
        print(f"Generating stats for {dataset_path}")
        stats = compute_stats(dataset_path, args.num_workers)
        with open(dataset_path / STATS_FILENAME, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Saved {dataset_path / STATS_FILENAME}")


if __name__ == "__main__":
    main()
