#!/usr/bin/env python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Download Kimi K2.6 ModelOpt calibration samples into a local JSONL file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "cnn_dailymail": {
        "config": {"path": "cnn_dailymail", "name": "3.0.0", "split": ["train"]},
        "preprocess": lambda sample: sample["article"],
    },
    "nemotron-post-training-dataset-v2": {
        "config": {
            "path": "nvidia/Nemotron-Post-Training-Dataset-v2",
            "split": ["stem", "chat", "math", "code"],
        },
        "preprocess": lambda sample: "\n".join(turn["content"] for turn in sample["messages"]),
    },
}


def parse_csv_strings(value: str) -> list[str]:
    """Split a comma-separated string into a list of non-empty items."""
    return [item for item in value.split(",") if item]


def parse_csv_ints(value: str) -> list[int]:
    """Split a comma-separated string into a list of integers."""
    return [int(item) for item in value.split(",") if item]


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name aliases to the canonical lower-case key."""
    aliases = {
        "Nemotron-Post-Training-Dataset-v2": "nemotron-post-training-dataset-v2",
        "nvidia/Nemotron-Post-Training-Dataset-v2": "nemotron-post-training-dataset-v2",
    }
    return aliases.get(name, name)


def load_dataset_split(config: dict[str, Any], split: str | None):
    """Load a HuggingFace dataset split in streaming mode, with optional HF token."""
    kwargs = dict(config)
    kwargs["streaming"] = True
    if split is not None:
        kwargs["split"] = split

    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token

    try:
        return load_dataset(**kwargs)
    except TypeError:
        if not token:
            raise
        kwargs.pop("token", None)
        kwargs["use_auth_token"] = token
        return load_dataset(**kwargs)


def get_dataset_samples(dataset_name: str, num_samples: int) -> list[str]:
    """Download ``num_samples`` calibration texts from a registered dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise SystemExit(
            f"Unsupported dataset {dataset_name}; supported: {sorted(DATASET_CONFIGS)}"
        )

    dataset_config = DATASET_CONFIGS[dataset_name]
    config = dataset_config["config"].copy()
    splits = config.pop("split", [None])
    preprocess = dataset_config["preprocess"]

    per_split = [num_samples // len(splits) for _ in splits]
    per_split[-1] += num_samples - sum(per_split)

    samples: list[str] = []
    for split, split_samples in zip(splits, per_split, strict=True):
        dataset = load_dataset_split(config, split)
        for idx, sample in enumerate(dataset):
            if idx >= split_samples:
                break
            samples.append(preprocess(sample))
    return samples


def parse_dataset_plan(dataset_arg: str, size_arg: str) -> list[tuple[str, int]]:
    """Build a list of (dataset_name, num_samples) pairs from CLI arguments."""
    datasets = [normalize_dataset_name(item) for item in parse_csv_strings(dataset_arg)]
    sizes = parse_csv_ints(size_arg)
    if len(sizes) == 1 and len(datasets) > 1:
        sizes = sizes * len(datasets)
    if len(datasets) != len(sizes):
        raise SystemExit(f"CALIB_DATASET and CALIB_SIZE lengths differ: {datasets} vs {sizes}")
    return list(zip(datasets, sizes, strict=True))


def write_samples(dataset_arg: str, size_arg: str, output: Path) -> None:
    """Download calibration samples and write them as JSONL to ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as f:
        for dataset_name, num_samples in parse_dataset_plan(dataset_arg, size_arg):
            samples = get_dataset_samples(dataset_name, num_samples)
            if len(samples) < num_samples:
                raise SystemExit(
                    f"Downloaded too few samples for {dataset_name}: "
                    f"{len(samples)} < {num_samples}"
                )
            for idx, text in enumerate(samples):
                f.write(
                    json.dumps(
                        {"dataset": dataset_name, "index": idx, "text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"Downloaded {len(samples)} sample(s) for {dataset_name}", flush=True)
            total += len(samples)

    print(f"Wrote {total} calibration sample(s) to {output}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset download."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--calib-size", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Entry point: parse arguments and download calibration samples."""
    args = parse_args()
    write_samples(args.dataset, args.calib_size, args.output)


if __name__ == "__main__":
    main()
