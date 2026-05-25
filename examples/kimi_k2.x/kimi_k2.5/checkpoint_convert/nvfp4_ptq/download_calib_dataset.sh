#! /bin/bash

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-"python"}
CALIB_DATASET=${CALIB_DATASET:-"cnn_dailymail"}
CALIB_SIZE=${CALIB_SIZE:-512}
CALIB_DATASET_CACHE_DIR=${CALIB_DATASET_CACHE_DIR:-"/mnt/cluster/LoongForge/modelopt_calib_datasets"}
CALIB_DATASET_TAG=$(printf '%s_%s' "$CALIB_DATASET" "$CALIB_SIZE" | tr -c '[:alnum:]._-' '_')
CALIB_DATASET_JSONL=${CALIB_DATASET_JSONL:-"$CALIB_DATASET_CACHE_DIR/${CALIB_DATASET_TAG}.jsonl"}
FORCE_CALIB_DATASET_DOWNLOAD=${FORCE_CALIB_DATASET_DOWNLOAD:-0}

ENABLE_HF_PROXY=${ENABLE_HF_PROXY:-1}
HF_PROXY_URL=${HF_PROXY_URL:-"http://agent.baidu.com:8891"}

export HF_HOME=${HF_HOME:-"$CALIB_DATASET_CACHE_DIR/hf_home"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$CALIB_DATASET_CACHE_DIR/hf_datasets"}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}

if [ "$ENABLE_HF_PROXY" = "1" ]; then
    export http_proxy=${http_proxy:-"$HF_PROXY_URL"}
    export https_proxy=${https_proxy:-"$HF_PROXY_URL"}
    export HTTP_PROXY=${HTTP_PROXY:-"$http_proxy"}
    export HTTPS_PROXY=${HTTPS_PROXY:-"$https_proxy"}
fi

mkdir -p "$CALIB_DATASET_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$(dirname "$CALIB_DATASET_JSONL")"

echo "===== Download ModelOpt calibration dataset ($(date '+%F %T')) ====="
echo ">> SCRIPT_DIR:              $SCRIPT_DIR"
echo ">> CALIB_DATASET:           $CALIB_DATASET"
echo ">> CALIB_SIZE:              $CALIB_SIZE"
echo ">> CALIB_DATASET_JSONL:     $CALIB_DATASET_JSONL"
echo ">> CALIB_DATASET_CACHE_DIR: $CALIB_DATASET_CACHE_DIR"
echo ">> HF_HOME:                 $HF_HOME"
echo ">> HF_HUB_CACHE:            $HF_HUB_CACHE"
echo ">> HF_DATASETS_CACHE:       $HF_DATASETS_CACHE"
echo ">> ENABLE_HF_PROXY:         $ENABLE_HF_PROXY"
if [ "$ENABLE_HF_PROXY" = "1" ]; then
    echo ">> http_proxy:              ${http_proxy:-<unset>}"
    echo ">> https_proxy:             ${https_proxy:-<unset>}"
fi

required_samples=$(
    "$PYTHON_BIN" - "$CALIB_SIZE" <<'PY'
import sys

print(sum(int(item) for item in str(sys.argv[1]).split(",") if item))
PY
)

existing_samples=0
if [ -f "$CALIB_DATASET_JSONL" ]; then
    existing_samples=$(wc -l < "$CALIB_DATASET_JSONL")
fi

if [ "$FORCE_CALIB_DATASET_DOWNLOAD" != "1" ] && [ "$existing_samples" -ge "$required_samples" ]; then
    echo "Calibration dataset JSONL already has $existing_samples sample(s), need $required_samples; skip download."
    exit 0
fi

tmp_jsonl="${CALIB_DATASET_JSONL}.tmp.$$"
rm -f "$tmp_jsonl"
trap 'rm -f "$tmp_jsonl"' EXIT

"$PYTHON_BIN" - "$CALIB_DATASET" "$CALIB_SIZE" "$tmp_jsonl" <<'PY'
import json
import sys
from pathlib import Path

from modelopt.torch.utils import dataset_utils

dataset_arg = sys.argv[1]
size_arg = sys.argv[2]
output = Path(sys.argv[3])



def parse_csv_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def normalize_dataset_name(name: str) -> str:
    aliases = {
        "Nemotron-Post-Training-Dataset-v2": "nemotron-post-training-dataset-v2",
        "nvidia/Nemotron-Post-Training-Dataset-v2": "nemotron-post-training-dataset-v2",
    }
    return aliases.get(name, name)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


datasets = [normalize_dataset_name(item) for item in parse_csv_strings(dataset_arg)]
sizes = parse_csv_ints(size_arg)
if len(sizes) == 1 and len(datasets) > 1:
    sizes = sizes * len(datasets)
if len(datasets) != len(sizes):
    raise SystemExit(f"CALIB_DATASET and CALIB_SIZE lengths differ: {datasets} vs {sizes}")

output.parent.mkdir(parents=True, exist_ok=True)
total = 0
with output.open("w", encoding="utf-8") as f:
    for dataset_name, num_samples in zip(datasets, sizes):
        samples = dataset_utils._get_dataset_samples(dataset_name, num_samples)
        if len(samples) < num_samples:
            raise SystemExit(
                f"Downloaded too few samples for {dataset_name}: {len(samples)} < {num_samples}"
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
PY

mv "$tmp_jsonl" "$CALIB_DATASET_JSONL"
trap - EXIT
echo "Calibration dataset ready: $CALIB_DATASET_JSONL"
