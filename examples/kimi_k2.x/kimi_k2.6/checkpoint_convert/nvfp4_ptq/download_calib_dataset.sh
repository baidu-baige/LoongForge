#! /bin/bash

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-"python"}
CALIB_DATASET=${CALIB_DATASET:-"cnn_dailymail,nemotron-post-training-dataset-v2"}
CALIB_SIZE=${CALIB_SIZE:-"512,512"}
CALIB_DATASET_CACHE_DIR=${CALIB_DATASET_CACHE_DIR:-"/mnt/cluster/LoongForge/modelopt_calib_datasets"}
CALIB_DATASET_TAG=$(printf '%s_%s' "$CALIB_DATASET" "$CALIB_SIZE" | tr -c '[:alnum:]._-' '_')
CALIB_DATASET_JSONL=${CALIB_DATASET_JSONL:-"$CALIB_DATASET_CACHE_DIR/${CALIB_DATASET_TAG}.jsonl"}
FORCE_CALIB_DATASET_DOWNLOAD=${FORCE_CALIB_DATASET_DOWNLOAD:-0}
ALLOW_HF_TOKENLESS_GATED_DATASETS=${ALLOW_HF_TOKENLESS_GATED_DATASETS:-0}

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

if [[ "$CALIB_DATASET" == *"Nemotron-Post-Training-Dataset-v2"* || "$CALIB_DATASET" == *"nemotron-post-training-dataset-v2"* ]]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        if [ "$ALLOW_HF_TOKENLESS_GATED_DATASETS" != "1" ]; then
            echo "ERROR: nemotron-post-training-dataset-v2 is gated, but HF_TOKEN is not set." >&2
            echo "Set HF_TOKEN to a Hugging Face token with access to nvidia/Nemotron-Post-Training-Dataset-v2." >&2
            echo "For a CNN-only smoke dataset, run:" >&2
            echo "  CALIB_DATASET=cnn_dailymail CALIB_SIZE=512 $0" >&2
            exit 1
        fi
        echo "WARNING: nemotron-post-training-dataset-v2 is gated and HF_TOKEN is not set." >&2
        echo "         Continuing because ALLOW_HF_TOKENLESS_GATED_DATASETS=1." >&2
    fi
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

required_samples=0
IFS=',' read -ra calib_size_items <<< "$CALIB_SIZE"
for item in "${calib_size_items[@]}"; do
    if [ -n "$item" ]; then
        required_samples=$((required_samples + item))
    fi
done

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

"$PYTHON_BIN" "${SCRIPT_DIR}/download_calib_dataset.py" \
    --dataset "$CALIB_DATASET" \
    --calib-size "$CALIB_SIZE" \
    --output "$tmp_jsonl"

mv "$tmp_jsonl" "$CALIB_DATASET_JSONL"
trap - EXIT
echo "Calibration dataset ready: $CALIB_DATASET_JSONL"
