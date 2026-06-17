#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

show_help() {
  cat <<USAGE
Usage:
  bash preprocess.sh wan2.1 [output_path]
  bash preprocess.sh wan2.2 [output_path]
  bash preprocess.sh --help

Arguments:
  wan2.1       Preprocess Wan2.1 I2V data with T5 + VAE + CLIP.
  wan2.2       Preprocess Wan2.2 I2V data with T5 + VAE.
  output_path  Optional output directory.
               Default for wan2.1: <dataset_base_parent>/wan_2.1_preprocessed
               Default for wan2.2: <dataset_base_parent>/wan_2.2_preprocessed

Optional environment overrides:
  LOONGFORGE_ROOT        Default: /workspace/wan2.1/LoongForge
  WAN21_MODEL_ROOT       Default: /workspace/Wan2.1-I2V-14B-480P
  WAN22_MODEL_ROOT       Default: /workspace/Wan2.2-I2V-A14B
  DATASET_BASE_PATH      Default: ./dataset/samples
  DATASET_METADATA_PATH  Default: ./dataset/samples/metadata_100.jsonl
  HEIGHT                 Default: 480 (both versions)
  WIDTH                  Default: 832 (both versions)
  NUM_FRAMES             Default: 81 (wan2.1) / 49 (wan2.2)
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  show_help >&2
  exit 1
fi

case "$1" in
  -h|--help)
    show_help
    exit 0
    ;;
  wan2.1|2.1|wan2.2|2.2)
    WAN_VERSION="$1"
    ;;
  *)
    echo "Unsupported version: $1" >&2
    show_help >&2
    exit 1
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOONGFORGE_ROOT=${LOONGFORGE_ROOT:-/workspace/wan2.1/LoongForge}
WAN21_MODEL_ROOT=${WAN21_MODEL_ROOT:-/workspace/Wan2.1-I2V-14B-480P}
WAN22_MODEL_ROOT=${WAN22_MODEL_ROOT:-/workspace/Wan2.2-I2V-A14B}

DATASET_BASE_PATH=${DATASET_BASE_PATH:-"${SCRIPT_DIR}/dataset/samples"}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-"${DATASET_BASE_PATH}/metadata_100.jsonl"}
DATASET_PARENT_DIR=$(cd "$(dirname "${DATASET_BASE_PATH}")" && pwd)
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}

case "${WAN_VERSION}" in
  wan2.1|2.1)
    NUM_FRAMES=${NUM_FRAMES:-81}
    MODEL_T5=${MODEL_T5:-"${WAN21_MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"}
    MODEL_VAE=${MODEL_VAE:-"${WAN21_MODEL_ROOT}/Wan2.1_VAE.pth"}
    MODEL_CLIP=${MODEL_CLIP:-"${WAN21_MODEL_ROOT}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"}
    TOKENIZER_LOCAL_PATH=${TOKENIZER_LOCAL_PATH:-"${WAN21_MODEL_ROOT}/google/umt5-xxl"}
    DEFAULT_OUTPUT_PATH="${DATASET_PARENT_DIR}/wan_2.1_preprocessed"
    MODEL_PATHS="${MODEL_T5},${MODEL_VAE},${MODEL_CLIP}"
    ;;
  wan2.2|2.2)
    NUM_FRAMES=${NUM_FRAMES:-49}
    MODEL_T5=${MODEL_T5:-"${WAN22_MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"}
    MODEL_VAE=${MODEL_VAE:-"${WAN22_MODEL_ROOT}/Wan2.1_VAE.pth"}
    TOKENIZER_LOCAL_PATH=${TOKENIZER_LOCAL_PATH:-"${WAN22_MODEL_ROOT}/google/umt5-xxl"}
    DEFAULT_OUTPUT_PATH="${DATASET_PARENT_DIR}/wan_2.2_preprocessed"
    MODEL_PATHS="${MODEL_T5},${MODEL_VAE}"
    ;;
esac

OUTPUT_PATH=${2:-${OUTPUT_PATH:-"${DEFAULT_OUTPUT_PATH}"}}

echo "Preprocessing ${WAN_VERSION} dataset"
echo "  dataset: ${DATASET_BASE_PATH}"
echo "  metadata: ${DATASET_METADATA_PATH}"
echo "  output: ${OUTPUT_PATH}"
echo "  model_paths: ${MODEL_PATHS}"
echo "  resolution: ${HEIGHT}x${WIDTH}, num_frames: ${NUM_FRAMES}"

cd "${SCRIPT_DIR}"
accelerate launch "${LOONGFORGE_ROOT}/examples/wan/wan_preprocess.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --height "${HEIGHT}" --width "${WIDTH}" --num_frames "${NUM_FRAMES}" \
  --model_paths "${MODEL_PATHS}" \
  --tokenizer_local_path "${TOKENIZER_LOCAL_PATH}" \
  --output_path "${OUTPUT_PATH}"
