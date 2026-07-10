#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# ------------------------------------------------------------------------
# Qwen-Image-Edit-2511 数据预处理脚本
#
# Python dependencies required by qwen_image_preprocess.py:
#   diffsynth    == 2.0.6   (pip install diffsynth==2.0.6 --no-deps)
#   torch        >= 2.4
#   accelerate   >= 1.0
#   transformers >= 5.0     (needs Qwen2Tokenizer + Qwen2VLProcessor)
#   Pillow       >= 10.0
#   pandas       >= 2.0     (only used when metadata is .csv)
#   tqdm         >= 4.60
#
# Note: install the package with --no-deps to avoid pulling in an incompatible
# torchaudio that would break the torch build in the container. The preprocessing
# script internally stubs torchaudio (not needed for images).
#
# Set INSTALL_DEPS=1 to have this script pip-install the dependencies.
# ------------------------------------------------------------------------

show_help() {
  cat <<USAGE
Usage:
  QWEN_IMAGE_MODEL_ROOT=/path/to/Qwen-Image-Edit-2511 \\
    bash preprocess.sh qwen-image-edit-2511 [output_path]

  bash preprocess.sh --help

Quick start (first time, auto-install deps):
  INSTALL_DEPS=1 \\
  QWEN_IMAGE_MODEL_ROOT=/path/to/Qwen-Image-Edit-2511 \\
  DATASET_BASE_PATH=/path/to/dataset \\
  DATASET_METADATA_PATH=/path/to/dataset/metadata.json \\
    bash preprocess.sh qwen-image-edit-2511 /path/to/output_cache

Model root directory structure (QWEN_IMAGE_MODEL_ROOT):
  Qwen-Image-Edit-2511/
  ├── text_encoder/    # *.safetensors (Qwen2.5-VL text encoder)
  ├── vae/             # diffusion_pytorch_model.safetensors
  ├── tokenizer/       # tokenizer config & vocab files
  └── processor/       # Qwen2VL processor config

Metadata file format (json/jsonl/csv), one row per sample:
  [
    {
      "image": "target.jpg",
      "edit_image": ["ref1.jpg", "ref2.jpg"],
      "prompt": "Change the color of the dress..."
    }
  ]

Arguments:
  qwen-image-edit-2511  Preprocess Qwen-Image-Edit-2511 data with the
                        text encoder + VAE. Produces the AIAK flat-dict cache
                        consumed by loongforge/train/diffusion/pretrain_qwen_image.py.
  output_path           Optional output directory.
                        Default: <dataset_base_parent>/qwen_image_edit_2511_preprocessed

Environment variables:
  QWEN_IMAGE_MODEL_ROOT  Model root dir (required, see structure above)
  LOONGFORGE_ROOT        Default: auto-detected from script location
  DATASET_BASE_PATH      Default: ./data/samples
  DATASET_METADATA_PATH  Default: \${DATASET_BASE_PATH}/metadata.json
  MAX_PIXELS             Default: 1048576  (1MP)
  SEED                   Default: 1234
  TIMESTEP_ID            Default: 321
  TORCH_DTYPE            Default: bf16 (bf16/fp16/fp32)
  NUM_PROCESSES          Default: 1 (set >1 to enable multi-GPU preprocessing)
  TILED_VAE              Set to any non-empty value to enable tiled VAE
  INSTALL_DEPS           Set to 1 to pip-install dependencies before running
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
  qwen-image-edit-2511|edit-2511|2511)
    MODE="qwen-image-edit-2511"
    ;;
  *)
    echo "Unsupported preprocess mode: $1" >&2
    show_help >&2
    exit 1
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOONGFORGE_ROOT=${LOONGFORGE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
QWEN_IMAGE_MODEL_ROOT=${QWEN_IMAGE_MODEL_ROOT:-""}

if [[ -z "${QWEN_IMAGE_MODEL_ROOT}" ]]; then
  echo "Error: QWEN_IMAGE_MODEL_ROOT is not set." >&2
  echo "Please set it to the Qwen-Image-Edit-2511 model checkpoint directory." >&2
  exit 1
fi

DATASET_BASE_PATH=${DATASET_BASE_PATH:-"${SCRIPT_DIR}/data/samples"}
DATASET_METADATA_PATH=${DATASET_METADATA_PATH:-"${DATASET_BASE_PATH}/metadata.json"}
DATASET_PARENT_DIR=$(cd "$(dirname "${DATASET_BASE_PATH}")" && pwd)
DEFAULT_OUTPUT_PATH="${DATASET_PARENT_DIR}/qwen_image_edit_2511_preprocessed"
OUTPUT_PATH=${2:-${OUTPUT_PATH:-"${DEFAULT_OUTPUT_PATH}"}}

MAX_PIXELS=${MAX_PIXELS:-1048576}
SEED=${SEED:-1234}
TIMESTEP_ID=${TIMESTEP_ID:-321}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
NUM_PROCESSES=${NUM_PROCESSES:-8}

# Optionally install Python dependencies.
if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  echo "Installing dependencies..."
  python3 -m pip install --no-input diffsynth==2.0.6 --no-deps
  python3 -m pip install --no-input accelerate Pillow pandas tqdm
fi

# Fail fast if diffsynth is not installed.
if ! python3 -c "import importlib.metadata; importlib.metadata.version('diffsynth')" >/dev/null 2>&1; then
  cat >&2 <<EOF
Error: 'diffsynth' package is not installed.
Please install it first:
    pip install diffsynth==2.0.6 --no-deps
Or run this script with INSTALL_DEPS=1 to auto-install.
EOF
  exit 1
fi

echo "Preprocessing ${MODE} dataset"
echo "  dataset:  ${DATASET_BASE_PATH}"
echo "  metadata: ${DATASET_METADATA_PATH}"
echo "  output:   ${OUTPUT_PATH}"
echo "  model:    ${QWEN_IMAGE_MODEL_ROOT}"
echo "  max_pixels=${MAX_PIXELS}, seed=${SEED}, timestep_id=${TIMESTEP_ID}, dtype=${TORCH_DTYPE}, num_processes=${NUM_PROCESSES}"

TILED_ARG=()
if [[ -n "${TILED_VAE:-}" ]]; then
  TILED_ARG=(--tiled_vae)
fi

cd "${SCRIPT_DIR}"
accelerate launch --num_processes "${NUM_PROCESSES}" "${LOONGFORGE_ROOT}/examples/qwen_image/qwen_image_preprocess.py" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --model_root "${QWEN_IMAGE_MODEL_ROOT}" \
  --output_path "${OUTPUT_PATH}" \
  --max_pixels "${MAX_PIXELS}" \
  --seed "${SEED}" \
  --timestep_id "${TIMESTEP_ID}" \
  --torch_dtype "${TORCH_DTYPE}" \
  "${TILED_ARG[@]+"${TILED_ARG[@]}"}"
