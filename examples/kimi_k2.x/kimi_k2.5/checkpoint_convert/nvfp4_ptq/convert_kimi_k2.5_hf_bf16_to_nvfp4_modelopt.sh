#!/bin/bash

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_NVFP4_DIR=${COMMON_NVFP4_DIR:-"$(cd "${SCRIPT_DIR}/../../../nvfp4_ptq" && pwd)"}

# Checkpoint path config
export BF16_HF=${BF16_HF:-"/mnt/cluster/LoongForge/moonshotai/Kimi-K2.5-hf-bf16"}
export SAVE=${SAVE:-"/mnt/cluster/LoongForge/moonshotai/Kimi-K2.5-nvfp4-modelopt"}

# Model-Optimizer code
export MODELOPT_REPO_REF=${MODELOPT_REPO_REF:-"b02e8885509c53b4e187f9fd5f56c5497e937d7e"}
export MODELOPT_REPO=${MODELOPT_REPO:-"${TMPDIR:-/tmp}/kimi_modelopt_repos/Model-Optimizer-kimi-k2.5-b02e888"}

# PTQ config
export OFFICIAL_QUANT_CONFIG=${OFFICIAL_QUANT_CONFIG:-"${SCRIPT_DIR}/kimi_k2.5_nvfp4_official_modelopt_config.json"}
export CALIB_DATASET=${CALIB_DATASET:-"cnn_dailymail"}
export CALIB_SIZE=${CALIB_SIZE:-"512"}
# The JSONL can be a larger prefetched pool; CALIB_SIZE controls how many samples are consumed.
export CALIB_DATASET_JSONL=${CALIB_DATASET_JSONL:-"/mnt/cluster/LoongForge/modelopt_calib_datasets/cnn_dailymail_512.jsonl"}

# Model placement and memory behavior
export INFERENCE_TENSOR_PARALLEL=${INFERENCE_TENSOR_PARALLEL:-1}
export INFERENCE_PIPELINE_PARALLEL=${INFERENCE_PIPELINE_PARALLEL:-1}
export LOW_MEMORY_MODE=${LOW_MEMORY_MODE:-0}

exec bash "${COMMON_NVFP4_DIR}/convert_hf_bf16_to_nvfp4_modelopt.sh" "$@"
