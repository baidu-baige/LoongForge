#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Convert Pi05 HuggingFace checkpoint to PyTorch format.
#
# This script converts Pi05 safetensors weights from HuggingFace format to a single
# PyTorch checkpoint file (.pt) compatible with Megatron-LM training.
#
# Usage:
#   # Using default paths (edit LOAD/SAVE below)
#   bash convert_pi05_hf_to_torch.sh
#
#   # Using environment variables
#   LOAD=/path/to/hf_model SAVE=/path/to/output bash convert_pi05_hf_to_torch.sh
#
# Output structure:
#   ${SAVE}/
#     ├── latest_checkpointed_iteration.txt  # contains "release"
#     └── release/
#         └── mp_rank_00/
#             └── model_optim_rng.pt          # Megatron checkpoint

set -euo pipefail

LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-"python3"}

# Input/Output paths - modify as needed
LOAD=${LOAD:-"/workspace/pi05_huggingface/"}
SAVE=${SAVE:-"/workspace/pi05_torch/"}

# Conversion options
DTYPE=${DTYPE:-"fp32"}
PREFIX_ADD=${PREFIX_ADD:-"model."}

echo "=========================================="
echo "Pi05 HuggingFace to PyTorch Checkpoint Convert"
echo "=========================================="
echo "Input:  ${LOAD}"
echo "Output: ${SAVE}"
echo "Dtype:  ${DTYPE}"
echo "Prefix: ${PREFIX_ADD}"
echo "=========================================="

# Create output directory structure
mkdir -p "${SAVE}/release/mp_rank_00"

# Run conversion
"${PYTHON_BIN}" "${LOONGFORGE_PATH}/tools/convert_checkpoint/pi05/convert_hf_to_torch.py" \
    --input "${LOAD}" \
    --output "${SAVE}/release/mp_rank_00/model_optim_rng.pt" \
    --dtype "${DTYPE}" \
    --prefix-add "${PREFIX_ADD}" \
    --megatron-format

# Write checkpoint marker file
echo release > "${SAVE}/latest_checkpointed_iteration.txt"

echo ""
echo "=========================================="
echo "Conversion complete!"
echo "Checkpoint saved to: ${SAVE}"
echo "=========================================="
