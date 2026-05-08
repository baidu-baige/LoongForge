#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Convert gr00tn1.6 HuggingFace checkpoint to PyTorch format.
#
# This script converts gr00tn1.6 safetensors weights from HuggingFace format to a single
# PyTorch checkpoint file (.pt) compatible with Megatron-LM training.
#
# Usage:
#   # Using default paths (edit LOAD/SAVE below)
#   bash convert_g r 00 t_hf_to_torch.sh
#
#   # Using environment variables
#   LOAD=/path/to/hf_model SAVE=/path/to/output bash convert_gr00tn1.6_hf_to_torch.sh
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
LOAD=${LOAD:-"/workspace/gr00tn1.6_huggingface/"}
SAVE=${SAVE:-"/workspace/gr00tn1.6_torch/"}


echo "=========================================="
echo "gr00tn1.6 HuggingFace to PyTorch Checkpoint Convert"
echo "=========================================="
echo "Input:  ${LOAD}"
echo "Output: ${SAVE}"
echo "=========================================="

# Create output directory structure
mkdir -p "${SAVE}/release/mp_rank_00"


"${PYTHON_BIN}" "${LOONGFORGE_PATH}/tools/convert_checkpoint/pi05/convert_hf_to_torch.py" \
    --input "${LOAD}" \
    --output "${SAVE}/release/mp_rank_00/model_optim_rng.pt" \
    --megatron-format

# Write checkpoint marker file
echo release > "${SAVE}/latest_checkpointed_iteration.txt"

echo ""
echo "=========================================="
echo "Conversion complete!"
echo "Checkpoint saved to: ${SAVE}"
echo "=========================================="
