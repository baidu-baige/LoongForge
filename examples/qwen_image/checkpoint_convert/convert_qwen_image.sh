#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Convert Qwen-Image-Edit DiT checkpoints between HuggingFace and Megatron DCP.
#   ./convert_qwen_image.sh hg2mcore   # HF diffusers transformer -> Megatron DCP
#   ./convert_qwen_image.sh mcore2hg   # Megatron DCP -> HF diffusers transformer
set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"hg2mcore\" or \"mcore2hg\""
    exit 1
fi
input_string=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HF diffusers "transformer" dir with diffusion_pytorch_model-*.safetensors
HF_CKPT=${HF_CKPT:-"/ssd1/loongfore_data/Qwen/Qwen-Image-Edit-2511/transformer"}
# Megatron DCP (fsdp_dtensor) checkpoint dir
MCORE_CKPT=${MCORE_CKPT:-"/ssd1/loongfore_data/Qwen/qwen_image_edit_2511_mcore"}
# HF output dir for mcore2hg
HF_OUT=${HF_OUT:-"/ssd1/loongfore_data/Qwen/qwen_image_edit_2511_hf_from_mcore"}

if [ "$input_string" == "hg2mcore" ]; then
    echo "convert weight from huggingface to megatron"
    python "$SCRIPT_DIR/convert_checkpoint_hg2mcore.py" \
        --checkpoint_path="$HF_CKPT" \
        --save_path="$MCORE_CKPT"
elif [ "$input_string" == "mcore2hg" ]; then
    echo "convert weight from megatron to huggingface"
    python "$SCRIPT_DIR/convert_checkpoint_mcore2hg.py" \
        --load_path="$MCORE_CKPT" \
        --save_path="$HF_OUT"
else
    echo "Usage: $0 \"hg2mcore\" or \"mcore2hg\""
    exit 1
fi
