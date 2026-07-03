#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Convert DeepSeek-V4 from HF format to Megatron Core format (FP8 version)
# This script converts BF16 HF weights to FP8 mcore format for training efficiency.
set -e

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

# Input/Output paths (overridable via environment)
HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base"}
SAVE_ROOT=${SAVE_ROOT:-"/mnt/cluster/loongforge-ckpt/deepseek_v4/mcore_deepseek_v4_flash_base_fp8"}

MODEL_CONFIG_FILE=$LOONGFORGE_PATH/configs/models/deepseek4/deepseek_v4_flash_base.yaml
CONVERT_FILE=$LOONGFORGE_PATH/configs/models/deepseek4/ckpt_convert/deepseek_v4_convert.yaml

echo "=== Converting HF -> mcore (FP8) ==="
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=2 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$HF_MODEL_PATH \
    --save_ckpt_path=$SAVE_ROOT \
    --safetensors \
    --max_workers=4 \
    --moe-grouped-gemm \
    --convert_to_fp8

echo "=== HF -> mcore FP8 conversion completed ==="
echo "Output: $SAVE_ROOT"
