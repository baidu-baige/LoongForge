#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Decode released DeepSeek-V4.1 FP8/FP4 weights into BF16 MCore weights.
set -e

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/mcore_checkpoint_convert"

# Input/Output paths (overridable via environment)
HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"}
SAVE_ROOT=${SAVE_ROOT:-"/mnt/cluster/loongforge-ckpt/deepseek_v41/mcore_deepseek_v41_flash"}

MODEL_CONFIG_FILE=$LOONGFORGE_PATH/configs/models/deepseek4.1/deepseek_v41_flash.yaml
CONVERT_FILE=$LOONGFORGE_PATH/configs/models/deepseek4.1/ckpt_convert/deepseek_v41_convert.yaml

echo "=== Converting HF -> mcore (BF16) ==="
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=${TP_SIZE:-8} \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=${EP_SIZE:-8} \
    --expert_tensor_parallel_size=${ETP_SIZE:-1} \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$HF_MODEL_PATH \
    --save_ckpt_path=$SAVE_ROOT \
    --safetensors \
    --distributed_convert \
    --hf-dequantize-mxfp4 \
    --hf-dequantize-dtype=bfloat16 \
    --max_workers=1 \
    --moe-grouped-gemm \
    --no-fp8-param-gather

echo "=== HF -> mcore BF16 conversion completed ==="
echo "Output: $SAVE_ROOT"
