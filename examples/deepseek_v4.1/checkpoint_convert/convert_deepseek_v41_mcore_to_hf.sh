#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Convert DeepSeek-V4.1 from Megatron Core format back to HF format (BF16 version)
set -e

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/mcore_checkpoint_convert"

# Input/Output paths (overridable via environment)
LOAD=${LOAD:-"/mnt/cluster/loongforge-ckpt/deepseek_v41/mcore_deepseek_v41_flash/release"}
SAVE=${SAVE:-"/mnt/cluster/loongforge-ckpt/deepseek_v41/rebuilt_hf"}
HF_CONFIG_FILE=${HF_CONFIG_FILE:?Set HF_CONFIG_FILE to the matching native HF config.json}

MODEL_CONFIG_FILE=$LOONGFORGE_PATH/configs/models/deepseek4.1/deepseek_v41_flash.yaml
CONVERT_FILE=$LOONGFORGE_PATH/configs/models/deepseek4.1/ckpt_convert/deepseek_v41_convert.yaml

echo "=== Converting mcore -> HF ==="
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=${TP_SIZE:-8} \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=${EP_SIZE:-8} \
    --expert_tensor_parallel_size=${ETP_SIZE:-1} \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --hf-official-config-file="$HF_CONFIG_FILE" \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm

echo "=== mcore -> HF conversion completed ==="
echo "Output: $SAVE"
