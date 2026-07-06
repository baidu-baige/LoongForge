#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Convert DeepSeek-V4 from Megatron Core format back to HF format (BF16 version)
set -e

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

# Input/Output paths (overridable via environment)
LOAD=${LOAD:-"/mnt/cluster/loongforge-ckpt/deepseek_v4/mcore_deepseek_v4_flash_base/release"}
SAVE=${SAVE:-"/mnt/cluster/loongforge-ckpt/deepseek_v4/rebuilt_hf"}

MODEL_CONFIG_FILE=$LOONGFORGE_PATH/configs/models/deepseek4/deepseek_v4_flash_base.yaml
CONVERT_FILE=$LOONGFORGE_PATH/configs/models/deepseek4/ckpt_convert/deepseek_v4_convert.yaml

echo "=== Converting mcore -> HF ==="
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=2 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --max_workers=4 \
    --moe-grouped-gemm

echo "=== mcore -> HF conversion completed ==="
echo "Output: $SAVE"
