#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Convert ERNIE-5.5-Flash-Base weights with the common checkpoint converter.
set -e

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/mcore_checkpoint_convert"
MODEL_CONFIG_FILE=$LOONGFORGE_PATH/configs/models/ernie5_5/ernie5_5_flash_base.yaml
CONVERT_FILE=$LOONGFORGE_PATH/configs/models/ernie5_5/ckpt_convert/ernie5_5_convert.yaml

LOAD=${LOAD:-"/mnt/cluster/LoongForge/ernie5.5/mcore/release"}
SAVE=${SAVE:-"/mnt/cluster/LoongForge/ernie5.5/rebuilt_hf"}

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python "$CONVERT_CHECKPOINT_PATH/module_convertor/model.py" \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file "$MODEL_CONFIG_FILE" \
    --convert_file "$CONVERT_FILE" \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size="${PP_SIZE:-1}" \
    --expert_parallel_size="${EP_SIZE:-8}" \
    --expert_tensor_parallel_size=1 \
    --megatron_path="$MEGATRON_PATH" \
    --load_ckpt_path="$LOAD" \
    --save_ckpt_path="$SAVE" \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    "$@"
