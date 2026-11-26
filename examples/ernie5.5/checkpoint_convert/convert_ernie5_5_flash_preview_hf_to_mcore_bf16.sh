#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Convert BF16 or FP8 ERNIE-5.5-Flash-Preview weights with the common checkpoint converter.
# DSpark draft head (model.mtp.0/1/2) is not converted; the Preview config sets mtp_num_layers=0, so the export
# has no draft-head tensors. Those tensors are not filtered by shard: if they share a safetensors shard with
# trunk weights the loader reads the whole shard, then drops `model.mtp.*` before MXFP4 dequantization.
set -e

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/mcore_checkpoint_convert"
MODEL_CONFIG_FILE="$LOONGFORGE_PATH/configs/models/ernie5_5/ernie5_5_flash_preview.yaml"
CONVERT_FILE="$LOONGFORGE_PATH/configs/models/ernie5_5/ckpt_convert/ernie5_5_convert.yaml"

HF_MODEL_PATH=${HF_MODEL_PATH:-"/mnt/cluster/huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16"}
SAVE_ROOT=${SAVE_ROOT:-"/mnt/cluster/LoongForge/ernie5.5/preview_bf16_mcore"}

PYTHONPATH="$MEGATRON_PATH:$PYTHONPATH" \
    python "$CONVERT_CHECKPOINT_PATH/module_convertor/model.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file "$MODEL_CONFIG_FILE" \
    --convert_file "$CONVERT_FILE" \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size="${PP_SIZE:-1}" \
    --expert_parallel_size="${EP_SIZE:-8}" \
    --expert_tensor_parallel_size=1 \
    --megatron_path="$MEGATRON_PATH" \
    --load_ckpt_path="$HF_MODEL_PATH" \
    --save_ckpt_path="$SAVE_ROOT" \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    "$@"
