#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Convert ERNIE-5.5-Flash-Preview weights with the common checkpoint converter.
# DSpark draft head (model.mtp.0/1/2) is not supported; the Preview config sets mtp_num_layers=0, so the export has no draft-head tensors.
# This script exports weights only and writes no config.json; the shard count follows tensor sizes, so the output can be several model-XXXXX-of-YYYYY.safetensors plus a model.safetensors.index.json and is not guaranteed to be a single model.safetensors.
set -e

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/mcore_checkpoint_convert"
MODEL_CONFIG_FILE="$LOONGFORGE_PATH/configs/models/ernie5_5/ernie5_5_flash_preview.yaml"
CONVERT_FILE="$LOONGFORGE_PATH/configs/models/ernie5_5/ckpt_convert/ernie5_5_convert.yaml"

# This converter joins --load_ckpt_path directly with the mp_rank_* subdirectories, so LOAD must be
# the directory that directly contains them -- `release/` for a Release checkpoint. The save side uses
# the same layout: once the iteration is greater than 0 it writes `iter_XXXXXXX/`, which also directly
# contains the mp_rank_* subdirectories. Megatron's training --load instead wants the parent that holds
# latest_checkpointed_iteration.txt; the two conventions must not be mixed.
LOAD=${LOAD:-"/mnt/cluster/LoongForge/ernie5.5/preview_mcore/release"}
SAVE=${SAVE:-"/mnt/cluster/LoongForge/ernie5.5/preview_rebuilt_hf"}

PYTHONPATH="$MEGATRON_PATH:$PYTHONPATH" \
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
