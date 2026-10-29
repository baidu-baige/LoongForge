#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

if [ $# -eq 0 ]; then
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\""
    exit 1
fi
input_string=$1

export MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/wan2.1/Loong-Megatron/"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/wan2.1/LoongForge"}
export PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH

HF_CHECKPOINT_PATH=${HF_CHECKPOINT_PATH:-"/workspace/Wan-AI/Wan2.1-I2V-14B-480P"}
MCORE_CHECKPOINT_PATH=${MCORE_CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/wan2.1/hg2mcore/i2v_14b_480p/Megatron_Release/"}
HG_SAVE_PATH=${HG_SAVE_PATH:-"/mnt/cluster/LoongForge/wan2.1/hg/i2v_14b_480p_dcp/"}

if [ "$input_string" == "hg2mcore" ]; then
    echo "convert Wan2.1 I2V weight from huggingface to megatron"
    python ./convert_checkpoint_hg2mcore.py \
        --save_path="$MCORE_CHECKPOINT_PATH" \
        --checkpoint_path="$HF_CHECKPOINT_PATH" \
        --num_checkpoints=7 \
        --num_layers=40 \
        --model_name="wan2_1_i2v"
elif [ "$input_string" == "mcore2hg" ]; then
    echo "convert Wan2.1 I2V weight from megatron to huggingface"
    python ./convert_checkpoint_mcore2hg.py \
        --load_path="$MCORE_CHECKPOINT_PATH" \
        --save_path="$HG_SAVE_PATH" \
        --num_layers=40 \
        --model_name="wan2_1_i2v"
else
    echo "Usage: $0 input \"hg2mcore\" or \"mcore2hg\""
    exit 1
fi
