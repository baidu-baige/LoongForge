#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# Convert ERNIE-4.5-VL checkpoints from HuggingFace to Megatron-Core format.
#
# Usage:
#   bash convert.sh
#
# Pipeline-parallel (PP > 1) example:
#   Set PP and PP_LAYER_OFFSETS below, then run:
#   bash convert.sh
#
# PP_LAYER_OFFSETS: comma-separated global LM-layer start index for each PP stage.
#   - Length must equal PP.
#   - Leave empty ("") when PP=1 (the argument is omitted automatically).
#   - Example for PP=8 with split [4,4,4,4,4,4,2,2]:
#       PP_LAYER_OFFSETS="0,4,8,12,16,20,24,26"

MEGATRON_PATH=/workspace/ernie/Loong-Megatron/
LOONGFORGE_PATH=/workspace/ernie/LoongForge/

export PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH

# ---------- user-configurable parameters ----------
LOAD_HG_PATH="/workspace/ernie/ERNIE-4.5-VL-28B-A3B-PT/"
LOAD_MCORE_PATH="/workspace/ernie/ckpt/ERNIE-4.5-VL-28B-A3B-MCORE_save_new/iter_0000002/"
SAVE_MCORE_PATH="/workspace/ernie/ckpt/ERNIE-4.5-VL-28B-A3B-MCORE_hg2mcore_fixed/"

TP=1
PP=1
NUM_VIT_LAYERS=32
NUM_LM_LAYERS=28
NUM_EXPERTS=64

# For PP=1 leave this empty; for PP>1 set e.g. "0,4,8,12,16,20,24,26"
PP_LAYER_OFFSETS=""
# --------------------------------------------------

# Build optional --pp_layer_offsets flag
PP_OFFSETS_ARG=""
if [ -n "$PP_LAYER_OFFSETS" ]; then
    PP_OFFSETS_ARG="--pp_layer_offsets=$PP_LAYER_OFFSETS"
fi

python ./ernie4.5vl_hg2mcore.py \
    --load_hg_path="$LOAD_HG_PATH" \
    --load_mcore_path="$LOAD_MCORE_PATH" \
    --save_mcore_path="$SAVE_MCORE_PATH" \
    --tp=$TP \
    --pp=$PP \
    --num_vit_layers=$NUM_VIT_LAYERS \
    --num_lm_layers=$NUM_LM_LAYERS \
    --num_experts=$NUM_EXPERTS \
    $PP_OFFSETS_ARG
