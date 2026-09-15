#! /bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/loongforge-omni-ckpt/GLM/GLM-5.3-Flash-bf16-tp8pp5ep8etp1/release  # the GLM-5.3-Flash MCore BF16 checkpoint
SAVE=/mnt/cluster/huggingface.co/GLM/GLM-5.3-Flash-bf16-hf  # the GLM-5.3-Flash BF16 HF checkpoint
# NOTE: converting back to HF yields a BF16 checkpoint; the released
# zai-org/GLM-5.3-Flash is FP8, so the round-tripped weights are the
# dequantized values, not the original FP8 tensors.

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/glm5_next/glm5.3_flash.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/glm5_next/ckpt_convert/glm5.3_flash_convert.yaml

# Uniform TP8 PP5 EP8 ETP1 layout (45 layers, no MTP): no custom
# --pipeline_model_parallel_layout is required.
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=5 \
    --expert_parallel_size=8 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm
