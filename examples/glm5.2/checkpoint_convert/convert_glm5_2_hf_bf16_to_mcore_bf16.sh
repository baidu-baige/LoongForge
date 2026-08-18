#! /bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/GLM/GLM-5.2  # the GLM-5.2 BF16 checkpoint
SAVE=/mnt/cluster/loongforge-omni-ckpt/GLM/GLM-5.2-bf16-tp8pp8ep8etp1/  # the converted checkpoint will be in MCore BF16 format

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/glm5.2/glm5_2.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/glm5.2/ckpt_convert/glm5_2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --expert_parallel_size=8 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --pipeline_model_parallel_layout "Et*6|t*4|t*4|t*4|t*8|t*4|t*4|t*4|t*4|t*4|t*8|t*4|t*4|t*4|t*8|t*4mL" \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm
