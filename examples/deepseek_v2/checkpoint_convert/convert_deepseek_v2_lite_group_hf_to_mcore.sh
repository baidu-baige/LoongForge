#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V2-Lite
SAVE=/mnt/cluster/loongforge-ckpt/deepseek2/DeepSeek_V2_Lite_group_tp1pp1ep8/

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/deepseek2/deepseek_v2_lite.yaml

CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/deepseek2/ckpt_convert/deepseek_v2_lite_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --checkpoint-format=model-{i:05d}-of-{num_checkpoints:06d}.safetensors \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --expert_parallel_size=8 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --no-te \
    --moe-grouped-gemm \
    --safetensors
