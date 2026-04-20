#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/meta-llama/Meta-Llama-3.1-405B
SAVE=/mnt/cluster/loongforge-ckpt/llama3.1/mcore_llama3.1_405b_tp8_pp16_omni

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/llama3/llama3_1_70b.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/llama3/ckpt_convert/llama3_1_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=16 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --no_save_optim \
    --no_load_optim
