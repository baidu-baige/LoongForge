#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/meta-llama/Llama-2-70b-hf/
SAVE=/mnt/cluster/loongforge-ckpt/llama2/mcore_llama2_70b_tp4_pp4_omni

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/llama2/llama2_70b.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/llama2/ckpt_convert/llama2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=4 \
    --pipeline_model_parallel_size=4 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_save_optim \
    --no_load_optim
