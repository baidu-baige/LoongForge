#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/loongforge-ckpt/llama2/mcore_llama2_7b_tp1_pp1_omni/release
SAVE=/mnt/cluster/loongforge-ckpt/huggingface.co/meta-llama/Llama-2-7b-hf/

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/llama2/llama2_7b.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/llama2/ckpt_convert/llama2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_save_optim \
    --no_load_optim
