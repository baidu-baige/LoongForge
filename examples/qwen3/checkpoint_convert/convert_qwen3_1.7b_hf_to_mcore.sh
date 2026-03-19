#! /bin/bash

export BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/BaigeOmni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
CONVERT_CHECKPOINT_PATH="$BAIGE_OMNI_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/models/Qwen3-1.7B
SAVE=/mnt/cluster/BaigeOmni/qwen3/qwen3-1.7b-tp1-pp1-Dec24

MODEL_CONFIG_FILE=${BAIGE_OMNI_PATH}/configs/models/qwen3/qwen3_1_7b.yaml

CONVERT_FILE=${BAIGE_OMNI_PATH}/configs/models/qwen3/ckpt_convert/qwen3_convert.yaml

TP=1
PP=1


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --no_save_optim \
    --no_load_optim