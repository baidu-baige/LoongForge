#! /bin/bash

export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen3-4B
SAVE=/mnt/cluster/aiak-training-llm/qwen3/qwen3-4b-tp1-pp1-Dec24

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/qwen3/qwen3_4b.yaml

CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/qwen3/ckpt_convert/qwen3_convert.yaml

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