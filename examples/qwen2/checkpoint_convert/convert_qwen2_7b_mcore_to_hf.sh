#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-omni-ckpt/qwen2/Qwen2_7B_mcore_tp1pp1_omni/release
SAVE=/mnt/cluster/aiak-omni-ckpt/huggingface.co/Qwen/Qwen2-7B/

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/qwen2/qwen2_7b.yaml
CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/qwen2/ckpt_convert/qwen2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --safetensors