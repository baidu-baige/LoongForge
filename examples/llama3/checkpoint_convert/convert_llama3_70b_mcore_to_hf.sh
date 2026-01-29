#! /bin/bash

export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-omni-ckpt/llama3/mcore_llama3_70b_tp4_pp4_omni/release
SAVE=/mnt/cluster/aiak-omni-ckpt/huggingface.co/meta-llama/Meta-Llama-3-70B

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/llama3/llama3_1_70b.yaml
CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/llama3/ckpt_convert/llama3_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=4 \
    --pipeline_model_parallel_size=4 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --no_save_optim \
    --no_load_optim
