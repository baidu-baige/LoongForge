#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-training-llm/llama2/megatron_llama2_70b_tp2_pp8/iter_0000300/
SAVE=/mnt/cluster/aiak-training-llm/llama2/mcore_llama2_70b_tp4_pp4_megatron_iter300_convert

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=megatron \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llama2-70b.json \
    --tensor_model_parallel_size=4 \
    --pipeline_model_parallel_size=4 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_save_optim \
    --no_load_optim
