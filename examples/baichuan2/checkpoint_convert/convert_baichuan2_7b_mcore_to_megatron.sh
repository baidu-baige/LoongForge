#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-training-llm/baichuan2/baichuan2-7b-tp1-pp1/release/
SAVE=/mnt/cluster/aiak-training-llm/baichuan2/baichuan2-7b-tp1-pp1_megatron/

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=megatron \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/baichuan2-7b.json \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --data_parallel_size=8 \
    --use_distributed_optimizer \
    --no_load_optim \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE