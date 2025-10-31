#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/baichuan-inc/Baichuan2-7B-Base/
SAVE=/mnt/cluster/aiak-training-llm/baichuan2/baichuan2-7b-tp1-pp1/

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/baichuan2-7b.json \
    --load_ckpt_path=$LOAD \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --save_ckpt_path=$SAVE \
    --no_save_optim
