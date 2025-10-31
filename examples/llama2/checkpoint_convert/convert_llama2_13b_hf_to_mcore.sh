#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/meta-llama/Llama-2-13b-hf/
SAVE=/mnt/cluster/aiak-training-llm/llama2/mcore_llama2_13b_tp1_pp2

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/llama2-13b.json \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=2 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_save_optim \
    --no_load_optim
