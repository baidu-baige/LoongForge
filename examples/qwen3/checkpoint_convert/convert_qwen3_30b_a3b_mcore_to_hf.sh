#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-training-llm/qwen3/Qwen3_30B_A3B_mcore_tp2pp2ep4/release/
SAVE=/mnt/cluster/aiak-training-llm/qwen3/Qwen3_30B_A3B_mcore_tp2pp2ep4_hf

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen3-30b-a3b.json \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=2 \
    --num_experts=128 \
    --expert_parallel_size=4 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --safetensors
