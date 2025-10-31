#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
SAVE=/mnt/cluster/aiak-training-llm/qwen3/Qwen3_Coder_30B_A3B_mcore_tp2pp2ep4

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen3-coder-30b-a3b.json \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=2 \
    --num_experts=128 \
    --expert_parallel_size=4 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --safetensors \
    --resume-convert \
    --max_workers=32 \
    --moe-grouped-gemm
