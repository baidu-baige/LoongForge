#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/Qwen3-235B-A22B/
SAVE=/mnt/cluster/aiak-training-llm/qwen3/Qwen3_235B_A22B_mcore_tp4pp8ep16etp1

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen3-235b-a22b.json \
    --tensor_model_parallel_size=4 \
    --pipeline_model_parallel_size=8 \
    --num_experts=128 \
    --expert_parallel_size=16 \
    --expert_tensor_parallel_size=1 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --custom_pipeline_layers 11,11,12,12,12,12,12,12 \
    --safetensors \
    --resume-convert \
    --max_workers=32 \
    --moe-grouped-gemm