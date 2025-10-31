#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-training-llm/deepseek2/DeepSeek_V2_Lite_group_tp1pp1ep8/release
SAVE=/mnt/cluster/aiak-training-llm/deepseek2/DeepSeek_V2_Lite_group_HF

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/deepseek-v2-lite.json \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --num_experts=64 \
    --expert_parallel_size=8 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --no-te \
    --moe-grouped-gemm \
    --safetensors
