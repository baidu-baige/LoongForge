#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct
SAVE=/mnt/cluster/aiak-training-llm/qwen3//Qwen3_480B_A35B_mcore_tp8pp8ep16etp1

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/qwen3-480b-a35b.json \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --num_experts=160 \
    --expert_parallel_size=16 \
    --expert_tensor_parallel_size=1 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --custom_pipeline_layers 7,7,8,8,8,8,8,8 \
    --safetensors \
    --resume-convert \
    --max_workers=32 \
    --moe-grouped-gemm
