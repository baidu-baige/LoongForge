#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V3-bf16  # the BF16 format checkpoint, converted via fp8_to_bf16 casting script
SAVE=/mnt/cluster/aiak-training-llm/deepseek3/DeepSeek-V3-bf16-tp8pp8ep32etp1/  # the converted checkpoint will be in MCore BF16 format
CACHE=/mnt/cluster/aiak-training-llm/deepseek3/DeepSeek-V3-cache

python $CONVERT_CHECKPOINT_PATH/model.py \
    --checkpoint-format=model-{i:05d}-of-{num_checkpoints:06d}.safetensors \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/deepseek-v3.json \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --num_experts=256 \
    --expert_parallel_size=32 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --cache-path=$CACHE \
    --no_load_optim \
    --no_save_optim \
    --no-te \
    --custom_pipeline_layers 8,7,8,8,8,8,8,6 \
    --safetensors \
    --resume-convert \
    --max_workers=32 \
    --moe-grouped-gemm \
    --amax_epsilon=1e-4