#! /bin/bash

export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-omni-ckpt/deepseek2/DeepSeek_V2_group_tp1pp16ep8/release
SAVE=/mnt/cluster/aiak-omni-ckpt/huggingface.co/deepseek-ai/DeepSeek-V2

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek2/deepseek_v2.yaml
CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek2/ckpt_convert/deepseek_v2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --checkpoint-format=model-{i:05d}-of-{num_checkpoints:06d}.safetensors \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=16 \
    --expert_parallel_size=8 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --no-te \
    --moe-grouped-gemm \
    --safetensors
