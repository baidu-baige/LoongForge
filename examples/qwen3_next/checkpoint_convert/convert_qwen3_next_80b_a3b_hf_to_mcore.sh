#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/models/ckpt/Qwen3-Next-80B-A3B-Instruct
SAVE=/models/ckpt/Qwen3-Next-80B-A3B-tp2pp2ep8etp1

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3_next/qwen3_next_80b_a3b.yaml

CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3_next/ckpt_convert/qwen3_next_moe_convert.yaml

TP=1
PP=4
EP=8
ETP=1


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --mtp_num_layers 1 \
    --safetensors \
    --moe-grouped-gemm