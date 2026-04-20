#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/LoongForge/qwen3/qwen3-coder-30b-a3b-tp2-pp2-ep4-Dec15/release/
SAVE=/mnt/cluster/LoongForge/qwen3/qwen3-coder-30b-a3b-hf-Dec24

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/qwen3_coder_30b_a3b.yaml

CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert.yaml

TP=2
PP=2
EP=4


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --moe-grouped-gemm