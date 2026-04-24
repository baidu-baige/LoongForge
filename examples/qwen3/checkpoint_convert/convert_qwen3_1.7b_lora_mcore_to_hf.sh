#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/LoongForge/qwen3/qwen3-1.7b-tp1-pp1-Dec24/release/
SAVE=/mnt/cluster/LoongForge/qwen3/qwen3-1.7b-hf-Dec24
LOAD_LORA=/mnt/cluster/LoongForge/qwen3/qwen3-1.7b-tp1-pp1-Dec24/iter_0000010/

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/qwen3_1_7b.yaml

CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_convert.yaml

TP=1
PP=1

LORA_ALPHA=32
LORA_DIM=16


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --load_lora_ckpt_path=$LOAD_LORA \
    --lora_alpha=$LORA_ALPHA \
    --lora_dim=$LORA_DIM \
    --safetensors \
    --no_save_optim \
    --no_load_optim