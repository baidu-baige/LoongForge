#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/LoongForge/mini_max/MiniMax_mcore_tp8pp4ep8etp1/release
SAVE=/mnt/cluster/LoongForge/mini_max/MiniMax-M2.1-hf/

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/minimax/minimax_m2_1.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/minimax/ckpt_convert/minimax_convert.yaml

TP=8
PP=4
EP=8
ETP=1

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$ETP \
    --custom_pipeline_layers 16,16,16,14 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --num_experts=256 \
    --no_load_optim \
    --no_save_optim \
    --fp8_force_no_requant \
    --safetensors \
    --resume-convert \
    --max_workers=32 \
    --pretrain_as_fp8 \
    --moe-grouped-gemm