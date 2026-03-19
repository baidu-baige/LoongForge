#! /bin/bash

export BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/BaigeOmni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
CONVERT_CHECKPOINT_PATH="$BAIGE_OMNI_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/baige-omni-ckpt/qwen/Qwen_1_8B_mcore_tp1pp1_omni/release
SAVE=/mnt/cluster/baige-omni-ckpt/huggingface.co/Qwen/Qwen-1.8B/

MODEL_CONFIG_FILE=${BAIGE_OMNI_PATH}/configs/models/qwen/qwen_1_8b.yaml
CONVERT_FILE=${BAIGE_OMNI_PATH}/configs/models/qwen/ckpt_convert/qwen_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --safetensors
