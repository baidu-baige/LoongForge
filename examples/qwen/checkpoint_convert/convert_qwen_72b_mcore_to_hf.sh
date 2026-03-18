#! /bin/bash

export OMNI_PATH=${OMNI_PATH:-"/workspace/BaigeOmni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
CONVERT_CHECKPOINT_PATH="$OMNI_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/baige-omni-ckpt/qwen/Qwen_72B_mcore_tp2pp8_omni/release
SAVE=/mnt/cluster/baige-omni-ckpt/huggingface.co/Qwen/Qwen-72B/

MODEL_CONFIG_FILE=${OMNI_PATH}/configs/models/qwen/qwen_72b.yaml
CONVERT_FILE=${OMNI_PATH}/configs/models/qwen/ckpt_convert/qwen_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=8 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --safetensors
