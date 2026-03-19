#! /bin/bash

export BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/BaigeOmni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
CONVERT_CHECKPOINT_PATH="$BAIGE_OMNI_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen2-1.5B/
SAVE=/mnt/cluster/baige-omni-ckpt/qwen2/Qwen2_1.5B_mcore_tp1pp1_omni

MODEL_CONFIG_FILE=${BAIGE_OMNI_PATH}/configs/models/qwen2/qwen2_1_5b.yaml
CONVERT_FILE=${BAIGE_OMNI_PATH}/configs/models/qwen2/ckpt_convert/qwen2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
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