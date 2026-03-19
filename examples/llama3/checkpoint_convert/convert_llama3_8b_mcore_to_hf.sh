#! /bin/bash

export BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/BaigeOmni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
CONVERT_CHECKPOINT_PATH="$BAIGE_OMNI_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/baige-omni-ckpt/llama3/mcore_llama3_8b_tp2_pp2_omni/release
SAVE=/mnt/cluster/baige-omni-ckpt/huggingface.co/meta-llama/Meta-Llama-3-8B

MODEL_CONFIG_FILE=${BAIGE_OMNI_PATH}/configs/models/llama3/llama3_1_8b.yaml
CONVERT_FILE=${BAIGE_OMNI_PATH}/configs/models/llama3/ckpt_convert/llama3_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=2 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --no_save_optim \
    --no_load_optim
