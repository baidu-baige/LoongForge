#! /bin/bash

export OMNI_PATH=${OMNI_PATH:-"/workspace/BaigeOmni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
CONVERT_CHECKPOINT_PATH="$OMNI_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/meta-llama/Llama-2-13b-hf/
SAVE=/mnt/cluster/baige-omni-ckpt/llama2/mcore_llama2_13b_tp1_pp2_omni

MODEL_CONFIG_FILE=${OMNI_PATH}/configs/models/llama2/llama2_13b.yaml
CONVERT_FILE=${OMNI_PATH}/configs/models/llama2/ckpt_convert/llama2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=2 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_save_optim \
    --no_load_optim
