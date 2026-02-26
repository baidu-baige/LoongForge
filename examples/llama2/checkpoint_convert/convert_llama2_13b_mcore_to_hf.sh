#! /bin/bash

export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/OmniTraining"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-omni-ckpt/llama2/mcore_llama2_13b_tp1_pp2_omni/release
SAVE=/mnt/cluster/aiak-omni-ckpt/huggingface.co/meta-llama/Llama-2-13b-hf/

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/llama2/llama2_13b.yaml
CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/llama2/ckpt_convert/llama2_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=1 \
    --pipeline_model_parallel_size=2 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_save_optim \
    --no_load_optim
