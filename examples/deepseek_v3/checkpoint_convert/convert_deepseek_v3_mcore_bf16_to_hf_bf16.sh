#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/aiak-omni-ckpt/deepseek3/DeepSeek-V3-bf16-tp8pp8ep32etp1  # source checkpoint in MCore BF16 format from training
SAVE=/mnt/cluster/aiak-omni-ckpt/deepseek3/DeepSeek-V3-tp8pp8ep32etp1-hf  # converted checkpoint in HF BF16 format for inferencing

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek3/deepseek_v3.yaml
CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek3/ckpt_convert/deepseek_v3_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --num_experts=256 \
    --expert_parallel_size=32 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --no_load_optim \
    --no_save_optim \
    --no-te \
    --custom_pipeline_layers 8,7,8,8,8,8,8,6 \
    --safetensors \
    --resume-convert \
    --moe-grouped-gemm \
    --max_workers=32