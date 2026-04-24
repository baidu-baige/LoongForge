#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/GLM/GLM-5-FP8  # the original GLM-5 checkpoint is FP8 format
SAVE=/mnt/cluster/loongforge-omni-ckpt/GLM/GLM-5-FP8-tp8pp8ep8etp1/  # the converted checkpoint will be in MCore FP8 format

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/glm5/glm5.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/glm5/ckpt_convert/glm5_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --expert_parallel_size=8 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --custom_pipeline_layers 10,10,10,10,10,10,10,8 \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    --fp8_force_no_requant
