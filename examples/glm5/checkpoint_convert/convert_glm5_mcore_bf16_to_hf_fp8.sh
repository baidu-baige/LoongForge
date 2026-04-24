#! /bin/bash


export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/loongforge-ckpt/GLM/GLM-5-bf16-tp8pp8ep8etp1/release  # the converted checkpoint will be in MCore BF16 format
SAVE=/mnt/cluster/huggingface.co/GLM/GLM-5-fp8-hf  # the FP8 format checkpoint, converted via fp8_to_bf16 casting script

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/glm5/glm5.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/glm5/ckpt_convert/glm5_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
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
    --amax_epsilon=1e-12 \
    --pretrain_as_fp8 \
    --convert_to_fp8 \
    --force_pow_2_scales
