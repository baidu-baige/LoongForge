#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/
SAVE=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-tp1-pp1-Dec15

SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-mcore
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-mcore
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-mcore
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-mcore

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5vl/qwen2_5_vl_3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen2_5_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_mlp_adapter_convert.yaml

ETP=1
DTP=1
PP=1


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --adapter_convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --vision_patch_convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --encoder_tensor_model_parallel_size=$ETP \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --safetensors \
    --no_save_optim \
    --no_load_optim