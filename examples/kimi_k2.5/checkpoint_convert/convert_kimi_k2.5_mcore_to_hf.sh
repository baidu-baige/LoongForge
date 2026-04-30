#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

SAVE=/mnt/cluster/huggingface.co/moonshotai/Kimi-K2.5-hf
LOAD=/mnt/cluster/LoongForge/moonshotai/Kimi-K2.5-entp8dtp8pp8ep32etp1/release

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/kimi_k2.5/kimi_k2_5.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/kimi_k2/ckpt_convert/kimi_k2_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/moon_vit_3d_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/patch_merger_adapter_convert.yaml

ETP=8
DTP=8
PP=8
EP=32
Expert_TP=1


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --adapter_convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --vision_patch_convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --encoder_tensor_model_parallel_size=$ETP \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$Expert_TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --enable-full-hetero-dp \
    --safetensors \
    --fp8_force_no_requant \
    --moe-grouped-gemm \
    --no_save_optim \
    --no_load_optim
