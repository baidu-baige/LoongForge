#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

SAVE=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-hf-Dec22
LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-tp1-pp1-Original/release
OMNI_LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-tp1-pp1-Dec15/release

LOAD_LORA=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-tp1-pp1-Original/iter_0000010
OMNI_LOAD_LORA=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-tp1-pp1-Dec15/iter_0000010

SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-hf
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-hf
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-hf
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-hf

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5vl/qwen2_5_vl_3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen2_5_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_mlp_adapter_convert.yaml


PP=1
ETP=1
DTP=1

LORA_ALPHA=32
LORA_DIM=16

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py \
  --load_omni_ckpt_path $OMNI_LOAD \
  --save_original_ckpt_path $LOAD \
  --decoder_tensor_model_parallel_size=$DTP \
  --pipeline_model_parallel_size=$PP \
  --config_file $MODEL_CONFIG_FILE

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py \
  --load_omni_ckpt_path $OMNI_LOAD_LORA \
  --save_original_ckpt_path $LOAD_LORA \
  --decoder_tensor_model_parallel_size=$DTP \
  --pipeline_model_parallel_size=$PP \
  --config_file $MODEL_CONFIG_FILE


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --load_lora_ckpt_path=$LOAD_LORA \
    --lora_alpha=$LORA_ALPHA \
    --lora_dim=$LORA_DIM \
    --safetensors \
    --no_save_optim \
    --no_load_optim


# vit
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
    LOAD_LORA_PATH=$LOAD_LORA
else
    LOAD_PATH=$LOAD/tmp/
    LOAD_LORA_PATH=$LOAD_LORA/tmp/
    mkdir -p $LOAD_PATH
    mkdir -p $LOAD_LORA_PATH
    for ((i=0;i<$ETP;i++)); do
        from=`printf "mp_rank_%02d_000" $i`
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
        cp -r $LOAD_LORA/$from $LOAD_LORA_PATH/$to
    done
fi

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --load_lora_ckpt_path=$LOAD_LORA_PATH \
    --lora_alpha=$LORA_ALPHA \
    --lora_dim=$LORA_DIM \
    --safetensors \
    --no_save_optim \
    --no_load_optim \


if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

if [[ $LOAD_LORA != $LOAD_LORA_PATH ]]; then
    rm -rf $LOAD_LORA_PATH
fi

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size $PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size $PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# merge
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/huggingface/merge_huggingface.py \
    --megatron_path $MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL\
    --vision_model_path $SAVE_VISION_MODEL\
    --vision_patch $SAVE_PATCH\
    --adapter_path $SAVE_ADAPTER\
    --save_ckpt_path $SAVE\


rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH