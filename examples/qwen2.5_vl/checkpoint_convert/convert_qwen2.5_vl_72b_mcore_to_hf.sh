#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

SAVE=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-72b-hf-Dec23
LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-72b-tp8-pp4-custom-Original/release
OMNI_LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-72b-tp8-pp4-custom-Dec19/release

SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-hf
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-hf
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-hf
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-hf

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5vl/qwen2_5_vl_72b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen2_5_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_mlp_adapter_convert.yaml


PP=4
ETP=8
DTP=8
CUSTOM_PIPELINE_LAYERS=20,20,20,20



PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py \
  --load_omni_ckpt_path $OMNI_LOAD \
  --save_original_ckpt_path $LOAD \
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
    --custom_pipeline_layers=$CUSTOM_PIPELINE_LAYERS \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim


# vit
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$ETP;i++)); do
        from=`printf "mp_rank_%02d_000" $i`
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
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
    --safetensors \
    --no_save_optim \
    --no_load_optim \


if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
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