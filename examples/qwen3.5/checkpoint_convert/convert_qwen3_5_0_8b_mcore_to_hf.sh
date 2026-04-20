#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

SAVE=/workspace/Qwen3.5-0.8B-HF
LOAD=/workspace/qwen3.5-0.8b-tp1pp1_tmp/release
OMNI_LOAD=/workspace/qwen3.5-0.8b-tp1pp1/release

SAVE_LANGUAGE_MODEL=${SAVE}/tmp/language-hf
SAVE_VISION_MODEL=${SAVE}/tmp/vision-model-hf
SAVE_ADAPTER=${SAVE}/tmp/adapter-hf
SAVE_PATCH=${SAVE}/tmp/patch-hf

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3.5/qwen3_5_0_8b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3.5/ckpt_convert/qwen3_5_dense_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen3_5_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_3_mlp_adapter_convert.yaml

ETP=1
DTP=1
PP=1

# Step 1: Reverse omni keys back to vanilla mcore keys (dense: use key_reverser.py)
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py \
  --load_omni_ckpt_path $OMNI_LOAD \
  --save_original_ckpt_path $LOAD \
  --decoder_tensor_model_parallel_size=$DTP \
  --pipeline_model_parallel_size=$PP \
  --config_file $MODEL_CONFIG_FILE

# Step 2: Language model mcore -> hf
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --mtp_num_layers 1

# Step 3: Vision model mcore -> hf
# When PP>1, ViT resides in the first pipeline stage (pp_rank=000).
# Extract it to a flat directory so the converter can load it with PP=1.
if [ $PP -eq 1 ]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$ETP;i++)); do
        from=`printf "mp_rank_%02d" $i`
        if [ $PP != 1 ]; then
          from+="_000"
        fi
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# Step 4: Adapter mcore -> hf
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# Step 5: Vision patch mcore -> hf
PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# Step 6: Merge all components into final HF checkpoint
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/huggingface/merge_huggingface.py \
    --megatron_path $MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE

# Cleanup
if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
