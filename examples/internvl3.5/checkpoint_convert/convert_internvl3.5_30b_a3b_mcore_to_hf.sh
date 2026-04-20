#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

SAVE=/mnt/cluster/LoongForge/internvl3.5/internvl3.5-30b-a3b-hf-Dec23
LOAD=/mnt/cluster/LoongForge/internvl3.5/internvl3.5-30b-a3b-tp2-pp2-ep4-etp1-Original/release
OMNI_LOAD=/mnt/cluster/LoongForge/internvl3.5/internvl3.5-30b-a3b-tp2-pp2-ep4-etp1-Dec15/release

SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-hf
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-hf
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-hf
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-hf

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/internvl3.5/internvl3_5_30b_a3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert_intern.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/internvl_vit_0.3b_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/intern_mlp_adapter_convert.yaml


ETP=2
DTP=2
PP=2
EP=4
Expert_TP=1



PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser_expert.py \
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
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$Expert_TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim


# vit: vision model
if [ -z "${EP:-}" ]; then
  EP=1
fi
if [ -z "${Expert_TP:-}" ]; then
  Expert_TP=1
fi
if [ $PP -eq 1 ] && [ $EP -eq 1 ]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$ETP;i++)); do
        from=`printf "mp_rank_%02d" $i`
        if [ $PP != 1 ]; then
          from+="_000"
        fi
        if [ $EP != 1 ]; then
          from+=`printf "_%03d" $((i/Expert_TP))`
        fi
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
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \



PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
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
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
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

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH