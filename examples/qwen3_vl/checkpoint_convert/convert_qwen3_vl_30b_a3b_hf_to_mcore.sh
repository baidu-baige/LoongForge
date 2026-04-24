#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/
SAVE=/mnt/cluster/LoongForge/qwen3-vl-30b-a3b/qwen3-vl-30b-etp1etp1pp8ep1

SAVE_LANGUAGE_MODEL=${SAVE}/tmp/language-mcore
SAVE_VISION_MODEL=${SAVE}/tmp/vision-model-mcore
SAVE_ADAPTER=${SAVE}/tmp/adapter-mcore
SAVE_PATCH=${SAVE}/tmp/patch-mcore

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert_qwen3vl.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen3_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_3_mlp_adapter_convert.yaml

ETP=1
DTP=1
PP=8
EP=1
EXPERT_TP=1


PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$EXPERT_TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --moe-grouped-gemm

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# merge
if [ $EP -gt 1 ]; then
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron_expert.py\
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --expert_parallel_size $EP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE 
else
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py\
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE 
fi

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
