#! /bin/bash

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
AIAK_MAGATRON_PATH=${AIAK_MAGATRON_PATH:-"/workspace/AIAK-Magatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B/
SAVE=/mnt/cluster/aiak-training-llm/cogvlm2/mcore_cogvlm2_llama3_chinese_chat_19B_tp4_pp1

SAVE_LANGUAGE_EXPERT=/mnt/cluster/aiak-training-llm/cogvlm2/tmp/language-expert-mcore
SAVE_VISION_EXPERT=/mnt/cluster/aiak-training-llm/cogvlm2/tmp/vision-expert-mcore
SAVE_VISION_MODEL=/mnt/cluster/aiak-training-llm/cogvlm2/tmp/vision-model-mcore
SAVE_ADAPTER=/mnt/cluster/aiak-training-llm/cogvlm2/tmp/adapter-mcore
SAVE_PATCH=/mnt/cluster/aiak-training-llm/cogvlm2/tmp/patch-mcore

TP=4

# llama: language expert
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/cogvlm2-19b/language-expert.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_EXPERT \
    --safetensors \
    --no-te \
    --no_save_optim \
    --no_load_optim

# llama: vision expert
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/cogvlm2-19b/vision-expert.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_EXPERT \
    --safetensors \
    --no-te \
    --no_save_optim \
    --no_load_optim

# vit
python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/cogvlm2-19b/vision-model.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no-te \
    --no_save_optim \
    --no_load_optim

# adapter
python $CONVERT_CHECKPOINT_PATH/custom/cogvlm/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/cogvlm2-19b/adapter.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# vision patch in vit
python $CONVERT_CHECKPOINT_PATH/custom/cogvlm/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/cogvlm2-19b/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

# merge
python $CONVERT_CHECKPOINT_PATH/custom/cogvlm/merge_megatron.py \
    --megatron_path $AIAK_MAGATRON_PATH \
    --language_expert_path $SAVE_LANGUAGE_EXPERT/release \
    --vision_expert_path $SAVE_VISION_EXPERT/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_EXPERT
rm -rf $SAVE_VISION_EXPERT
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
