export PYTHONPATH="/workspace/AIAK-Megatron/:$PYTHONPATH"

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
AIAK_MEGATRON_PATH=${AIAK_MEGATRON_PATH:-"/workspace/AIAK-Magatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/rapidfs/users/baige/zhaiyanfeng/models/internvl/InternVL3_5-38B
SAVE=/mnt/rapidfs/users/baige/zhaiyanfeng/out/ckpt_convert/Internvl3_5-38B-tp4-pp2

TMP=$SAVE/convert_cache
mkdir -p $TMP
SAVE_LANGUAGE_MODEL=$TMP/language-model-mcore
SAVE_ADAPTER=$TMP/adapter-mcore
SAVE_VISION_MODEL=$TMP/vision-model-mcore
SAVE_PATCH=$TMP/patch-mcore
TP=4
PP=2

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/internvl3.5-38b.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --custom_pipeline_layers 2,62 \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/vision-model-6b.json \
    --tensor_model_parallel_size=$TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# patch embeddings in vit
python $CONVERT_CHECKPOINT_PATH/custom/internvl/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

python $CONVERT_CHECKPOINT_PATH/custom/internvl/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --tensor_model_parallel_size=$TP \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/adapter.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# merge
python $CONVERT_CHECKPOINT_PATH/custom/internvl/merge_megatron.py \
    --megatron_path $AIAK_MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --save_ckpt_path $SAVE/release \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP

echo 'release' > $SAVE/latest_checkpointed_iteration.txt
rm -rf $TMP