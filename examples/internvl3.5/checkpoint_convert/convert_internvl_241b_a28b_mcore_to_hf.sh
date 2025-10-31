export PYTHONPATH="/workspace/AIAK-Megatron/:$PYTHONPATH"

AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}
AIAK_MEGATRON_PATH=${AIAK_MEGATRON_PATH:-"/workspace/AIAK-Magatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/mnt/rapidfs/users/baige/zhaiyanfeng/out/ckpt_convert/InternVL3_5-241B-A28B-tp4pp4ep8etp1-save/iter_0000002/
SAVE=/mnt/rapidfs/users/baige/zhaiyanfeng/models/internvl/test-mcore-to-hf/internvl3.5-241b-a28b

TMP=$SAVE/convert_cache
mkdir -p $TMP
SAVE_LANGUAGE_MODEL=$TMP/language-model-hf
SAVE_VISION_MODEL=$TMP/vision-model-hf
SAVE_ADAPTER=$TMP/adapter-hf
SAVE_PATCH=$TMP/patch-hf

TP=4
PP=4
EP=8
ETP=1

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/internvl3.5-241b-a28b.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --num_experts=128 \
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --no_load_optim \
    --no_save_optim \
    --custom_pipeline_layers 11,28,28,27 \
    --safetensors \
    --resume-convert \
    --max_workers=32 \
    --moe-grouped-gemm

# vit: vision model
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$TP;i++)); do
        from=`printf "mp_rank_%02d_000" $i`
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

python $CONVERT_CHECKPOINT_PATH/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --megatron_path $AIAK_MEGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/vision-model-6b.json \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

# patch embeddings in vit
python $CONVERT_CHECKPOINT_PATH/custom/internvl/vision_patch.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --megatron_path $AIAK_MEGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/vision-patch.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH

python $CONVERT_CHECKPOINT_PATH/custom/internvl/adapter.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --megatron_path $AIAK_MEGATRON_PATH \
    --common_config_path=$CONVERT_CHECKPOINT_PATH/config/internvl3_5/adapter.json \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER

# merge
python $CONVERT_CHECKPOINT_PATH/custom/internvl/merge_huggingface.py \
    --megatron_path $AIAK_MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL \
    --vision_model_path $SAVE_VISION_MODEL \
    --vision_patch $SAVE_PATCH \
    --adapter_path $SAVE_ADAPTER \
    --save_ckpt_path $SAVE \


# clean
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH