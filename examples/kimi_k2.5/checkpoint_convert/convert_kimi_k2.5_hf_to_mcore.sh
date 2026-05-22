#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

LOAD=/mnt/cluster/huggingface.co/moonshotai/Kimi-K2.5/
SAVE=/mnt/cluster/LoongForge/moonshotai/Kimi-K2.5-entp1dtp8pp8ep16etp1

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/kimi_k2.5/kimi_k2_5.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/kimi_k2/ckpt_convert/kimi_k2_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/moon_vit_3d_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/patch_merger_adapter_convert.yaml

ETP=1
DTP=8
PP=8
EP=16
Expert_TP=1

EXTRA_ARGS=(
    --hf-dequantize-int4
    --hf-dequantize-dtype "${HF_DEQUANTIZE_DTYPE:-bfloat16}"
)

QUANT_CONFIG_FILE="${LOAD%/}/config.json"
if [[ -f "$QUANT_CONFIG_FILE" ]]; then
    EXTRA_ARGS+=(--hf-quant-config-file "$QUANT_CONFIG_FILE")
else
    echo "WARNING: compressed-tensors config not found: $QUANT_CONFIG_FILE; using fallback Kimi INT4 quantization args."
fi


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
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$Expert_TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --enable-full-hetero-dp \
    --fp8_force_no_requant \
    --moe-grouped-gemm \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --custom_pipeline_layers 8,8,8,8,8,8,8,5 \
    --force_pow_2_scales \
    --max_workers 64 \
    "${EXTRA_ARGS[@]}"
