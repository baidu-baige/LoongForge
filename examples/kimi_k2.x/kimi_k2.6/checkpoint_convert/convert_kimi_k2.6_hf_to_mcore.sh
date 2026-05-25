#! /bin/bash

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"
TORCHRUN=${TORCHRUN:-torchrun}

LOAD=${LOAD:-"/mnt/cluster/huggingface.co/moonshotai/Kimi-K2.6"}
SAVE=${SAVE:-"/mnt/cluster/LoongForge/moonshotai/Kimi-K2.6-entp1dtp8pp8ep16etp1"}

MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-"${LOONGFORGE_PATH}/configs/models/kimi_k2.6/kimi_k2_6.yaml"}

FOUNDATION_CONVERT_FILE=${FOUNDATION_CONVERT_FILE:-"${LOONGFORGE_PATH}/configs/models/kimi_k2/ckpt_convert/kimi_k2_convert.yaml"}
IMAGE_ENCODER_CONVERT_FILE=${IMAGE_ENCODER_CONVERT_FILE:-"${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/moon_vit_3d_convert.yaml"}
IMAGE_PROJECTOR_CONVERT_FILE=${IMAGE_PROJECTOR_CONVERT_FILE:-"${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/patch_merger_adapter_convert.yaml"}

ENCODER_TP=${ENCODER_TP:-1}
DTP=${DTP:-8}
PP=${PP:-8}
EP=${EP:-16}
EXPERT_TP=${EXPERT_TP:-1}
VPP=${VPP:-2}
CUSTOM_PIPELINE_LAYERS=${CUSTOM_PIPELINE_LAYERS:-"4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,2"}

# This script always uses torchrun with --distributed_convert. Defaults are for
# one node; set NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT in each pod for multinode.
MAX_WORKERS=${MAX_WORKERS:-4}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

mkdir -p "$SAVE/logs"
LOG=${LOG:-"$SAVE/logs/convert_node${NODE_RANK}_$(date +%Y%m%d-%H%M%S).log"}
exec &> >(tee "$LOG")

EXTRA_ARGS=(
    --hf-dequantize-int4
    --hf-dequantize-dtype "${HF_DEQUANTIZE_DTYPE:-bfloat16}"
)

QUANT_CONFIG_FILE=${QUANT_CONFIG_FILE:-"${LOAD%/}/config.json"}
if [[ -f "$QUANT_CONFIG_FILE" ]]; then
    EXTRA_ARGS+=(--hf-quant-config-file "$QUANT_CONFIG_FILE")
else
    echo "WARNING: compressed-tensors config not found: $QUANT_CONFIG_FILE; using fallback Kimi INT4 quantization args."
fi

echo "===== Kimi K2.6 HuggingFace -> M-Core conversion ($(date '+%F %T')) ====="
echo ">> LOG:                $LOG"
echo ">> LOAD:               $LOAD"
echo ">> SAVE:               $SAVE"
echo ">> QUANT_CONFIG_FILE:  ${QUANT_CONFIG_FILE:-<none>}"
echo ">> MODEL_CONFIG:       $MODEL_CONFIG_FILE"
echo ""

if [[ ! -d "$LOAD" ]]; then
    echo "Missing HuggingFace checkpoint directory: $LOAD" >&2
    exit 1
fi

if [[ ! -f "${LOAD%/}/config.json" ]]; then
    echo "Missing config.json under HuggingFace checkpoint directory: $LOAD" >&2
    exit 1
fi

echo ">> TORCHRUN:           $TORCHRUN"
echo ">> NNODES:             $NNODES"
echo ">> NPROC_PER_NODE:     $NPROC_PER_NODE"
echo ">> NODE_RANK:          $NODE_RANK"
echo ">> MASTER_ADDR:        $MASTER_ADDR"
echo ">> MASTER_PORT:        $MASTER_PORT"
echo ">> MAX_WORKERS/rank:   $MAX_WORKERS"
echo ""

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    "$TORCHRUN" \
    --nnodes "$NNODES" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$CONVERT_CHECKPOINT_PATH/module_convertor/model.py" \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file "$MODEL_CONFIG_FILE" \
    --convert_file "$FOUNDATION_CONVERT_FILE" \
    --adapter_convert_file "$IMAGE_PROJECTOR_CONVERT_FILE" \
    --vision_patch_convert_file "$IMAGE_ENCODER_CONVERT_FILE" \
    --encoder_tensor_model_parallel_size="$ENCODER_TP" \
    --tensor_model_parallel_size="$DTP" \
    --pipeline_model_parallel_size="$PP" \
    --expert_parallel_size="$EP" \
    --expert_tensor_parallel_size="$EXPERT_TP" \
    --num-virtual-stages-per-pipeline-rank="$VPP" \
    --custom_pipeline_layers "$CUSTOM_PIPELINE_LAYERS" \
    --load_ckpt_path="$LOAD" \
    --save_ckpt_path="$SAVE" \
    --enable-full-hetero-dp \
    --fp8_force_no_requant \
    --moe-grouped-gemm \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --force_pow_2_scales \
    --distributed_convert \
    --max_workers "$MAX_WORKERS" \
    "${EXTRA_ARGS[@]}"

echo "Saved M-Core checkpoint to: $SAVE"
