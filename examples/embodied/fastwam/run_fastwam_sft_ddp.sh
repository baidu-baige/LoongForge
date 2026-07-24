#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOONGFORGE_PATH="${LOONGFORGE_PATH:-/workspace/LoongForge}"


DATASET_PATH=${DATASET_PATH:-/path/to/libero}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/fastwam_sft_1gpu_$(date +%Y%m%d_%H%M%S)"}


export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH:-}
TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/path/to/tokenizer"}


MASTER_PORT=${MASTER_PORT:-29519}

TRAINING_ARGS=(
  --model-name fastwam
  --trainer-type FinetuneTrainer
  --distributed-strategy ddp
  --dtype bfloat16
  --train-iters 20000
  --save-interval 2000
  --clip-grad 1.0
  --gradient-accumulation-steps 1
  --log-interval 10
  --seed 3047
  --output-dir "$OUTPUT_DIR"
  --tokenizer-path "$TOKENIZER_PATH"
)

if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
  TRAINING_ARGS+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
fi

LR_ARGS=(
  --lr-base 1.0e-8
  --lr-decay-style cosine_warmup_with_min_lr
  --lr-warmup-iters 0
  --min-lr 1.0e-9
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.95
)

DATA_ARGS=(
  --dataset-format lerobot_datasets
  --dataset-strategy fastwam
  --dataset-path "$DATASET_PATH"
  --robot-type libero_franka
  --per-device-batch-size 1
  --num-workers 16
  --lerobotdataset-version v2.1
  --video-backend pyav
)

LOGGING_ARGS=(
  --wandb-project loongforge-vla
  --wandb-mode disabled
)


mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  LoongForge FastWAM SFT"
echo "  Model:    $MODEL_NAME"
echo "  Data:     $DATASET_PATH"
echo "  Output:   $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}" \
  torchrun --nproc_per_node 1 --master_port "$MASTER_PORT" \
  "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
  "${TRAINING_ARGS[@]}" \
  "${LR_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "$@" 2>&1 | tee "$OUTPUT_DIR/$(basename "${BASH_SOURCE[0]}" .sh)_$(date +%Y%m%d_%H%M%S).log"
