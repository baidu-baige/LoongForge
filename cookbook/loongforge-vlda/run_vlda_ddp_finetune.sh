#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# LoongForge-VLDA LIBERO two-stage baseline. Run each stage as a separate torchrun job.
# STAGE=1 CHECKPOINT_PATH=<pi05-base> VJEPA_CHECKPOINT=<vitg.pt> bash ...
# STAGE=2 CHECKPOINT_PATH=<stage1-checkpoint> bash ...
# RESUME=1 STAGE=2 OUTPUT_DIR=<existing-stage2-run> bash ...
# ZERO_OPTIMIZER=1 enables the existing LoongForge ZeRO1 optimizer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
STAGE=${STAGE:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MODEL_NAME=${MODEL_NAME:-vlda}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/vlda/stage${STAGE}"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-tensorboard}
RESUME=${RESUME:-0}
SEED=${SEED:-42}
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${DATA_PATH:?Set DATA_PATH to the LoongForge-VLDA LeRobot v2 dataset}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to paligemma_tokenizer.model}"
NORM_STATS_PATH=${NORM_STATS_PATH:-"$DATA_PATH/norm_stats.json"}
PREPROCESSING_BACKEND=${PREPROCESSING_BACKEND:-native}

case "$STAGE" in
    1)
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-48}
        TRAIN_ITERS=${TRAIN_ITERS:-20000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
        WARMUP_ITERS=${WARMUP_ITERS:-1000}
        PEAK_LR=${PEAK_LR:-5e-5}
        MIN_LR=${MIN_LR:-2e-6}
        ;;
    2)
        GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
        TRAIN_ITERS=${TRAIN_ITERS:-160000}
        SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
        WARMUP_ITERS=${WARMUP_ITERS:-10000}
        PEAK_LR=${PEAK_LR:-1e-4}
        MIN_LR=${MIN_LR:-1e-5}
        ;;
    *) echo "STAGE must be 1 or 2" >&2; exit 1 ;;
esac

GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
BATCH_DIVISOR=$((GPUS_PER_NODE * NNODES * GRADIENT_ACCUMULATION_STEPS))
if (( BATCH_DIVISOR <= 0 || GLOBAL_BATCH_SIZE <= 0 || GLOBAL_BATCH_SIZE % BATCH_DIVISOR != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by GPUs times gradient accumulation" >&2
    exit 1
fi

ARGS=(
    --model-name "$MODEL_NAME"
    --trainer-type VLDAFinetuneTrainer
    --dataset-format lerobot_datasets
    --dataset-strategy vlda
    --dataset-path "$DATA_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --num-workers "${NUM_WORKERS:-8}"
    --batch-drop-last
    --dataloader-seed-workers
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$((GLOBAL_BATCH_SIZE / BATCH_DIVISOR))"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --set-seed-by-rank
    --optimizer AdamW
    --weight-decay 1e-10
    --weight-decay-grouping all
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
    --clip-grad 1.0
    --lr-base "$PEAK_LR" --min-lr "$MIN_LR"
    --lr-decay-style cosine_with_min_lr --custom-lr-lambda
    --lr-warmup-iters "$WARMUP_ITERS" --lr-decay-iters "$TRAIN_ITERS"
    --lr-decay-start-iters "${LR_DECAY_START_ITERS:-0}"
    --distributed-strategy ddp --dtype bfloat16
    --ddp-find-unused-parameters --ddp-gradient-as-bucket-view
    --no-dynamo-optimize-ddp
    --no-check-for-nan-in-loss-and-grad --loss-spike-threshold inf
    --save-format safetensors --save-interval "$SAVE_INTERVAL"
    --log-interval "${LOG_INTERVAL:-100}"
    --tensorboard-dir "$TENSORBOARD_DIR"
)
if [[ "${DDP_STATIC_GRAPH:-0}" == 1 ]]; then
    ARGS+=(--ddp-static-graph)
fi
MODEL_OVERRIDES=("model.stage=$STAGE" "data.norm_stats_path=$NORM_STATS_PATH" "data.preprocessing_backend=$PREPROCESSING_BACKEND")
if [[ "$RESUME" == 1 ]]; then
    ARGS+=(--resume)
else
    : "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to pi05 base (stage 1) or VLDA stage 1 weights (stage 2)}"
    ARGS+=(--pretrained-checkpoint "$CHECKPOINT_PATH")
    if [[ "$STAGE" == 1 && -n "${VJEPA_CHECKPOINT:-}" ]]; then
        MODEL_OVERRIDES+=("model.vjepa_checkpoint=$VJEPA_CHECKPOINT")
    fi
fi
if [[ "${ZERO_OPTIMIZER:-0}" == 1 ]]; then
    ARGS+=(--zero-optimizer)
fi

# argparse's positional dotlist must follow all optional flags. Preserve user
# override precedence while allowing flags and dotlist entries in either order.
while (( $# )); do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="${2:?--output-dir requires a directory}"
            ARGS+=("$1" "$2")
            shift
            ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; ARGS+=("$1") ;;
        model.*=*|data.*=*) MODEL_OVERRIDES+=("$1") ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

export PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
LOG_DIR=${LOG_DIR:-"$OUTPUT_DIR/logs"}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"
echo "Stage $STAGE: output=$OUTPUT_DIR; console log=$LOG_FILE"
torchrun \
    --nproc_per_node "$GPUS_PER_NODE" --nnodes "$NNODES" --node_rank "$NODE_RANK" \
    --master_addr "${MASTER_ADDR:-localhost}" --master_port "${MASTER_PORT:-29500}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" "${ARGS[@]}" -- "${MODEL_OVERRIDES[@]}" \
    2>&1 | tee -a "$LOG_FILE"
