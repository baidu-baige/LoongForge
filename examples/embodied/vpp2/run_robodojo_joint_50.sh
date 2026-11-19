#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_robodojo_joint_50.sh - RoboDojo Joint Training (ZeRO-2 CPU Offload)
#
# Usage:
#   bash run_robodojo_joint_50.sh
#   TRAIN_ITERS=50 SEED=100008 bash run_robodojo_joint_50.sh
#   bash run_robodojo_joint_50.sh --train-iters 100
#   bash run_robodojo_joint_50.sh model.action_horizon=16
#
# Defaults: 8 GPUs × batch 2 = GBS16, 50 updates, no evaluation/checkpoint saves.
# Set ROBODOJO_RESOURCES to the shell file defining training resource paths.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
ROBODOJO_RESOURCES=${ROBODOJO_RESOURCES:-/workspace/vpp2_assets/robodojo_training_resources.sh}
source "$ROBODOJO_RESOURCES"
# Select system Python after sourcing the shared community resource file.
PYTHON_BIN=${LOONGFORGE_PYTHON_BIN:-/usr/bin/python3}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$LOONGFORGE_PATH"
export SEED=${SEED:-100008}
export PYTHONHASHSEED="$SEED"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VPP2_TORCH_LOAD_MMAP=1

# Deterministic CUDA policy used by the validated community baseline.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1
export NCCL_ALGO=Ring
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUDA_DEVICE_MAX_CONNECTIONS=8
# Single-threaded MKL avoids the observed CPU AdamW sqrt crash on this image.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

# ── CUDA libraries ────────────────────────────────────────────
# Match the image cuBLAS used for the precision-validated baseline.
CUBLAS_ROOT=${ROBODOJO_CUBLAS_ROOT:-/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib}
for LIBRARY in libcublasLt.so.13 libcublas.so.13; do
    if [[ ! -r "$CUBLAS_ROOT/$LIBRARY" ]]; then
        echo "Missing image CUDA library: $CUBLAS_ROOT/$LIBRARY" >&2
        exit 1
    fi
done
export LD_PRELOAD="$CUBLAS_ROOT/libcublasLt.so.13:$CUBLAS_ROOT/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
# This single-node recipe must not inherit a managed job's rendezvous settings.
MASTER_ADDR=127.0.0.1
MASTER_PORT=${ROBODOJO_MASTER_PORT:-29652}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

# ── Paths ─────────────────────────────────────────────────────
DATASET_PATH=${DATASET_PATH:-"$VPP2_PREPARED/train.csv"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/vpp2_robodojo_zero2/$(date +%Y%m%d_%H%M%S_%N)"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"$SCRIPT_DIR/deepspeed_robodojo_zero2.json"}

# ── Training defaults ─────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-50}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}

# ── Model config ──────────────────────────────────────────────
MODEL_CONFIG_ARGS=(
    --model-name vpp2
    --config-file "$LOONGFORGE_PATH/configs/models/embodied/vpp2_robodojo.yaml"
)

CONFIG_OVERRIDES=(
    "model.initialization_seed=$SEED"
    "data.pipeline.seed=$SEED"
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy vpp2_robodojo
    --dataset-path "$DATASET_PATH"
    --num-workers "$NUM_WORKERS"
    --dataloader-multiprocessing-context fork
)

# ── Training params ───────────────────────────────────────────
TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --set-seed-by-rank
    --deterministic-mode
    --output-dir "$OUTPUT_DIR"
    # Keep the original 500 / 80k / 100k schedule when stopping early.
    --lr-base 1e-4
    --lr-group video_expert=5e-6
    --lr-warmup-iters 500
    --lr-decay-iters 100000
    --optimizer AdamW
    --adam-beta1 0.9
    --adam-beta2 0.95
    --weight-decay 0.01
    --clip-grad 1.0
    --save-interval "$SAVE_INTERVAL"
    --no-check-for-nan-in-loss-and-grad
    --loss-spike-threshold inf
)

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy deepspeed
    --deepspeed-config "$DEEPSPEED_CONFIG"
    --dtype bfloat16
    --no-init-on-meta
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --wandb-mode disabled
)

# ── Launch ────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cd "$LOONGFORGE_PATH"

echo "════════════════════════════════════════════════════════════"
echo "  LoongForge RoboDojo Joint Training (DeepSpeed ZeRO-2)"
echo "  Python:     $PYTHON_BIN"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Model:      vpp2 (joint)"
echo "  Data:       $DATASET_PATH"
echo "  Iterations: $TRAIN_ITERS"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

launch_training() {
    # argparse's trailing dotlist positional must follow all option arguments.
    local argument
    local -a user_training_args=() user_config_args=()
    for argument in "$@"; do
        case "$argument" in
            model.*=*|data.*=*) user_config_args+=("$argument") ;;
            *) user_training_args+=("$argument") ;;
        esac
    done
    exec "$PYTHON_BIN" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
        "${MODEL_CONFIG_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DISTRIBUTED_TRAINING_ARGS[@]}" \
        "${LOGGING_ARGS[@]}" \
        "${user_training_args[@]}" \
        "${CONFIG_OVERRIDES[@]}" \
        "${user_config_args[@]}"
}

launch_training "$@"
