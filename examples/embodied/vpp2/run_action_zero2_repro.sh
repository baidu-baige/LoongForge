#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
: "${VPP2_ASSETS_ROOT:?Set VPP2_ASSETS_ROOT to the shared asset directory}"
export VPP2_DATASET_ROOT="${VPP2_DATASET_ROOT:-$VPP2_ASSETS_ROOT/libero_subset}"
export VPP2_VIDEO_CHECKPOINT="${VPP2_VIDEO_CHECKPOINT:-$VPP2_ASSETS_ROOT/libero_video_step010000.pt}"
export VPP2_ALIGNMENT_INIT="${VPP2_ALIGNMENT_INIT:-$VPP2_ASSETS_ROOT/alignment/init_seed42.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/action_zero2_$(date +%Y%m%d_%H%M%S)}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}" PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
export TRAIN_ITERS="${TRAIN_ITERS:-100}" SCHEDULE_STEPS="${SCHEDULE_STEPS:-30000}"
export NUM_WORKERS="${NUM_WORKERS:-0}" SEED="${SEED:-42}"
export VPP2_DETERMINISTIC="${VPP2_DETERMINISTIC:-0}"
if [[ "$VPP2_DETERMINISTIC" == 1 ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
export PYTHONHASHSEED="$SEED" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "${1:-}" == --dry-run ]]; then
  exec "$PYTHON_BIN" "$HERE/prepare_run.py" --repo "$ROOT" --side loongforge --dry-run
fi
[[ $# == 0 ]] || { echo 'Use environment overrides, not positional arguments' >&2; exit 2; }
"$PYTHON_BIN" "$HERE/prepare_run.py" --repo "$ROOT" --side loongforge
DETERMINISM_ARGS=()
if [[ "$VPP2_DETERMINISTIC" == 1 ]]; then
  DETERMINISM_ARGS+=(--deterministic-mode)
fi
"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$GPUS_PER_NODE"   --master_port "${MASTER_PORT:-29542}" "$ROOT/loongforge/embodied/train.py"   --model-name vpp2 --dataset-format lerobot_datasets --dataset-strategy vpp2   --dataset-path "$VPP2_DATASET_ROOT/libero_spatial_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_object_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_goal_no_noops_lerobot,$VPP2_DATASET_ROOT/libero_10_no_noops_lerobot"   --video-backend pyav --num-workers "$NUM_WORKERS" --trainer-type FinetuneTrainer   --train-iters "$TRAIN_ITERS" --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"   --gradient-accumulation-steps 1 --seed "$SEED" --set-seed-by-rank --output-dir "$OUTPUT_DIR"   --lr-base 1e-4 --lr-decay-style linear_warmup_cosine_annealing --lr-warmup-iters 500   --lr-decay-iters "$SCHEDULE_STEPS" --min-lr 1e-6 --optimizer AdamW --weight-decay 0.01   --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 0 --dtype bfloat16   --save-interval "${SAVE_INTERVAL:-0}" --save-format pt --distributed-strategy deepspeed   --deepspeed-config "$OUTPUT_DIR/zero2.json" --no-init-on-meta   --log-interval 1 --no-check-for-nan-in-loss-and-grad --loss-spike-threshold inf --wandb-mode disabled   "${DETERMINISM_ARGS[@]}" model.disable_train_autocast=true model.alignment_init_path="$VPP2_ALIGNMENT_INIT"   model.alignment_seed="$SEED" 2>&1 | tee "$OUTPUT_DIR/console.log"
