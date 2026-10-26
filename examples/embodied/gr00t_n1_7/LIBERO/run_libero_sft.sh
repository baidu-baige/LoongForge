#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# run_libero_sft.sh - GR00T N1.7 LIBERO SFT on LoongForge, DDP
#
# Each suite is finetuned separately -> one checkpoint per suite.
#
# Usage:
#   bash run_libero_sft.sh spatial
#   bash run_libero_sft.sh all                                    # spatial object goal 10
#   TRAIN_ITERS=20 SAVE_INTERVAL=20 bash run_libero_sft.sh goal   # smoke test
#   GPUS_PER_NODE=4 bash run_libero_sft.sh goal                   # 4x160
#   bash run_libero_sft.sh spatial -- --log-interval 10           # extra train.py flags
#   LOONGFORGE_PATH=/path/to/LoongForge CHECKPOINT_PATH=/path/to/GR00T-N1.7-3B \
#   COSMOS_LOCAL_PATH=/path/to/Cosmos-Reason2-2B DATASET_ROOT=/path/to/datasets \
#   OUTPUT_ROOT=/path/to/outputs bash run_libero_sft.sh spatial
set -euo pipefail

# ---- Paths ----
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/path/to/LoongForge"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/path/to/gr00t/GR00T-N1.7-3B"}
# Holds libero_<suite>_no_noops_1.0.0_lerobot/ for each suite.
DATASET_ROOT=${DATASET_ROOT:-"/path/to/datasets/IPEC-COMMUNITY"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/path/to/outputs/gr00t_n1_7_libero"}
# Qwen3-VL / Cosmos-Reason2 backbone (architecture + processor/tokenizer).
export COSMOS_LOCAL_PATH=${COSMOS_LOCAL_PATH:-"/path/to/gr00t/Cosmos-Reason2-2B"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"$COSMOS_LOCAL_PATH"}
GR00T_N1_7_OPS_SRC=${GR00T_N1_7_OPS_SRC:-"$LOONGFORGE_PATH/ops/cuda_source/gr00t_n1_7_op"}
TRAIN_PY=${TRAIN_PY:-"python"}

# ---- Suites ----
SUITES=()
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; EXTRA_ARGS=("$@"); break; fi
    SUITES+=("$1"); shift
done
if [ "${#SUITES[@]}" -eq 1 ] && [ "${SUITES[0]}" = "all" ]; then
    SUITES=(spatial object goal 10)
fi

# GR00T-N1.7 fused CUDA ops (built once, then importable).
if ! "$TRAIN_PY" -c "import gr00t_n1_7_op" >/dev/null 2>&1; then
    [ -f "$GR00T_N1_7_OPS_SRC/setup.py" ] || {
        echo "ERROR: gr00t_n1_7_op not installed and sources not found at $GR00T_N1_7_OPS_SRC." >&2
        exit 1
    }
    echo "Installing GR00T-N1.7 CUDA operators from $GR00T_N1_7_OPS_SRC"
    NVCC_THREADS=${NVCC_THREADS:-${MAX_JOBS:-8}} \
        "$TRAIN_PY" -m pip install --no-build-isolation -e "$GR00T_N1_7_OPS_SRC"
fi

# ---- Env ----
export PYTHONPATH="$LOONGFORGE_PATH:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}

# ---- Distributed ----
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "${NODE_RANK:-${RANK:-0}}"
    --master_addr "${MASTER_ADDR:-localhost}"
    --master_port "${MASTER_PORT:-29500}"
)

# ---- Hyper-parameters (official recipe) ----
TRAIN_ITERS=${TRAIN_ITERS:-20000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-640}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-4}
# warmup_ratio 0.05 x max_steps, as iterations.
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-$((TRAIN_ITERS / 20))}

DENOM=$((GPUS_PER_NODE * NNODES * GRADIENT_ACCUMULATION_STEPS))
if [ $((GLOBAL_BATCH_SIZE % DENOM)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE not divisible by" \
         "gpus($((GPUS_PER_NODE * NNODES))) x grad_accum($GRADIENT_ACCUMULATION_STEPS)." >&2
    exit 1
fi
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / DENOM))}

# ---- Preflight ----
[ -f "$CHECKPOINT_PATH/config.json" ] || {
    echo "ERROR: no base model at $CHECKPOINT_PATH (config.json missing)." >&2
    exit 1
}
[ -f "$COSMOS_LOCAL_PATH/config.json" ] || {
    echo "ERROR: no VL backbone at $COSMOS_LOCAL_PATH (config.json missing)." >&2
    exit 1
}
for suite in "${SUITES[@]}"; do
    d="$DATASET_ROOT/libero_${suite}_no_noops_1.0.0_lerobot"
    [ -f "$d/meta/modality.json" ] || {
        echo "ERROR: $d/meta/modality.json missing (LIBERO modality patch not applied)." >&2
        exit 1
    }
done

# ---- Fixed args (shared by all suites) ----
DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    --ddp-bucket-cap-mb 100
    --no-ddp-find-unused-parameters
    --no-check-for-nan-in-loss-and-grad
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --cuda-graph-warmup-steps 3
    --cuda-graph-pad-length 0
    --cuda-graph-ddp-sync-in-graph
    --ddp-static-graph
)

# LIBERO embodiment: projector index 2 + libero_sim modality layout
# (state [x, y, z, axis_angle(3), gripper_qpos(2)], 16-step action chunk),
# the counterpart of the official --embodiment-tag LIBERO_PANDA.
OVERRIDES=(
    data.embodiment_tag=libero_sim
)
mkdir -p "$OUTPUT_ROOT"

for suite in "${SUITES[@]}"; do
    DATASET_PATH="$DATASET_ROOT/libero_${suite}_no_noops_1.0.0_lerobot"
    OUTPUT_DIR="$OUTPUT_ROOT/libero_${suite}"
    LOG_FILE="$OUTPUT_ROOT/libero_${suite}.log"
    mkdir -p "$OUTPUT_DIR"

    DATA_ARGS=(
        --dataset-format lerobot_datasets
        --dataset-strategy gr00t_n1_7
        --dataset-path "$DATASET_PATH"
        --lerobotdataset-version v2.1
        --video-backend torchcodec
        --num-workers "$NUM_WORKERS"
        --dataloader-prefetch-factor 8
        --dataloader-multiprocessing-context fork
        --distributed-sampler-mode block
    )

    TRAINING_ARGS=(
        --trainer-type GrootN1d7Trainer
        --save-processor-config
        --train-iters "$TRAIN_ITERS"
        --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
        --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
        --seed "$SEED"
        --output-dir "$OUTPUT_DIR"
        # HF cosine schedule decaying to 0, 5% warmup
        --lr-base 1.0e-4
        --lr-decay-style cosine_with_min_lr
        --lr-warmup-iters "$LR_WARMUP_ITERS"
        --min-lr 0.0
        # HF Trainer AdamW defaults; weight decay on everything except bias/norm.
        --optimizer AdamW
        --clip-grad 1.0
        --weight-decay 1.0e-5
        --weight-decay-grouping bias_norm
        --adam-beta1 0.9
        --adam-beta2 0.999
        --adam-eps 1e-8
        # pt, not safetensors: under transformers 5.3.0 visual.patch_embed.proj.weight
        # is non-contiguous and the safetensors writer rejects it.
        --save-interval "$SAVE_INTERVAL"
        --save-format "${SAVE_FORMAT:-pt}"
        --pretrained-checkpoint "$CHECKPOINT_PATH"
    )

    LOGGING_ARGS=(
        --log-interval "${LOG_INTERVAL:-1}"
        --loss-log-rank -1
        --wandb-mode "${WANDB_MODE:-disabled}"
        --tensorboard-dir "$OUTPUT_DIR/tensorboard"
    )

    echo "=============================================================="
    echo "  GR00T-N1.7 SFT - LIBERO ${suite}"
    echo "  Base model: $CHECKPOINT_PATH"
    echo "  Backbone:   $COSMOS_LOCAL_PATH"
    echo "  Dataset:    $DATASET_PATH"
    echo "  Output:     $OUTPUT_DIR"
    echo "  GPUs=$((GPUS_PER_NODE * NNODES)) iters=$TRAIN_ITERS global_bs=$GLOBAL_BATCH_SIZE" \
         "(per-device $PER_DEVICE_BATCH_SIZE x accum $GRADIENT_ACCUMULATION_STEPS)"
    echo "  warmup=$LR_WARMUP_ITERS save_interval=$SAVE_INTERVAL seed=$SEED"
    echo "  Log:        $LOG_FILE"
    echo "=============================================================="

    "$TRAIN_PY" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
        "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
        --model-name gr00t_n1_7 \
        "${DATA_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DISTRIBUTED_TRAINING_ARGS[@]}" \
        "${LOGGING_ARGS[@]}" \
        "${OVERRIDES[@]}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
done

echo "All requested LIBERO suites finished. Checkpoints under $OUTPUT_ROOT/libero_<suite>/checkpoints/steps_<N>"
