#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# run_fractal_sft.sh - GR00T N1.7 SimplerEnv Fractal (Google Robot) SFT on LoongForge, DDP
#
# Usage:
#   bash run_fractal_sft.sh
#   TRAIN_ITERS=20 SAVE_INTERVAL=20 bash run_fractal_sft.sh          # smoke test
#   GPUS_PER_NODE=4 bash run_fractal_sft.sh                          # 4x256
#   bash run_fractal_sft.sh -- --log-interval 10                     # extra train.py flags
#   LOONGFORGE_PATH=/path/to/LoongForge CHECKPOINT_PATH=/path/to/GR00T-N1.7-3B \
#   COSMOS_LOCAL_PATH=/path/to/Cosmos-Reason2-2B DATASET_PATH=/path/to/fractal20220817_data_lerobot \
#   OUTPUT_ROOT=/path/to/outputs bash run_fractal_sft.sh
set -euo pipefail

# ---- Paths ----
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/path/to/LoongForge"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/path/to/gr00t/GR00T-N1.7-3B"}
# Videos must already be converted from AV1 to H.264 (torchcodec cannot decode AV1).
DATASET_PATH=${DATASET_PATH:-"/path/to/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/path/to/outputs/gr00t_n1_7_fractal"}
# Qwen3-VL / Cosmos-Reason2 backbone (architecture + processor/tokenizer).
export COSMOS_LOCAL_PATH=${COSMOS_LOCAL_PATH:-"/path/to/gr00t/Cosmos-Reason2-2B"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"$COSMOS_LOCAL_PATH"}
GR00T_N1_7_OPS_SRC=${GR00T_N1_7_OPS_SRC:-"$LOONGFORGE_PATH/ops/cuda_source/gr00t_n1_7_op"}
TRAIN_PY=${TRAIN_PY:-"python"}

# ---- Extra train.py flags (after --) ----
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; EXTRA_ARGS=("$@"); break; fi
    echo "WARN: ignoring positional arg '$1' (use '-- <flags>' to pass train.py flags)." >&2
    shift
done

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

# ---- Hyper-parameters (official Fractal recipe) ----
TRAIN_ITERS=${TRAIN_ITERS:-50000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
# Official --state-dropout-prob 0.5, applied in both data preprocessing and the model.
STATE_DROPOUT_PROB=${STATE_DROPOUT_PROB:-0.5}
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

OUTPUT_DIR="$OUTPUT_ROOT/fractal"
LOG_FILE="$OUTPUT_ROOT/fractal.log"
mkdir -p "$OUTPUT_DIR"

MODEL_CONFIG_ARGS=(
    --model-name gr00t_n1_7
)

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
    # HF Trainer adamw_torch_fused defaults; weight decay on everything except bias/norm.
    --optimizer TEFusedAdamW
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

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    --ddp-bucket-cap-mb 100
    --no-ddp-find-unused-parameters
)

LOGGING_ARGS=(
    --log-interval "${LOG_INTERVAL:-1}"
    --loss-log-rank -1
    --wandb-mode "${WANDB_MODE:-disabled}"
    --tensorboard-dir "$OUTPUT_DIR/tensorboard"
)

# Google robot embodiment, the counterpart of the official --embodiment-tag SIMPLER_ENV_GOOGLE.
# No built-in modality config for it: the layout comes from the dataset meta/modality.json.
# 8-step action chunk, so override the yaml's 16.
OVERRIDES=(
    data.embodiment_tag=simpler_env_google
    model.state_dropout_prob=$STATE_DROPOUT_PROB
    data.state_dropout_prob=$STATE_DROPOUT_PROB
    data.preprocess_action_horizon=8
)

echo "=============================================================="
echo "  GR00T-N1.7 SFT - SimplerEnv Fractal (Google Robot)"
echo "  Base model: $CHECKPOINT_PATH"
echo "  Backbone:   $COSMOS_LOCAL_PATH"
echo "  Dataset:    $DATASET_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "  GPUs=$((GPUS_PER_NODE * NNODES)) iters=$TRAIN_ITERS global_bs=$GLOBAL_BATCH_SIZE" \
     "(per-device $PER_DEVICE_BATCH_SIZE x accum $GRADIENT_ACCUMULATION_STEPS)"
echo "  warmup=$LR_WARMUP_ITERS save_interval=$SAVE_INTERVAL seed=$SEED" \
     "state_dropout=$STATE_DROPOUT_PROB"
echo "  Log:        $LOG_FILE"
echo "=============================================================="

"$TRAIN_PY" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${OVERRIDES[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"

echo "Fractal SFT finished. Checkpoints under $OUTPUT_DIR/checkpoints/steps_<N>"
