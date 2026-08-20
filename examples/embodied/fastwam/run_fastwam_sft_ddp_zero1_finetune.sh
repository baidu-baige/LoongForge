#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# ═══════════════════════════════════════════════════════════════
# run_fastwam_sft_ddp_zero1_finetune.sh - FastWAM SFT Launch Script (DDP + ZeRO-1)
#
# Delta versus run_fastwam_sft_ddp_finetune.sh:
#   --zero-optimizer                    wrap the optimizer in
#                                       ZeroRedundancyOptimizer, sharding
#                                       optimizer states across ranks. Only
#                                       effective with --distributed-strategy ddp.
#   --no-ddp-find-unused-parameters     skip the unused-parameter scan; FastWAM's
#                                       forward graph has no conditional branches.
#   --ddp-static-graph                  graph is identical every iteration, lets
#                                       DDP reuse its bucket/reduction plan.
#   --ddp-gradient-as-bucket-view       expose grads as views into the comm
#                                       buckets instead of separate allocations.
#   --no-ddp-broadcast-buffers          no BN-style buffers to sync each forward.
#   --ddp-bucket-cap-mb                 larger buckets: fewer, bigger all-reduces.
#
# The memory saved by ZeRO-1 is what makes the larger --per-device-batch-size
# below affordable relative to the plain DDP script.
#
# One optional ZeRO knob is left off by default:
#   --zero-master-param-dtype fp32      rank-local fp32 master params, broadcast
#                                       after each step. Better numerics under
#                                       bf16 training at some bandwidth cost.
#
# ── Throughput recipe ─────────────────────────────────────────
# The flags below the ZeRO block were each measured on 8xA800 (LIBERO-10, 224x448
# two-camera, 9 frames), 110 iterations with 10 warmed up and 100 timed, comparing
# paired runs inside one sweep (across sweeps the same config drifts by up to 3.4%,
# so only paired numbers are meaningful):
#
#   --optimizer TorchFusedAdamW           +3.9%  the default AdamW is unfused here
#   --cudnn-benchmark                     +1.4%  autotunes the VAE convolutions
#   --zero-parameters-as-bucket-view      +2.0%  1651 per-tensor broadcasts -> 8
#   model.disable_train_autocast          +3.2%  params are already bf16, so the
#                                                autocast wrapper only adds casts
#   model.drop_all_true_cross_attn_mask   +0.8%  an all-True mask forces SDPA off
#                                                its flash kernel onto cutlass
#   model.compile_vae_encode              +1.6%
#   model.mot_compile_blocks=both         +5.7%  at --per-device-batch-size 24
#   + model.rmsnorm_impl=wan                     (see below)
#
# Cumulative: 82.9 -> 102.3 samples/s (8 GPUs) at batch 24.
#
# `mot_compile_blocks=both` and `rmsnorm_impl=wan` must be set together, and the
# pairing is conditional:
#   * Compiling the MoT blocks only pays if the graph has no breaks. The TE
#     RMSNorm and the Triton RoPE are opaque to Dynamo, so with `rmsnorm_impl=te`
#     the compiled region fragments and throughput *drops* 7.2%.
#   * `rmsnorm_impl=wan` (`F.rms_norm`) is the slow path in eager on torch < 2.9,
#     where it decomposes into seven fp32 kernels - but inside a compiled region
#     Inductor fuses it into one Triton kernel, so the penalty is never paid and
#     peak memory even drops slightly (74.90 vs 75.06 GiB).
#   * So: compiling -> `wan`; not compiling -> `te`. On torch >= 2.9 `F.rms_norm`
#     has a native fused kernel and this should be re-measured.
#   * The gain is batch-dependent: at batch 16 the step is kernel-launch bound and
#     the same config is a wash. Measure on the batch size you will train at.
#
# Two more knobs are deliberately left off:
#   --no-check-for-nan-in-loss-and-grad  worth ~1.4% (a nan_to_num sweep over
#                                        6.02 B gradients each step), but it
#                                        removes the divergence guard.
#   PER_DEVICE_BATCH_SIZE=24             worth +7.9% over 16 and fits in 75.06 of
#                                        79.33 GiB, but it changes the effective
#                                        batch size, which is a training decision.
#
# Usage:
#   bash run_fastwam_sft_ddp_zero1_finetune.sh
#   DATASET_PATH=/path/to/libero TOKENIZER_PATH=/path/to/tokenizer \
#     bash run_fastwam_sft_ddp_zero1_finetune.sh
#   GPUS_PER_NODE=4 bash run_fastwam_sft_ddp_zero1_finetune.sh                   # override via env
#   bash run_fastwam_sft_ddp_zero1_finetune.sh --train-iters 50                  # override a flag
#   bash run_fastwam_sft_ddp_zero1_finetune.sh model.action_dit_pretrained_path=/path  # dotlist form
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "$SCRIPT_DIR/../../.." && pwd)"}
export LOCAL_VLA_ARTIFACTS_ROOT=${LOCAL_VLA_ARTIFACTS_ROOT:-"/ssd2/loongforge_embodied_ci/vla_artifacts"}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/"}

# ── Distributed ───────────────────────────────────────────────
# Cluster schedulers commonly export WORLD_SIZE (node count) and RANK (node rank).
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29519"}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-${RANK:-0}}

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# ── Paths ─────────────────────────────────────────────────────
TOKENIZER_PATH=${TOKENIZER_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/Wan2.2-TI2V-5B"}
DATASET_PATH=${DATASET_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"$LOONGFORGE_PATH/outputs/fastwam_sft_zero1"}

PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/models/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"}
TEXT_EMBEDDING_CACHE_DIR=${TEXT_EMBEDDING_CACHE_DIR:-"$LOCAL_VLA_ARTIFACTS_ROOT/fastwam/datasets/text_embeds"}

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"fastwam"}
MODEL_CONFIG_ARGS=(
    --model-name "$MODEL_NAME"
)

# ── Data params ───────────────────────────────────────────────
NUM_WORKERS=${NUM_WORKERS:-16}
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy fastwam
    --dataset-path "$DATASET_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --robot-type libero_franka
    --num-workers "$NUM_WORKERS"
    --lerobotdataset-version v2.1
    --video-backend pyav
)

# ── Training params ───────────────────────────────────────────
TRAIN_ITERS=${TRAIN_ITERS:-20000}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
SEED=${SEED:-3047}

TRAINING_ARGS=(
    --trainer-type FinetuneTrainer
    --train-iters "$TRAIN_ITERS"
    --per-device-batch-size "$PER_DEVICE_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    # Learning rate
    --lr-base 1.0e-8
    --lr-decay-style cosine_warmup_with_min_lr
    --lr-warmup-iters 0
    --min-lr 1.0e-9
    # Optimizer
    --clip-grad 1.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    # Checkpoint
    --save-interval "$SAVE_INTERVAL"
)

if [[ -n "$PRETRAINED_CHECKPOINT" ]]; then
    TRAINING_ARGS+=(--pretrained-checkpoint "$PRETRAINED_CHECKPOINT")
fi

DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --zero-optimizer
    --no-ddp-find-unused-parameters
    --ddp-static-graph
    --ddp-gradient-as-bucket-view
    --no-ddp-broadcast-buffers
    --ddp-bucket-cap-mb 200
    --dtype bfloat16
    --zero-parameters-as-bucket-view
)

# ── Throughput params ─────────────────────────────────────────
# See the recipe block at the top of this file for the measured gain of each.
PERF_ARGS=(
    --optimizer TorchFusedAdamW
    --cudnn-benchmark
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --wandb-project loongforge-vla
    --wandb-mode disabled
)

# ── Model/data dotlist overrides ──────────────────────────────
# The four performance overrides pair with PERF_ARGS above; `mot_compile_blocks`
# and `rmsnorm_impl` must move together (see the recipe block at the top).
MODEL_DATA_OVERRIDES=(
    model.disable_train_autocast=true
    model.drop_all_true_cross_attn_mask=true
    model.compile_vae_encode=true
    model.mot_compile_blocks=both
    model.rmsnorm_impl=wan
)
if [[ -n "$ACTION_DIT_PRETRAINED_PATH" ]]; then
    MODEL_DATA_OVERRIDES+=("model.action_dit_pretrained_path=$ACTION_DIT_PRETRAINED_PATH")
fi
if [[ -n "$TEXT_EMBEDDING_CACHE_DIR" ]]; then
    MODEL_DATA_OVERRIDES+=("data.text_embedding_cache_dir=$TEXT_EMBEDDING_CACHE_DIR")
fi

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForge FastWAM SFT (DDP + ZeRO-1)"
echo "  GPUs:       $GPUS_PER_NODE x $NNODES node(s)"
echo "  Model:      $MODEL_NAME"
echo "  Data:       $DATASET_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${MODEL_DATA_OVERRIDES[@]+"${MODEL_DATA_OVERRIDES[@]}"}" \
    "$@"
