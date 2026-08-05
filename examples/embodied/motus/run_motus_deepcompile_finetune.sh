#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ═══════════════════════════════════════════════════════════════
# run_motus_deepcompile_finetune.sh - Motus VLA Training (DeepCompile)
#   (WAN video + Qwen3-VL + Action Expert three-modal MoT)
#   DeepSpeed ZeRO-1 + DeepCompile (compiled ZeRO-aware gradient reduce
#   scheduled into the inductor graph), single node, 8 GPUs.
#
# This is the DeepCompile variant of run_motus_ddp_zero1_finetune.sh:
#   - The model is wrapped by a DeepSpeed engine (direct deepspeed.initialize +
#     engine.compile(), no accelerate) that owns optimizer / backward / step.
#   - The CUDA-graph capture path is disabled (mutually exclusive), so the
#     capture-only env (DISABLE_ADDMM_CUDA_LT / CUBLAS_WORKSPACE_CONFIG) is NOT set.
#   - Gradient accumulation + batch geometry are delegated to the ds config
#     (injected from the launch flags at engine init).
#
# Usage:
#   bash run_motus_deepcompile_finetune.sh
#   bash run_motus_deepcompile_finetune.sh --train-iters 50000
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/AIAK-Training-Omni"}

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-"/workspace/motus/data/aloha_mobile_cabinet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/motus/outputs/motus-lerobot-deepcompile"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"${OUTPUT_DIR}/tensorboard-log"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/zero1_deepcompile.json"}
mkdir -p "$OUTPUT_DIR"

# ── DeepCompile env ───────────────────────────────────────────
# Prevent 8 ranks from compiling in parallel and blowing the cgroup memory limit.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=8
export TORCH_NCCL_USE_COMM_NONBLOCKING=1
# Skip albumentations' online version check (no network on the training node).
export NO_ALBUMENTATIONS_UPDATE=1

# ── Distributed ───────────────────────────────────────────────
GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── Model config ──────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"motus"}
MODEL_CONFIG_ARGS=(
    --model-name $MODEL_NAME
)

# ── Data params ───────────────────────────────────────────────
DATA_ARGS=(
    --dataset-format lerobot_datasets
    --dataset-strategy motus
    --dataset-path $DATA_PATH
    --video-backend torchcodec
    --num-workers 16
    # Seed DataLoader workers deterministically (worker_init_fn + generator from
    # --seed): worker w gets random/np/torch seeds = seed + w, matching base
    # (train.py create_dataloaders always seeds workers 42+worker_id). Without
    # this the transplant default (dataloader_seed_workers=False) leaves workers
    # unseeded -> any global-RNG op in the pipeline diverges from base.
    --dataloader-seed-workers
)

# ── Training params (mirror source configs/lerobot.yaml) ──────
TRAINER_TYPE=${TRAINER_TYPE:-"MotusTrainer"}
TRAINING_ARGS=(
    --trainer-type $TRAINER_TYPE
    --train-iters 1000000
    --per-device-batch-size 8
    --gradient-accumulation-steps 1
    --seed 42
    --output-dir $OUTPUT_DIR
    # Learning rate (mirror base deepcompile: LambdaLinearScheduler)
    #   base configs/lerobot.yaml: scheduler_type=linear, warmup=200, cycle=1000000,
    #   f_max=0.99, f_min=0.4, f_start=1e-6. lambda_linear here is formula-identical.
    --lr-base 5.0e-5
    --lr-decay-style lambda_linear
    --lr-warmup-iters 200
    --lambda-f-max 0.99
    --lambda-f-min 0.4
    --lambda-f-start 1e-6
    --lambda-cycle-length 1000000
    # step-0 LR parity with base: base's LambdaLinearScheduler.__init__ leaves
    # optimizer.lr untouched, so its FIRST step runs at the unscaled base_lr;
    # torch's LambdaLR applies schedule(0)=f_start(~0). This flag (OFF by
    # default) turns on that base-aligned step-0 behavior; drop it for torch's
    # default. step>=1 is identical either way.
    --lr-step0-unscaled
    # Optimizer
    --optimizer TorchFusedAdamW
    # Align to base: base's DeepCompile path skips grad clipping entirely
    # (train.py _skip_clip=True when USE_DEEPCOMPILE=1, and its DeepSpeed config
    # has no "gradient_clipping"). Set clip-grad<=0 so base_trainer takes the
    # no-clip branch (grad_norm=get_grad_norm, no scaling) -> optimizer update
    # matches base, removing a step-1 update discrepancy that seeds step-2 loss
    # divergence.
    --clip-grad 0.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    # Checkpoint
    --save-interval 5000
)

# ── DeepCompile / distributed ─────────────────────────────────
# --distributed-strategy ddp is required (DeepCompile is DDP-only); DO NOT pass
# --zero-optimizer (the DeepSpeed engine provides ZeRO-1 itself).
DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    --use-deepcompile
    --deepspeed-config $DEEPSPEED_CONFIG
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForgeVLA Motus Training (DeepSpeed ZeRO-1 + DeepCompile)"
echo "  Model:      $MODEL_NAME    Trainer: $TRAINER_TYPE"
echo "  Strategy:   lerobot_datasets / motus"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "  DS config:  $DEEPSPEED_CONFIG"
echo "════════════════════════════════════════════════════════════"

LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
