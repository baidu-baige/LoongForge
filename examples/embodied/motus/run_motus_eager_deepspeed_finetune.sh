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
# run_motus_eager_deepspeed_finetune.sh - Motus VLA Training
#   EAGER + DeepSpeed ZeRO-1 (NO DeepCompile / NO inductor compile).
#
# This is the eager counterpart of run_motus_deepcompile_finetune.sh, used for
# loss-parity vs the base eager+deepspeed run (USE_DEEPSPEED=1, no USE_DEEPCOMPILE):
#   - EAGER_DEEPSPEED=1 makes MotusTrainer build a DeepSpeed ZeRO-1 engine via
#     direct deepspeed.initialize but SKIP engine.compile() and the static attn
#     path; forward/backward run eager (varlen flash, model.training_step) while
#     ZeRO-1 owns the reduce + fp32-master Adam step.
#   - --use-deepcompile is NOT passed (that would take the compiled path); the
#     ds config's "compile" block is stripped at init.
#   - Same ZeRO-1 config, batch geometry, LR schedule, optimizer, no grad clip.
#
# Usage:
#   EAGER_DEEPSPEED=1 PARITY_DATA_SEED=42 PARITY_FM_SEED=42 \
#     bash run_motus_eager_deepspeed_finetune.sh --loss-log-rank 0
#   (step-0 LR parity is now the built-in --lr-step0-unscaled flag; OFF by
#    default, enabled explicitly in the LR args below. Old PARITY_LR_STEP0 env
#    gate is gone.)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/AIAK-Training-Omni"}

# Force the eager DeepSpeed path in MotusTrainer.
export EAGER_DEEPSPEED=1

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-"/workspace/motus/data/aloha_mobile_cabinet"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/motus/outputs/motus-lerobot-eager-deepspeed"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"${OUTPUT_DIR}/tensorboard-log"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/zero1_deepcompile.json"}
mkdir -p "$OUTPUT_DIR"

# ── Env ───────────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=8
export TORCH_NCCL_USE_COMM_NONBLOCKING=1
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

# ── Training params (mirror base configs/lerobot.yaml) ────────
TRAINER_TYPE=${TRAINER_TYPE:-"MotusTrainer"}
TRAINING_ARGS=(
    --trainer-type $TRAINER_TYPE
    --train-iters 1000000
    --per-device-batch-size 8
    --gradient-accumulation-steps 1
    --seed 42
    --output-dir $OUTPUT_DIR
    --lr-base 5.0e-5
    --lr-decay-style lambda_linear
    --lr-warmup-iters 200
    --lambda-f-max 0.99
    --lambda-f-min 0.4
    --lambda-f-start 1e-6
    --lambda-cycle-length 1000000
    # step-0 LR parity with base (flag is OFF by default; enabled here for the
    # parity run): base leaves optimizer.lr untouched at scheduler init so step-0
    # runs at the unscaled base_lr, while torch's LambdaLR applies
    # schedule(0)=f_start(~0). Drop this flag for torch's default.
    --lr-step0-unscaled
    --optimizer AdamW
    # No grad clip (match base eager+deepspeed: ds config has no gradient_clipping).
    --clip-grad 0.0
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --save-interval 5000
)

# ── DeepSpeed (eager) ─────────────────────────────────────────
# NOTE: --use-deepcompile is deliberately NOT passed. EAGER_DEEPSPEED=1 (env
# above) routes MotusTrainer to deepspeed.initialize WITHOUT engine.compile().
# --deepspeed-config still supplies the ZeRO-1 json (its "compile" block is
# stripped at init).
DISTRIBUTED_TRAINING_ARGS=(
    --distributed-strategy ddp
    --dtype bfloat16
    --deepspeed-config $DEEPSPEED_CONFIG
)

# ── Logging params ────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
)

# ── Launch ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  LoongForgeVLA Motus Training (EAGER + DeepSpeed ZeRO-1)"
echo "  Model:      $MODEL_NAME    Trainer: $TRAINER_TYPE"
echo "  GPUs:       $GPUS_PER_NODE"
echo "  Output:     $OUTPUT_DIR"
echo "  DS config:  $DEEPSPEED_CONFIG"
echo "  EAGER_DEEPSPEED=$EAGER_DEEPSPEED  (no engine.compile, no static attn path)"
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
