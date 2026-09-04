#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-/workspace/LoongForge}
DATA_PATH=${DATA_PATH:?Set DATA_PATH to an Energon WebDataset converted from OpenAI JSONL}
TOKENIZER_PATH=${TOKENIZER_PATH:?Set TOKENIZER_PATH to the Kimi K3 tokenizer}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the converted MCore checkpoint}
CHECKPOINT_SAVE_PATH=${CHECKPOINT_SAVE_PATH:-$CHECKPOINT_PATH}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

export FLA_TILELANG="${FLA_TILELANG:-0}"

torchrun \
  --nproc_per_node="$GPUS_PER_NODE" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$LOONGFORGE_PATH/loongforge/train.py" \
  --config-file "$LOONGFORGE_PATH/configs/models/kimi_k3/kimi_k3.yaml" \
  --tokenizer-type HFTokenizer \
  --hf-tokenizer-path "$TOKENIZER_PATH" \
  --data-path "$DATA_PATH" \
  --split 99990,8,2 \
  --training-phase pretrain \
  --task-encoder KimiTaskEncoder \
  --seq-length "${SEQ_LENGTH:-32768}" \
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-1048576}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-128}" \
  --train-iters "${TRAIN_ITERS:-1500}" \
  --lr "${LR:-1e-6}" \
  --min-lr "${MIN_LR:-1e-7}" \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --load "$CHECKPOINT_PATH" \
  --save "$CHECKPOINT_SAVE_PATH" \
  --save-interval "${SAVE_INTERVAL:-100}" \
  --eval-interval "${EVAL_INTERVAL:-30}" \
  --eval-iters "${EVAL_ITERS:-10}" \
  --no-load-optim \
  --no-load-rng \
  --moe-grouped-gemm \
  --tensor-model-parallel-size "${TP:-1}" \
  --pipeline-model-parallel-size "${PP:-1}" \
  --expert-model-parallel-size "${EP:-8}" \
  --expert-tensor-parallel-size "${ETP:-1}" \
  --moe-token-dispatcher-type alltoall \
  --use-distributed-optimizer
