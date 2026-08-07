#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}
MEGATRON_PATH=${MEGATRON_PATH:-"${LOONGFORGE_PATH}/third_party/Loong-Megatron"}

DATA_PATH=${DATA_PATH:?Set DATA_PATH to the Energon multimodal dataset}
TOKENIZER_PATH=${TOKENIZER_PATH:?Set TOKENIZER_PATH to the tokenizer directory}
SAVE_PATH=${SAVE_PATH:?Set SAVE_PATH to the output checkpoint directory}
LOAD_PATH=${LOAD_PATH:-}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"${SAVE_PATH}/tensorboard"}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-1}
TRAIN_ITERS=${TRAIN_ITERS:-1000}
EVAL_ITERS=${EVAL_ITERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

MODEL_CONFIG_ARGS=(
    --config-file "${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/minicpm_v_4_6.yaml"
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "${TOKENIZER_PATH}"
    --data-path "${DATA_PATH}"
    --dataloader-type external
    --task-encoder MiniCPMV46TaskEncoder
    --split 100,0,0
    --add-question-in-pretrain
    --num-workers "${NUM_WORKERS}"
)

TRAINING_ARGS=(
    --training-phase pretrain
    --seq-length 4096
    --max-position-embeddings 262144
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --lr 1e-5
    --min-lr 1e-6
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --train-iters "${TRAIN_ITERS}"
    --eval-iters "${EVAL_ITERS}"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.02
    --bf16
    --mtp-num-layers "${MTP_NUM_LAYERS}"
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --distributed-backend nccl
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --save "${SAVE_PATH}"
    --save-interval "${SAVE_INTERVAL}"
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
)

if [ -n "${LOAD_PATH}" ]; then
    TRAINING_ARGS+=(--load "${LOAD_PATH}")
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${LOONGFORGE_PATH}/loongforge/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"
