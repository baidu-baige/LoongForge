#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"$(cd "${SCRIPT_DIR}/../../.." && pwd)"}
MEGATRON_PATH=${MEGATRON_PATH:-"${LOONGFORGE_PATH}/third_party/Loong-Megatron"}

DATA_PATH=${DATA_PATH:?Set DATA_PATH to the multimodal SFT dataset}
PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the model checkpoint}
TOKENIZER_PATH=${TOKENIZER_PATH:-"${PRETRAINED_CHECKPOINT}"}
SAVE_PATH=${SAVE_PATH:?Set SAVE_PATH to the output checkpoint directory}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"${SAVE_PATH}/tensorboard"}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
CLIP_GRAD=${CLIP_GRAD:-1.0}
OPTIMIZER_BACKEND=${OPTIMIZER_BACKEND:-torch-fused}
TRAIN_ITERS=${TRAIN_ITERS:-1000}
EVAL_ITERS=${EVAL_ITERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

TRAINING_ARGS=(
    --config-file "${LOONGFORGE_PATH}/configs/models/minicpm_v_4_6/minicpm_v_4_6_lora.yaml"
    --training-phase sft
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "${TOKENIZER_PATH}"
    --data-path "${DATA_PATH}"
    --dataloader-type external
    --task-encoder MiniCPMV46TaskEncoder
    --split 100,0,0
    --num-workers "${NUM_WORKERS}"
    --chat-template minicpm-v-4.6-hf
    --chat-template-kwargs '{"add_generation_prompt":false}'
    --sft-dataset-config "${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml"
    --sft-dataset openai
    --seq-length 2048
    --max-position-embeddings 262144
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --lr 1e-4
    --min-lr 0.0
    --clip-grad "${CLIP_GRAD}"
    --weight-decay 0.1
    --optimizer adam
    --optimizer-backend "${OPTIMIZER_BACKEND}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --train-iters "${TRAIN_ITERS}"
    --eval-iters "${EVAL_ITERS}"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.02
    --bf16
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
    --no-load-optim
    --no-load-rng
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --distributed-backend nccl
    --use-distributed-optimizer
    --save "${SAVE_PATH}"
    --save-interval "${SAVE_INTERVAL}"
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
)

if [ "${MTP_NUM_LAYERS}" -gt 0 ]; then
    TRAINING_ARGS+=(--mtp-num-layers "${MTP_NUM_LAYERS}")
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${LOONGFORGE_PATH}/loongforge/train.py" \
    "${TRAINING_ARGS[@]}"
