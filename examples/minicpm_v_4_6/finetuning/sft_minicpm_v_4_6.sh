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
TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT:-"${LOONGFORGE_PATH}/loongforge/train.py"}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-0}
CLIP_GRAD=${CLIP_GRAD:-1.0}
TRAIN_ITERS=${TRAIN_ITERS:-1000}
EVAL_ITERS=${EVAL_ITERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-262144}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
MIN_LR=${MIN_LR:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
ADAM_EPS=${ADAM_EPS:-1e-8}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.05}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}
USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER:-1}
SEED=${SEED:-1234}

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

if [ -n "${LR_WARMUP_ITERS}" ]; then
    LR_WARMUP_ARGS=(--lr-warmup-iters "${LR_WARMUP_ITERS}")
else
    LR_WARMUP_ARGS=(--lr-warmup-fraction "${LR_WARMUP_FRACTION}")
fi

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
    --num-workers "${NUM_WORKERS}"
    --chat-template minicpm-v-4.6-hf
    --chat-template-kwargs '{"add_generation_prompt":false}'
    --sft-dataset-config "${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml"
    --sft-dataset openai
)

TRAINING_ARGS=(
    --training-phase sft
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --lr "${LEARNING_RATE}"
    --min-lr "${MIN_LR}"
    --clip-grad "${CLIP_GRAD}"
    --weight-decay "${WEIGHT_DECAY}"
    --optimizer adam
    --adam-beta1 "${ADAM_BETA1}"
    --adam-beta2 "${ADAM_BETA2}"
    --adam-eps "${ADAM_EPS}"
    --train-iters "${TRAIN_ITERS}"
    --eval-iters "${EVAL_ITERS}"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr-decay-style "${LR_DECAY_STYLE}"
    "${LR_WARMUP_ARGS[@]}"
    --bf16
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
    --no-load-optim
    --no-load-rng
    --seed "${SEED}"
)

MODEL_PARALLEL_ARGS=(
    --attention-backend "${ATTENTION_BACKEND}"
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --save "${SAVE_PATH}"
    --save-interval "${SAVE_INTERVAL}"
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
)

if [ "${USE_DISTRIBUTED_OPTIMIZER}" -eq 1 ]; then
    MODEL_PARALLEL_ARGS+=(--use-distributed-optimizer)
fi

if [ "${MTP_NUM_LAYERS}" -gt 0 ]; then
    TRAINING_ARGS+=(--mtp-num-layers "${MTP_NUM_LAYERS}")
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PYTHONPATH="${MEGATRON_PATH}:${LOONGFORGE_PATH}:${PYTHONPATH:-}" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${TRAIN_ENTRYPOINT}" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"
