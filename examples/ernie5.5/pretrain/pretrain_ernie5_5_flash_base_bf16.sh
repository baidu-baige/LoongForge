#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Pretrain ERNIE-5.5-Flash-Base with BF16 weights.
set -e
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MODEL_CONFIG_FILE="$LOONGFORGE_PATH/configs/models/ernie5_5/ernie5_5_flash_base.yaml"

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/ernie5.5/pretrain_text_document"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/baidu/ERNIE-5.5-Flash-Base"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$TOKENIZER_PATH"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/ernie5.5/pretrain"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/ernie5.5-pretrain"}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PP_SIZE=${PP_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"5000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

MODEL_CONFIG_ARGS=(
  --config-file "$MODEL_CONFIG_FILE"
  --rotary-base 8000000
  --norm-epsilon 1e-6
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path "$TOKENIZER_PATH"
  --split 100,0,0
)
if [[ -n "$DATA_PATH" ]]; then
  DATA_ARGS+=(--data-path "$DATA_PATH")
fi

TRAINING_ARGS=(
  --training-phase pretrain
  --bf16
  --seq-length "${SEQ_LENGTH:-2048}"
  --max-position-embeddings 1048576
  --micro-batch-size 1
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
  --train-iters "${TRAIN_ITERS:-1500}"
  --lr 1e-5
  --min-lr 1e-5
  --lr-decay-style constant
  --optimizer adam
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-8
  --weight-decay 0.1
  --clip-grad 0
  --seed 42
  --save "$CHECKPOINT_PATH_SAVE"
  --save-interval 1000
  --eval-iters 0
  --eval-interval 100000
  --no-gradient-accumulation-fusion
  --no-bias-dropout-fusion
  --no-bias-swiglu-fusion
  --no-masked-softmax-fusion
  --num-workers 0
)
if [[ -n "$CHECKPOINT_PATH" ]]; then
  TRAINING_ARGS+=(--load "$CHECKPOINT_PATH")
fi

MOE_ARGS=(
  --moe-grouped-gemm
  --moe-token-dispatcher-type alltoall
  --moe-router-dtype fp32
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size "$PP_SIZE"
  --expert-model-parallel-size "${EP_SIZE:-8}"
  --expert-tensor-parallel-size 1
  --use-distributed-optimizer
  --accumulate-allreduce-grads-in-fp32
)

MTP_ARGS=(
  --mtp-num-layers 1
  --mtp-loss-scaling-factor 0.1
)

LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir "$TENSORBOARD_PATH"
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
)

PYTHONPATH="$LOONGFORGE_PATH:$LOONGFORGE_PATH/tools:$MEGATRON_PATH:$PYTHONPATH" \
  torchrun "${DISTRIBUTED_ARGS[@]}" \
  "$LOONGFORGE_PATH/loongforge/train.py" \
  "${MODEL_CONFIG_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "$@"
