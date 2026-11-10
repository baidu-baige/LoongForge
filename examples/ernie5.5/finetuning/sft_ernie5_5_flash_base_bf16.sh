#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Fine-tune ERNIE-5.5-Flash-Base on OpenAI-format messages.
set -e
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MODEL_CONFIG_FILE="$LOONGFORGE_PATH/configs/models/ernie5_5/ernie5_5_flash_base.yaml"
DATASET_CONFIG_PATH="$LOONGFORGE_PATH/configs/data/sft_dataset_config.yaml"

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/ernie5.5/sft_data"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/baidu/ERNIE-5.5-Flash-Base"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"$TOKENIZER_PATH"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/ernie5.5/finetune"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/ernie5.5-sft"}
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-'{"enable_thinking": false}'}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PP_SIZE=${PP_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
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
  --data-path "$DATA_PATH"
  --split 100,0,0
)

SFT_ARGS=(
  --chat-template ernie5.5-hf
  --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"
  --sft-dataset openai
  --sft-dataset-config "$DATASET_CONFIG_PATH"
  --sft-num-preprocess-workers 16
)

TRAINING_ARGS=(
  --training-phase sft
  --bf16
  --seq-length "${SEQ_LENGTH:-2048}"
  --max-position-embeddings 1048576
  --micro-batch-size 1
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
  --train-iters "${TRAIN_ITERS:-100}"
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
  --load "$CHECKPOINT_PATH"
  --finetune
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
  "${SFT_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "$@"
