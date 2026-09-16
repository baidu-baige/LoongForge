#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sft_aplaca_zh_data.json"}
DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Kimi-K3"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/kimi_k3/kimi_k3-llm-tp8pp8ep32etp1"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/kimi_k3/save/sft/kimi_k3-llm-tp8pp8ep32etp1"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/kimi_k3_llm_sft"}

GPUS_PER_NODE=8

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NVSHMEM_HCA_LIST=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3

export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# KDA uses the Triton fallback when TileLang/TVM is unavailable.
export FLA_TILELANG=${FLA_TILELANG:-0}

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"5000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-${LOONGFORGE_PATH}/configs/models/kimi_k3/kimi_k3_backbone.yaml}
MODEL_CONFIG_ARGS=(
  --config-file $MODEL_CONFIG_FILE
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template kimi-k3-hf
  --sft-dataset openai
  --sft-dataset-config $DATASET_CONFIG_PATH
  --sft-num-preprocess-workers 16
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length 32768
  --max-position-embeddings 1048576
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 128
  --lr 1e-6
  --train-iters 1500
  --lr-decay-iters 1500
  --lr-decay-style cosine
  --min-lr 1e-7
  --weight-decay 0.1
  --lr-warmup-fraction 0.002
  --clip-grad 1.0
  --bf16
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_PATH_SAVE
  --save-interval 100
  --eval-interval 30
  --eval-iters 0
  --no-load-optim
  --no-load-rng
  --recompute-granularity full
  --recompute-method block
  --distributed-timeout-minutes 60
  --enable-experimental
)

MOE_ARGS=(
  --moe-router-topk 16
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-pre-softmax
  --moe-router-bias-update-rate 0.0
  --moe-router-num-groups 1
  --moe-router-group-topk 1
  --moe-router-score-function sigmoid
  --moe-router-dtype fp32
  --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 8
  --context-parallel-size 1
  --expert-model-parallel-size 32
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --moe-token-dispatcher-type alltoall
  --use-precision-aware-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --use-distributed-optimizer
  --moe-permute-fusion
  --cross-entropy-loss-fusion
  --overlap-grad-reduce
  --overlap-param-gather
)

LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
)

if [ -n "${WANDB_API_KEY:-}" ]; then
  LOGGING_ARGS+=(
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${WANDB_NAME}
  )
fi

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  -m loongforge train --engine mcore \
  ${MODEL_CONFIG_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  model_type=kimi_k3_llm
