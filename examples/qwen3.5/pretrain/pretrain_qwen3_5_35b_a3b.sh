#!/usr/bin/env bash
# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE=0
export TORCHDYNAMO_DISABLE=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Baige-Megatron"}
BAIGE_OMNI_PATH=${BAIGE_OMNI_PATH:-"/workspace/BaigeOmnni"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/BaigeOmni/datasets/filter_CC3M/"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/models/Qwen3.5-35B-A3B"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/BaigeOmni/qwen3.5-35b-a3b-tp1pp2ep4/"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/BaigeOmni/tensorboard-log/qwen3.5-35b-a3b-tp1pp2ep4"}

GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# To specify the model config file
MODEL_CONFIG_PATH=${BAIGE_OMNI_PATH}/configs/models/qwen3.5/qwen3_5_35b_a3b.yaml

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --enable-discard-sample
    --num-workers 32
    --rotary-percent 0.25
    --calculate-per-token-loss
)

TRAINING_ARGS=(
    --training-phase pretrain # options: pretrain, sft
    --seed 42
    --seq-length 4096
    --max-position-embeddings 262144
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 64
    --lr 1e-5
    --min-lr 1e-6
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 1000
    --lr-decay-iters 1000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.02
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 10000000
    --ckpt-format torch
    --no-save-optim
    --no-load-optim
    --no-load-rng

    --mtp-num-layers 1
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-permute-fusion
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --attention-softmax-in-fp32
    --moe-token-dispatcher-type alltoall
    --moe-shared-expert-intermediate-size 512
    --moe-shared-expert-overlap
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 4
    # --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME} 
    )
fi

PYTHONPATH=$MEGATRON_PATH:$BAIGE_OMNI_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $BAIGE_OMNI_PATH/baige_omni/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}