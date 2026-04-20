#!/usr/bin/env bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_COMPILE=0
export TORCHDYNAMO_DISABLE=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/datasets/filter_CC3M/"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/models/Qwen3.5-2B"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/qwen3.5-2b-tp1pp1/"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen3.5-2b-tp1pp1"}

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
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/qwen3.5/qwen3_5_2b.yaml

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 32
    --chat-template qwen2-vl
    --sft-dataset-config ${LOONGFORGE_PATH}/configs/data/sft_dataset_config.yaml
    # for packing
    # --packing-sft-data
    # --packing-buffer-size 5000
    # --max-packed-tokens 32768 
)

TRAINING_ARGS=(
    --training-phase sft # options: pretrain, sft
    --seed 42
    --seq-length 4096
    --max-position-embeddings 262144
    --rotary-percent 0.25
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
    --attention-softmax-in-fp32
    --calculate-per-token-loss
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    # --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
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

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
