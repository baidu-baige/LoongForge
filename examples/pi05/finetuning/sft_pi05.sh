#!/usr/bin/env bash

set -euo pipefail

# Paths
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/pi05/"}

export CUDA_DEVICE_MAX_CONNECTIONS=8 # mfsdp require CUDA_DEVICE_MAX_CONNECTIONS != 1
export USE_BF16_BUFFER=false #Dtensor not support
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # avoid OOM

# Distributed launch (defaults single node)
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
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

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty
  --num-workers 16
)

# Core training args — pi05 trainer only needs minimal Megatron flags
TRAINING_ARGS=(
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim
    --training-phase sft
    --micro-batch-size 16
    --global-batch-size 128
    --train-iters 30000
    --seq-length 762
    --max-position-embeddings 762
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format fsdp_dtensor
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
    --seed 1234
    --lr 2.5e-8
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --lr-decay-iters 30000
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-eps 1e-8
    --adam-beta2 0.95
    --weight-decay 0.01
    --no-strict-fsdp-dtensor-load
    --finetune
    --bf16
    --grad-reduce-in-bf16
    --use-precision-aware-optimizer
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
    --main-grads-dtype bf16
    --num-distributed-optimizer-instances 1
    --save $CHECKPOINT_PATH
    --save-interval 30000
)

MODEL_CONFIG_ARGS=(
    --model-name pi05
    --use-distributed-optimizer
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    # --record-memory-history
    # --memory-snapshot-path ${TENSORBOARD_PATH}/memory_snapshots.pickle
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
