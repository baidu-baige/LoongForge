#!/usr/bin/env bash
# Pi05 sanity SFT launcher. This leverages the lightweight pi05 trainer
# (dummy data, single forward/backward) to verify the wiring inside the Omni
# framework. Adjust paths if your repo layout differs.

set -euo pipefail

# Paths
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/vla/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/vla/AIAK-Training-Omni"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}


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
)

# Core training args — pi05 trainer only needs minimal Megatron flags
TRAINING_ARGS=(
    --use-megatron-fsdp
    --training-phase sft
    --micro-batch-size 1
    --global-batch-size 2
    --seq-length 1024
    --max-position-embeddings 1024
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format fsdp_dtensor
    --load /workspace/model_dcp/
    --no-load-optim
    --no-load-rng
    --no-strict-fsdp-dtensor-load
    --finetune
)

MODEL_CONFIG_ARGS=(
    --model-name pi05
)

LOGGING_ARGS=(
    --log-interval 1
)

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/aiak_training_omni/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
