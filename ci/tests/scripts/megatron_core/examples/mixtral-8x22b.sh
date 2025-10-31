#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH="/workspace/Megatron-LM"

# CKPT_SAVE_DIR="your model save ckpt path"
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/pfs/huggingface.co/mistralai/Mixtral-8x22B-v0.1"}
DATA_PATH=${DATA_PATH:-"/mnt/pfs/llama2/pile_llama_test/pile-llama_text_document"}
# CKPT_LOAD_DIR="your model ckpt path"

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

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --max-position-embeddings 32768
    --num-layers 56
    --hidden-size 6144
    --ffn-hidden-size 16384
    --num-attention-heads 48
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_PATH}
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --seq-length 1024
    --max-position-embeddings 32768
    --init-method-std 0.01
    --no-masked-softmax-fusion
    --micro-batch-size 1
    --global-batch-size 64
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --no-load-optim
    --no-load-rng
    # --load $CHECKPOINT_PATH
    # --save $CHECKPOINT_PATH
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    #--moe-grouped-gemm
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
)

torchrun ${DISTRIBUTED_ARGS[@]} $MEGATRON_PATH/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
