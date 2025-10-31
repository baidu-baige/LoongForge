#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH="/workspace/Megatron-LM"

# CKPT_SAVE_DIR="your model save ckpt path"
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/pfs/huggingface.co/Qwen/Qwen2-7B-Instruct"}
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
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 14336
   --num-attention-heads 32
   --swiglu
   --attention-dropout 0
   --hidden-dropout 0
   --disable-bias-linear
   --untie-embeddings-and-output-weights
   --num-query-groups 8
   --group-query-attention
   --normalization RMSNorm
   --position-embedding-type rope
#    --no-position-embedding
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
    --eod-mask-loss
    --data-path $DATA_PATH
    --split 99,1,0
)

TRAINING_ARGS=(
    --seq-length 4096
    --max-position-embeddings 32768
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 50000
    --lr-decay-iters 500000
    --lr-decay-style linear
    --lr-warmup-iters 500
    --initial-loss-scale 65536
    --bf16
    --accumulate-allreduce-grads-in-fp32
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 10
    # --load $CHECKPOINT_PATH
    # --save $CHECKPOINT_PATH
    #--ckpt-step 0
    #--no-load-optim
    #--no-load-rng
    #--num-workers 8
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
    #--sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1
)

torchrun ${DISTRIBUTED_ARGS[@]} $MEGATRON_PATH/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}