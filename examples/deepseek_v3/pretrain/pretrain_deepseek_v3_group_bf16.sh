#!/bin/bash

# This script is used for pre-training Deepseek-v3 in BF16 mixed precision.

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/aiak-training-llm/deepseek3/pile_test/pile-deepseek_text_document"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V3"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/aiak-training-llm/deepseek3/DeepSeek-V3-bf16-tp8pp8ep32etp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/deepseek-v3"}

GPUS_PER_NODE=8

# Change for multinode config
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

MODEL_ARGS=(
    --model-name deepseek-v3
    --multi-latent-attention
    --enable-fa-within-mla
    --norm-epsilon 1e-6
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --training-phase pretrain
    --seq-length 32768
    --max-position-embeddings 32768
    --init-method-std 0.01
    --no-masked-softmax-fusion
    --micro-batch-size 1
    --global-batch-size 2048
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --no-load-optim
    --no-load-rng
    --recompute-granularity full
    --recompute-method block
    --custom-pipeline-layers 8,7,8,8,8,8,8,6
    --custom-pipeline-recompute-layers 8,7,8,8,8,8,8,6
)

MOE_ARGS=(
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-router-enable-expert-bias
    --moe-router-num-groups 4
    --moe-router-group-topk 2
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8 
    --pipeline-model-parallel-size 8 
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1 
    --sequence-parallel
    --moe-token-dispatcher-type allgather
    --use-distributed-optimizer
    --moe-permute-fusion
)

MTP_ARGS=(
    --num-nextn-predict-layers 1
    --mtp-loss-coef 0.1
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/aiak_training_omni/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${MTP_ARGS[@]}