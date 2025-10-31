#!/bin/bash

# This script is used for SFT training Deepseek-v3 in BF16 mixed precision.

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/AIAK-Training-Omni"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/aiak-training-llm/dataset/sft_aplaca_zh_data.json"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/aiak-training-llm/deepseek3/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/AIAK-Training-Omni/configs/sft_dataset_config.json"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V3"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/aiak-training-llm/deepseek3/DeepSeek-V3-bf16-tp8pp8ep32etp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/aiak-training-llm/tensorboard-log/deepseek-v3"}

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
    --model-name deepseek-v3
    --multi-latent-attention
    --enable-fa-within-mla
    )

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
)

SFT_ARGS=(
    --chat-template deepseek3
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    --packing-sft-data
)

TRAINING_ARGS=(
    --training-phase sft
    --seq-length 32768
    --max-position-embeddings 32768
    --init-method-std 0.01
    --no-masked-softmax-fusion
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-06
    --norm-epsilon 1e-6
    --train-iters 300
    --lr-decay-iters 5000
    --lr-decay-style cosine
    --min-lr 1.0e-7
    --weight-decay 0.1
    --lr-warmup-fraction 0.002
    --clip-grad 1.0
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 100
    --eval-interval 1000
    --eval-iters 30
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
    ${SFT_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MTP_ARGS[@]} \
    ${LOGGING_ARGS[@]}