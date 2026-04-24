#!/bin/bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/ssd1/deepseek-ai/data_v3/sampled.jsonl"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/ssd3/qwen/Qwen3-235B-A22B-tokenizer"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/ssd2/qwen/Qwen3_235B_A22B_mcore_tp4pp8ep16etp1"}
CHECKPOINT_SAVE_PATH=${CHECKPOINT_SAVE_PATH:-"/ssd3/qwen/Qwen3_235B_A22B_mcore_tp4pp8ep16etp1_save"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/ssd2/qwen/tensorboard-log/qwen3-235b-a22b"}

GPUS_PER_NODE=8

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    --model-name qwen3-235b-a22b
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --eod-mask-loss
    --data-path $DATA_PATH
    --split 99,1,0
)

SFT_ARGS=(
    --chat-template qwen
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    --packing-sft-data
)

TRAINING_ARGS=(
    --training-phase sft # options: pretrain, sft
    --seq-length 131072
    --max-position-embeddings 131072
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 128
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
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_SAVE_PATH
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 10
    --recompute-granularity full
    --recompute-method block
    --custom-pipeline-layers 11,11,12,12,12,12,12,12
    --custom-pipeline-recompute-layers 11,11,12,12,12,12,12,12
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 16
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type allgather
    --use-distributed-optimizer
    --distributed-backend nccl
    --sequence-parallel
    --optimizer-cpu-offload
    --use-precision-aware-optimizer
    --optimizer-offload-fraction 1.0
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
