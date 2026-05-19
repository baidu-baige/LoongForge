#!/usr/bin/env bash
# Adjust paths to match your environment.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUDA_DEVICE_MAX_CONNECTIONS=8

export USE_BF16_BUFFER=false #Dtensor not support
export EAGLE_LOCAL_PATH=/workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6
set -euo pipefail

# Paths - adjust these to your environment
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero_object_no_noops_1.0.0_lerobot_3.0/"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"$EAGLE_LOCAL_PATH/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/gr00tn1.6_torch/"}

CHECKPOINT_SAVE_PATH=${CHECKPOINT_SAVE_PATH:-"/workspace/ckpt_save/"}

# Distributed launch (defaults single node)
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6020"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

GPUS_PER_NODE=8


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

# Core training args
TRAINING_ARGS=(
    --ckpt-format torch
    --training-phase sft
    --micro-batch-size 16
    --global-batch-size 128
    --seq-length 1024
    --max-position-embeddings 1024
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    #--no-masked-softmax-fusion
    --lr 1.0e-4
    --min-lr 0.0
    --lr-decay-iters 20
    --lr-warmup-fraction 0.05
    --lr-decay-style cosine
    --weight-decay 1.0e-5
    --clip-grad 1.0
    --load $CHECKPOINT_PATH
    #--save $CHECKPOINT_SAVE_PATH
    --save-interval 50
    --train-iters 20
    --eval-iters 0
    --num-workers 16
    --seed 1234
    --data-parallel-sharding-strategy no_shard
    --bf16
    #--grad-reduce-in-bf16
    #--exp-avg-dtype bf16
    #--exp-avg-sq-dtype bf16
    #--main-grads-dtype bf16
    #--use-precision-aware-optimizer
    --finetune
    --no-load-optim
    --no-load-rng
    --no-gradient-accumulation-fusion
    --deterministic-mode
    # Optional: enable CUDA graph (per-microbatch)
    #--cuda-graph-impl local
    #--cuda-graph-scope per_microbatch
    #--cuda-graph-warmup-steps 3
    #--cuda-graph-pad-length 220          # required: pad seqs to fixed length for graph capture
    #--no-check-for-nan-in-loss-and-grad  # required when CUDA graph enabled
)

MODEL_CONFIG_ARGS=(
    --model-name groot_n1_6
    --config-file $AIAK_TRAINING_PATH/configs/models/groot/groot_n1_6.yaml
    --distributed-backend nccl
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-interval 1
    #--profile
    #--use-pytorch-profiler
    #--profile-step-start 5
    #--profile-step-end 6
    #--profile-ranks 0
    #--tensorboard-dir /workspace/profiling/
)

# Run training
PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}