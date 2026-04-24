#!/usr/bin/env bash
# Adjust paths to match your environment.
export CUDA_DEVICE_MAX_CONNECTIONS=8 # mfsdp require CUDA_DEVICE_MAX_CONNECTIONS != 1
export USE_BF16_BUFFER=false #Dtensor not support
export EAGLE_LOCAL_PATH=/workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6
set -euo pipefail

# Paths - adjust these to your environment
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/single_data/single_data/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$EAGLE_LOCAL_PATH/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt"}
CHECKPOINT_SAVE_PATH=${CHECKPOINT_SAVE_PATH:-"/workspace/test_save/"}

# Distributed launch (defaults single node)
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6020"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

GPUS_PER_NODE=1


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
    --use-megatron-fsdp
    --ckpt-format fsdp_dtensor
    --training-phase sft
    --micro-batch-size 1
    --global-batch-size 1
    --seq-length 1024
    --max-position-embeddings 1024
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --lr 1.0e-4
    --min-lr 0.0
    --lr-decay-iters 100
    --lr-warmup-fraction 0.05
    --lr-decay-style cosine
    --weight-decay 1.0e-5
    --clip-grad 1.0
    --load $CHECKPOINT_PATH
    #--save $CHECKPOINT_SAVE_PATH
    --save-interval 50
    --train-iters 100
    --eval-iters 0
    --num-workers 0
    --seed 1234
    #--data-parallel-sharding-strategy  optim
    --data-parallel-sharding-strategy no_shard
    --bf16
    --use-distributed-optimizer
    #--grad-reduce-in-bf16
    #--exp-avg-dtype bf16
    #--exp-avg-sq-dtype bf16
    #--main-grads-dtype bf16
    #--use-precision-aware-optimizer
    --finetune
    --no-load-optim
    --no-load-rng
    --no-gradient-accumulation-fusion
)

MODEL_CONFIG_ARGS=(
    --model-name groot_n1_6
    --config-file $LOONGFORGE_PATH/configs/models/groot/groot_n1_6.yaml
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
)

# Run training
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${LOAD_ARGS[@]}