#!/bin/bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sft_aplaca_zh_data.json"}

#DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/deepseek2/sft_aplaca_zh_tokenized"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/LoongForge/deepseek2/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/LoongForge/configs/data/sft_dataset_config.yaml"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V2-Lite"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/deepseek2/DeepSeek_V2_Lite_group_tp1pp1ep8/"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/deepseek-v2"}

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
    --model-name deepseek-v2-lite
    --enable-fa-within-mla
    --norm-epsilon 1e-6
    --rotary-scaling-factor 40
    --mscale 0.707
    --mscale-all-dim 0.707
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
)

SFT_ARGS=(
    --chat-template deepseek
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    #--is-tokenized-data
    #--packing-sft-data
    #--sft-data-streaming

    #--train-on-prompt
    #--eod-mask-loss

    #--sft-dataset-config $DATASET_CONFIG_PATH
    #--sft-dataset sft_aplaca_zh_data # defined in --sft-dataset-config, default: default
    #--data-cache-path $DATA_CACHE_PATH
)

TRAINING_ARGS=(
    --training-phase sft
    --seq-length 2048
    --max-position-embeddings 2048
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
    --lr-warmup-fraction 0.002
    --clip-grad 1.0
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --no-load-optim
    --no-load-rng
    --enable-experimental
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 6
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
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