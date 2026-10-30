#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

# WAN2.1 I2V training with Megatron FSDP ZeRO-3.

# CUDA_DEVICE_MAX_CONNECTIONS must be UNSET for FSDP (not set to 1)
unset CUDA_DEVICE_MAX_CONNECTIONS
export NVTE_FUSED_ATTN=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export PYTORCH_CUDNN_V8_API_DISABLED=${PYTORCH_CUDNN_V8_API_DISABLED:-1}
export PATH=/home/opt/cuda_tools/:$PATH
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:$LD_LIBRARY_PATH

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/wan2.1/Loong-Megatron/"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/wan2.1/LoongForge/"}
DATASET_PATH=${DATASET_PATH:-"./data/preprocessed_wan2.1/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/wan2.1/hg2mcore/i2v_14b_480p/Megatron_Release/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/wan2.1/"}

GPUS_PER_NODE=${GPUS:-8}

# Distributed launch configuration.
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6007"}
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
    --model-name wan2-1-i2v
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 0
    --seed 42
    --data-path $DATASET_PATH
    --dataloader-type external
)

F=81
H=480
W=832
F_POST=$((($F - 1) / 4 + 1))           # VAE temporal compress (WAN = 4)
H_POST=$(($H / 8 / 2))                  # VAE spatial(8) + patch_h(2)
W_POST=$(($W / 8 / 2))                  # VAE spatial(8) + patch_w(2)
SEQ_LEN=$(($F_POST * $H_POST * $W_POST))

# Activation recompute. Wan2.1 14B has 40 DiT layers; block mode recomputes the first N layers.
RECOMPUTE_MODE=${RECOMPUTE_MODE:-"full"}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-"40"}

if [ "${RECOMPUTE_MODE}" = "selective" ]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity selective
    )
elif [ "${RECOMPUTE_MODE}" = "none" ]; then
    RECOMPUTE_ARGS=()
else
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method block
        --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}
    )
fi

TRAINING_ARGS=(
    --training-phase pretrain
    --num-latent-frames $F
    --max-latent-height $H
    --max-latent-width $W
    --max-text-length 512
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 8
    --lr 1e-5
    --min-lr 1e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --norm-epsilon 1e-06
    --train-iters ${TRAIN_ITERS:-50000}
    --lr-decay-iters ${LR_DECAY_ITERS:-50000}
    --lr-decay-style constant
    --initial-loss-scale 65536
    --bf16
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --save-interval 500000
    --no-load-optim
    --no-load-rng
    --no-strict-fsdp-dtensor-load
    --finetune
    ${RECOMPUTE_ARGS[@]}
)

# FSDP parallel configuration. Pipeline and context parallelism stay disabled.
MODEL_PARALLEL_ARGS=(
    --context-parallel-size ${CP_SIZE:-1}
    --context-parallel-ulysses-degree ${CP_ULYSSES_DEGREE:-1}
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --no-gradient-accumulation-fusion
    --ckpt-format fsdp_dtensor
    --use-precision-aware-optimizer
    --use-distributed-optimizer
    --distributed-backend nccl
    --attention-backend ${ATTENTION_BACKEND:-fused}
    --suggested-communication-unit-size 200000000
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME}
    )
fi

LOAD_SAVE_ARGS=(
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
)

TIMESTEP_BOUNDARY=(
    --max-timestep-boundary 1
    --min-timestep-boundary 0
)

# Train the single DiT model of wan2.1 I2V.
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOAD_SAVE_ARGS[@]} \
    ${TIMESTEP_BOUNDARY[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
