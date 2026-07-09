#! /bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Qwen-Image-Edit-2511 DiT training with Megatron FSDP + TP.
export NVTE_FUSED_ATTN=0
unset CUDA_DEVICE_MAX_CONNECTIONS
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/home/opt/cuda_tools/:$PATH
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:$LD_LIBRARY_PATH

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron/"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge/"}
DATASET_PATH=${DATASET_PATH:-"data/preprocessed_qwen_image_edit_2511/"}
MCORE_LOAD_CKPT=${MCORE_LOAD_CKPT:-"/ssd1/loongfore_data/Qwen/qwen_image_edit_2511_mcore"}
CKPT_SAVE_PATH=${CKPT_SAVE_PATH:-"/ssd1/loongfore_data/Qwen/qwen_image_edit_2511_mcore"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen_image_edit_2511/"}

GPUS_PER_NODE=${GPUS:-8}
TP_SIZE=${TP_SIZE:-1}
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

H=${H:-1024}
W=${W:-1024}
SEQ_LEN=$(((H / 16) * (W / 16) * 2))

MODEL_ARGS=(
    --model-name qwen-image-edit-2511
)

DATA_ARGS=(
    --tokenizer-type NullTokenizer
    --vocab-size 0
    --seed 42
    --data-path $DATASET_PATH
    --dataloader-type external
)

TRAINING_ARGS=(
    --training-phase pretrain
    --num-latent-frames 1
    --max-latent-height $H
    --max-latent-width $W
    --max-text-length 1024
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size ${GLOBAL_BATCH_SIZE:-8}
    --lr ${LR:-1e-5}
    --min-lr ${LR:-1e-5}
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --norm-epsilon 1e-06
    --train-iters ${TRAIN_ITERS:-50000}
    --eval-iters ${EVAL_ITERS:-0}
    --lr-decay-iters ${LR_DECAY_ITERS:-${TRAIN_ITERS:-50000}}
    --lr-decay-style constant
    --initial-loss-scale 65536
    --bf16
    --save-interval ${SAVE_INTERVAL:-500000}
    --no-load-optim
    --no-load-rng
    --no-strict-fsdp-dtensor-load
    --finetune
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers ${RECOMPUTE_NUM_LAYERS:-43}
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size 1
    --use-megatron-fsdp
    --data-parallel-sharding-strategy ${FSDP_SHARDING_STRATEGY:-optim_grads_params}
    --overlap-grad-reduce
    --overlap-param-gather
    --suggested-communication-unit-size 200000000
    --no-gradient-accumulation-fusion
    --ckpt-format fsdp_dtensor
    --use-precision-aware-optimizer
    --use-distributed-optimizer
    --distributed-backend nccl
    --attention-backend fused
)

LOAD_SAVE_ARGS=()
# Load pretrained weights via Megatron checkpoint loading (fsdp_dtensor DCP).
# Convert the HF DiT checkpoint first: ./convert_qwen_image.sh hg2mcore
if [ -n "$MCORE_LOAD_CKPT" ]; then
    LOAD_SAVE_ARGS+=(--load "$MCORE_LOAD_CKPT")
fi
if [ -n "$CKPT_SAVE_PATH" ]; then
    LOAD_SAVE_ARGS+=(--save "$CKPT_SAVE_PATH")
fi

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    $LOONGFORGE_PATH/loongforge/train.py \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${LOAD_SAVE_ARGS[@]}" \
    --max-timestep-boundary ${MAX_TIMESTEP_BOUNDARY:-1} \
    --min-timestep-boundary ${MIN_TIMESTEP_BOUNDARY:-0} \
    "${MODEL_PARALLEL_ARGS[@]}" \
    --log-interval 1 \
    --tensorboard-dir ${TENSORBOARD_PATH} \
    --log-timers-to-tensorboard
