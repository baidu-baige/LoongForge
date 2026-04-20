#!/bin/bash

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
echo "Using MEGATRON_PATH: ${MEGATRON_PATH}"
echo "Using LOONGFORGE_PATH: ${LOONGFORGE_PATH}"

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sb.jsonl"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/LoongForge/checkpoints/MiMo-7B-SFT-tokenizer"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/checkpoints/MiMo-7B-RL-tp1pp2"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"CHECKPOINT_PATH_SAVE"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"TENSORBOARD_PATH"}

echo "Using DATA_PATH: ${DATA_PATH}"
echo "Using TOKENIZER_PATH: ${TOKENIZER_PATH}"
echo "Using CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "Using CHECKPOINT_PATH_SAVE: ${CHECKPOINT_PATH_SAVE}"

export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONFAULTHANDLER=1

GPUS_PER_NODE=8

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3

export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_IB_TRAFFIC_CLASS=130

export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3

export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --model-name mimo
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 90,8,2
)

SFT_ARGS=(
  --chat-template no-template
  --sft-num-preprocess-workers 16
  --no-check-for-nan-in-loss-and-grad
  --packing-sft-data
  --sft-dataset sharegpt
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length 32768
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
  --rotary-base 640000
  --train-iters 5000
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --lr-warmup-fraction 0.002
  --initial-loss-scale 65536
  --bf16
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_PATH
  --save-interval 500
  --eval-interval 30
  --eval-iters 10
  --log-validation-ppl-to-tensorboard
  --distributed-timeout-minutes 360
  --enable-experimental
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
    --sequence-parallel
)

LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $LOONGFORGE_PATH/loongforge/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${SFT_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]}
