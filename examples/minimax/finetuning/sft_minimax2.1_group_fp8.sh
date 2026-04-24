#!/bin/bash

# This script is used for SFT training Minimax2.1 in FP8 mixed precision.



MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sft/think/sampled.jsonl"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co//MiniMax-M2.1"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/minimax_m2.1/MiniMax_mcore_tp8pp4ep8etp1/"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/minimax_m2"}

export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12


export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NVSHMEM_HCA_LIST=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3

export NCCL_NVLS_ENABLE=0
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

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


SFT_ARGS=(
  --chat-template minimax-m2
  --sft-num-preprocess-workers 16
  --no-check-for-nan-in-loss-and-grad
  --packing-sft-data
)

MODEL_ARGS=(
  --model-name minimax2.1-230b
  --rotary-percent 0.5
  --norm-epsilon 1e-6
  --rotary-base 5000000
  --use-fp32-dtype-for-param-pattern expert_bias
  --attention-backend fused
  
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 90,8,2
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length 196608
  --max-position-embeddings 196608  # not used
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 128
  --lr 1e-6
  --train-iters 1000
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --min-lr 1.0e-7
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
  --recompute-granularity full
  --recompute-method block
  --custom-pipeline-layers 16,16,16,14
  --custom-pipeline-recompute-layers 16,16,16,14
  --fp8-format e4m3
  --fp8-recipe blockwise
  --fp8-param-gather
)

MOE_ARGS=(
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk 8
  --moe-aux-loss-coeff 1e-3
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-score-function sigmoid
  --moe-router-dtype fp32
  --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 4
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --moe-token-dispatcher-type flex
  --moe-enable-deepep
  --use-precision-aware-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --use-distributed-optimizer
  --moe-permute-fusion

  --overlap-grad-reduce
  --overlap-param-gather
)


LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --check-weight-hash-across-dp-replicas-interval 30
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $LOONGFORGE_PATH/loongforge/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  ${SFT_ARGS[@]} 
