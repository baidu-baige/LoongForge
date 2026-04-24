#!/bin/bash
# This script is used for SFT training GLM5 in FP8 mixed precision.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sampled.jsonl"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/GLM/GLM-5-FP8"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/GLM/GLM-5-FP8-tp8pp8ep8etp1"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/GLM/save/GLM-5-FP8-tp8pp8ep8etp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/glm5"}

export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

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
MASTER_PORT=${MASTER_PORT:-"6657"}
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
  --config-file ${LOONGFORGE_PATH}/configs/models/glm5/glm5.yaml
  --multi-latent-attention
  --rotary-base 1000000
  --norm-epsilon 1e-5
  --attention-backend fused
  --use-fp32-dtype-for-param-pattern expert_bias
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 90,8,2
  --no-create-attention-mask-in-dataloader
)

SFT_ARGS=(
  --chat-template no-template
  --sft-num-preprocess-workers 16
  --no-check-for-nan-in-loss-and-grad
  --packing-sft-data
  --use-fixed-seq-lengths
  --sft-dataset sharegpt
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length 65536
  --max-position-embeddings 163840
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 128
  --lr 1e-06
  --train-iters 1500
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --min-lr 1.0e-7
  --weight-decay 0.1
  --lr-warmup-fraction 0.002
  --clip-grad 1.0
  --bf16
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_PATH_SAVE
  --save-interval 1000
  --eval-interval 10
  --eval-iters 1

  --no-load-optim
  --no-load-rng
  --recompute-granularity full
  --recompute-method block
  --custom-pipeline-layers 10,10,10,10,10,10,10,8
  --custom-pipeline-recompute-layers 10,10,10,10,10,10,10,8
  --num-virtual-stages-per-pipeline-rank 2
  --fp8-format e4m3
  --fp8-recipe blockwise
  --fp8-param-gather
  --distributed-timeout-minutes 60
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  --use-dsa-fused
  --disable-rotate-activation
)

MOE_ARGS=(
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk 8
  --moe-aux-loss-coeff 1e-3
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0.001
  --moe-router-num-groups 8
  --moe-router-group-topk 4
  --moe-router-score-function sigmoid
  --moe-router-topk-scaling-factor 2.5
  --moe-router-dtype fp32
  --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 8
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

MTP_ARGS=(
  --mtp-num-layers 1
  --mtp-loss-scaling-factor 0.1
  --should-get-embedding-weights-for-mtp
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
  ${SFT_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  ${MTP_ARGS[@]}
