#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/mllm/demo/wds/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/GLM/GLM-5.2-FP8"}
PROCESSOR_PATH=${PROCESSOR_PATH:-"/mnt/cluster/huggingface.co/kimi_2_6"}

# The base checkpoint contains GLM-5.2 plus Kimi-K2.6 MoonViT.
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/GLM/GLM-5.2-Kimi-K2.6-ViT-base-hf"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/GLM/GLM-5.2-Kimi-K2.6-ViT-sft"}
HF_CHECKPOINT_PATH=${HF_CHECKPOINT_PATH:-"${CHECKPOINT_PATH_SAVE}-hf"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/glm5.2-kimi-k2.6-vit-sft"}

export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

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
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)

MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-"${LOONGFORGE_PATH}/configs/models/glm5.2_vit/glm5_2_kimi_k2_6_vit.yaml"}
MODEL_ARGS=(
  --config-file "$MODEL_CONFIG_PATH"
  --rotary-base 8000000
  --norm-epsilon 1e-5
  --use-fp32-dtype-for-param-pattern expert_bias
)

DATA_ARGS=(
  --task-encoder KimiTaskEncoder
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path "$TOKENIZER_PATH"
  --hf-processor-path "$PROCESSOR_PATH"
  --data-path "$DATA_PATH"
  --dataloader-type external
  --split 100,0,0
  --num-workers 16
  --chat-template glm5.2-hf
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length 2048
  --enable-discard-sample
  --max-position-embeddings 1048576
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 128
  --lr 2e-05
  --train-iters 50000
  --lr-decay-iters 50000
  --lr-decay-style cosine
  --min-lr 1e-6
  --weight-decay 0.01
  --lr-warmup-fraction 0.01
  --clip-grad 1.0
  --bf16
  --load "$CHECKPOINT_PATH"
  --save "$CHECKPOINT_PATH_SAVE"
  --save-hf true
  --save-hf-path "$HF_CHECKPOINT_PATH"
  --save-interval 1000
  --ckpt-format torch
  --dataloader-save "${CHECKPOINT_PATH_SAVE}/dataloader"
  --allow-missing-adapter-checkpoint
  --no-load-optim
  --no-load-rng
  --recompute-granularity full
  --recompute-method block
  --custom-pipeline-recompute-layers 10,8,12,8,12,8,12,8
  # Every VPP chunk must start on an IndexShare computing layer.
  --custom-virtual-pipeline-layers 6,4,4,4,8,4,4,4,4,4,8,4,4,4,8,4
  --num-virtual-stages-per-pipeline-rank 2
  --reduce-variable-seq-shape-p2p-comm
  --fp8-format e4m3
  --fp8-recipe blockwise
  --fp8-param-gather
  --distributed-timeout-minutes 60
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  --enable-experimental
  --use-dsa-fused
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
  --tensorboard-dir "$TENSORBOARD_PATH"
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --check-weight-hash-across-dp-replicas-interval 30
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
  torchrun "${DISTRIBUTED_ARGS[@]}" \
  "$LOONGFORGE_PATH/loongforge/train.py" \
  "${MODEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${LOGGING_ARGS[@]}" \
  "${MTP_ARGS[@]}"
