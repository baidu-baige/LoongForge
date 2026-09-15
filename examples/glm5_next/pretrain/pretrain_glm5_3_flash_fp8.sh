#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# This script is used for pre-training GLM-5.3-Flash in FP8 mixed precision.
# The recipe matches the released checkpoint format: e4m3 with 128x128
# block-wise scales (experts / shared experts / dense MLP / DSA projections).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-""}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/zai-org/GLM-5.3-Flash"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/loongforge-omni-ckpt/GLM/GLM-5.3-Flash-FP8-tp8pp5ep8etp1"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/GLM/save/GLM-5.3-Flash-FP8-tp8pp5ep8etp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/glm5.3_flash"}

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
  # The GLM-5.3-Flash architecture (KDA linear attention / KPool-DSA schedule,
  # MLA, sigmoid routing, four-stream mHC) is fully described by the model
  # config file; no extra --multi-latent-attention / --rotary-base style flags
  # are needed here.
  --config-file ${LOONGFORGE_PATH}/configs/models/glm5_next/glm5.3_flash.yaml
  --use-fp32-dtype-for-param-pattern expert_bias
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 99990,8,2
  --no-create-attention-mask-in-dataloader
)

TRAINING_ARGS=(
  --training-phase pretrain
  --seq-length 65536
  --max-position-embeddings 1048576
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
  --distributed-timeout-minutes 60
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  --fp8-format e4m3
  --fp8-recipe blockwise
  --fp8-param-gather
)

MOE_ARGS=(
  # Router topology (sigmoid scoring, top-8, scaling 2.5, expert bias) comes
  # from the model config; only infrastructure flags live here.
  --moe-router-load-balancing-type seq_aux_loss
  --moe-aux-loss-coeff 1e-3
  --moe-grouped-gemm
  --moe-router-dtype fp32
  --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
  # GLM-5.3-Flash has 45 decoder layers, so the pipeline world must divide 45.
  # TP8 PP5 EP8 ETP1 -> 40 GPUs with a uniform 9 layers per stage.
  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 5
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --moe-token-dispatcher-type alltoall
  # If the DeepEP extension is available, flex dispatch + DeepEP can replace
  # alltoall above for lower all-to-all latency at large expert counts.
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

# GLM-5.3-Flash ships 1 MTP (nextn) layer (num_nextn_predict_layers=1 in the
# released config); the model yaml already sets mtp_num_layers/mtp_loss_scaling_factor.
MTP_ARGS=(
  --mtp-num-layers 1
  --mtp-loss-scaling-factor 0.1
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
  ${MTP_ARGS[@]}
