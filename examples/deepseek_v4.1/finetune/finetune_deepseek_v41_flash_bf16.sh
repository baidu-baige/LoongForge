#!/bin/bash
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# SFT training for the complete DeepSeek-V4.1-Flash main model.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/deepseek_v41/sft_data"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/deepseek_v41/DeepSeek-V4.1-bf16-tp8pp1ep8etp1"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/deepseek_v41/save/DeepSeek-V4.1-bf16-tp8pp1ep8etp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/deepseek-v4.1-sft"}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-${WORLD_SIZE:-"1"}}
NODE_RANK=${NODE_RANK:-${RANK:-"0"}}

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

MODEL_ARGS=(
  --model-name deepseek-v4.1-flash
  --multi-latent-attention
  --rotary-base 10000
  --original-max-position-embeddings 65536
  --norm-epsilon 1e-20
  --rotary-scaling-factor 16
  --qk-layernorm
)

DATA_ARGS=(
  --task-encoder deepseekv41taskencoder
  --sample-type multi_mix_qa
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 90,8,2
  --no-create-attention-mask-in-dataloader
)

SFT_ARGS=(
  --chat-template deepseek4.1
)

TRAINING_ARGS=(
  --training-phase sft
  --seq-length ${SEQ_LENGTH:-32768}
  --max-position-embeddings 1048576
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size ${GLOBAL_BATCH_SIZE:-128}
  --lr 1e-06
  --train-iters ${TRAIN_ITERS:-100}
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --min-lr 1.0e-7
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-8
  --lr-warmup-fraction 0.002
  --clip-grad 1.0
  --bf16
  --use-precision-aware-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_PATH_SAVE
  --init-model-with-meta-device
  --save-interval 100
  --eval-interval 30
  --eval-iters 10
  --no-load-optim
  --no-load-rng
  --distributed-timeout-minutes 60
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl linear
  --enable-experimental
  --use-fp32-dtype-for-param-pattern expert_bias mlp_hyper_connection attn_hc ffn_hc \
    sinks hc_head position_bias q_a_norm \
    kv_norm post_attention_layernorm norm
)

MOE_ARGS=(
  --moe-router-topk 6
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0
  --moe-router-topk-scaling-factor 1.5
  --moe-router-score-function sqrtsoftplus
  --moe-router-dtype fp32
  --moe-token-dispatcher-type alltoall
  --empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size ${TP_SIZE:-8}
  --pipeline-model-parallel-size ${PP_SIZE:-1}
  --num-virtual-stages-per-pipeline-rank ${VPP_SIZE:-1}
  --context-parallel-size 1
  --expert-model-parallel-size ${EP_SIZE:-8}
  --expert-tensor-parallel-size ${ETP_SIZE:-1}

  --use-distributed-optimizer
  --moe-permute-fusion
)

LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --check-weight-hash-across-dp-replicas-interval 30
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $LOONGFORGE_PATH/loongforge/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${SFT_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]}
