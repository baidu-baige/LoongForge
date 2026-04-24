#!/bin/bash
# This script is used for pre-training Deepseek-v3 in FP8 mixed precision.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/deepseek3/pile_test/pile-deepseek_text_document"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/deepseek-ai/DeepSeek-V3"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/deepseek3/DeepSeek-V3-tp8pp8ep32etp1"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/deepseek3/save/DeepSeek-V3-tp8pp8ep32etp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/deepseek-v3"}

export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

GPUS_PER_NODE=8

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NVSHMEM_HCA_LIST=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
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

MODEL_ARGS=(
  --model-name deepseek-v3
  --multi-latent-attention
  --rotary-base 10000
  --original-max-position-embeddings 4096
  --mscale 1.0
  --mscale-all-dim 1.0
  --norm-epsilon 1e-6
  --rotary-scaling-factor 40
  --enable-fa-within-mla
  --use-fp32-dtype-for-param-pattern '^expert_bias$' '.+\.expert_bias$'
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
  --seq-length 32768
  --max-position-embeddings 163840
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 1024
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
  --save-interval 100
  --eval-interval 10
  --eval-iters 1

  --no-load-optim
  --no-load-rng
  --recompute-granularity full
  --recompute-method block
  --custom-pipeline-layers 8,7,8,8,8,8,8,6
  --custom-pipeline-recompute-layers 8,7,8,8,8,8,8,6
  --num-virtual-stages-per-pipeline-rank 2
  --reduce-variable-seq-shape-p2p-comm
  --fp8-format e4m3
  --fp8-recipe blockwise
  --fp8-param-gather
  --enable-fp8-comm
  --distributed-timeout-minutes 60
  --enable-experimental
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
  --expert-model-parallel-size 32
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --moe-token-dispatcher-type flex
  --moe-enable-deepep
  --use-precision-aware-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --use-distributed-optimizer
  --moe-permute-fusion
  --cross-entropy-loss-fusion
  --overlap-grad-reduce
  --overlap-param-gather
)

MTP_ARGS=(
  --mtp-loss-scaling-factor 0.1
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
  ${MTP_ARGS[@]}