#!/bin/bash
# This script is used for pre-training Kimi K2.5 multimodal model.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/deepseek3/pile_test/pile-deepseek_text_document"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/kimi_2_5"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/kimi_2_5/kimi_k2_5-tp8pp8ep32etp1"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/mnt/cluster/LoongForge/kimi_2_5/save/kimi_k2_5-tp8pp8ep32etp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/kimi_k2_5"}

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

MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/kimi_k2.5/kimi_k2_5.yaml
MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

DATA_ARGS=(
    --task-encoder KimiTaskEncoder
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --add-question-in-pretrain
    --enable-discard-sample
    --num-workers 16
)

TRAINING_ARGS=(
  --training-phase pretrain
  --seq-length 2048
  --max-position-embeddings 163840
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 128
  --lr 1e-06
  --train-iters 50
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --min-lr 1.0e-7
  --weight-decay 0.1
  --lr-warmup-fraction 0.002
  --clip-grad 1.0
  --bf16
  --load $CHECKPOINT_PATH
  #--save $CHECKPOINT_PATH_SAVE
  --save-interval 100

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
  --distributed-timeout-minutes 60
  --enable-experimental
)

MOE_ARGS=(
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk 8
  --moe-aux-loss-coeff 1e-3
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-pre-softmax
  --moe-router-bias-update-rate 0.001
  --moe-router-num-groups 1
  --moe-router-group-topk 1
  --moe-router-score-function sigmoid
  --moe-router-topk-scaling-factor 2.827
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

LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  #--check-weight-hash-across-dp-replicas-interval 30
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $LOONGFORGE_PATH/loongforge/train.py \
  ${MODEL_CONFIG_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]}