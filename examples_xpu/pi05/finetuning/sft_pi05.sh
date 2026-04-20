#!/usr/bin/env bash
# Pi05 sanity SFT launcher. This leverages the lightweight pi05 trainer
# (dummy data, single forward/backward) to verify the wiring inside the Omni
# framework. Adjust paths if your repo layout differs.

set -euo pipefail

# Paths
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}

export XMLIR_ENABLE_FAST_FC=true         # torch.nn.linear.py 中用到(LinearWithActFunction 等)
export XMLIR_MATMUL_FAST_MODE=1          # 在bf16下xblas fc 计算累加加速
export XMLIR_ENABLE_LINEAR_FC_FUSION=1   # 允许某些场景下linear不走xblas fcfusion, 比如走addmm，默认为1 
export XMLIR_PARALLEL_SAVE_MEMORY=false  # 为false显存占用会多, 但会有性能提升; 为true显存会少, 但性能会下降
export XDNN_USE_FAST_GELU=true           # gelu 算子高精度实现
export BKCL_FORCE_SYNC=1                 # 通信前，在 CPU 强制同步，会降低性能
export BKCL_TREE_THRESHOLD=0             # 设置为 0，即关闭 tree 算法
export BKCL_ENABLE_XDR=1                 # 开启 XDR（xpu direct RDMA），使能 direct rdma，此时流量会直接从 XPU 到 RDMA网卡，多机训练需要开启。
export BKCL_RDMA_VERBS=1                 # 与 BKCL_QPS_PER_CONNECTION 配合使用，当前只用于海光机器才需要
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4   # 以实际情况为准，多机要按机器环境网卡联通性来配
export XTE_USE_MULTI_TENSOR_ADAMW=True   # 优化器adam对齐GPU multi_tensor_adamw实现

# Distributed launch (defaults single node)
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
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

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty
  --num-workers 16
)

# Core training args — pi05 trainer only needs minimal Megatron flags
TRAINING_ARGS=(
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim
    --training-phase sft
    --micro-batch-size 16
    --global-batch-size 128
    --train-iters 30000
    --seq-length 762
    --max-position-embeddings 762
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format fsdp_dtensor
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
    --seed 1234
    --lr 2.5e-8
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --lr-decay-iters 30000
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-eps 1e-8
    --adam-beta2 0.95
    --weight-decay 0.01
    --no-strict-fsdp-dtensor-load
    --finetune
    --bf16
    --grad-reduce-in-bf16
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --num-distributed-optimizer-instances 1
    --save $CHECKPOINT_PATH
    --save-interval 30000
)

MODEL_CONFIG_ARGS=(
    --model-name pi05
    --use-distributed-optimizer
    --distributed-backend nccl
    #--random-fallback-cpu
)

LOGGING_ARGS=(
    --log-interval 1
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
