# Quick Start: VLA Model SFT Training on Kunlunxin P800

## Quick Start: VLA Model SFT Training

This document guides you through the quick start process for fine-tuning Vision-Language-Action (VLA) models using the LoongForge framework on P800.


## SFT Training Script

LoongForge currently provides SFT training example scripts for various models. After entering the container, you can find relevant scripts in the `examples_xpu/{model}/finetuning/` directory. Below is an example SFT training script for `PI 0.5`. Please refer to the comments for the purpose of each script section:

```bash
#!/usr/bin/env bash
# Pi05 sanity SFT launcher. This leverages the lightweight pi05 trainer
# (dummy data, single forward/backward) to verify the wiring inside the Omni
# framework. Adjust paths if your repo layout differs.

set -euo pipefail

# Paths
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}

export XMLIR_ENABLE_FAST_FC=true         # Used in torch.nn.linear.py (e.g., LinearWithActFunction)
export XMLIR_MATMUL_FAST_MODE=1          # Accumulation acceleration for XBLAS fully-connected (FC) computation under BF16 precision
export XMLIR_ENABLE_LINEAR_FC_FUSION=1   # Allows linear layers to bypass XBLAS FC fusion in certain scenarios (e.g., using addmm instead), default value is 1
export XMLIR_PARALLEL_SAVE_MEMORY=false  # When set to false, GPU memory usage increases but performance is improved; when set to true, memory usage decreases but performance drops
export XDNN_USE_FAST_GELU=true           # High-precision implementation of the GELU operator
export BKCL_FORCE_SYNC=1                 # Force CPU synchronization before communication (this will reduce performance)
export BKCL_TREE_THRESHOLD=0             # Set to 0 to disable the tree algorithm
export BKCL_ENABLE_XDR=1                 # Enable XDR (XPU Direct RDMA) and direct RDMA. Traffic will then flow directly from the XPU to the RDMA NIC, which is required for multi-node training.
export BKCL_RDMA_VERBS=1                 # Used in conjunction with BKCL_QPS_PER_CONNECTION; only required for Hygon servers currently
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4   # Subject to actual conditions; for multi-node training, configure according to the NIC connectivity of the server environment
export XTE_USE_MULTI_TENSOR_ADAMW=True   # Align the Adam optimizer with the GPU multi_tensor_adamw implementation

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

```

## Monitoring Logs

By default, the script outputs TensorBoard logs to the directory specified by `TENSORBOARD_PATH`. You can view training curves through TensorBoard.

Additionally, if wandb is installed, you can configure the `WANDB_API_KEY` to upload training metrics to wandb for online monitoring.