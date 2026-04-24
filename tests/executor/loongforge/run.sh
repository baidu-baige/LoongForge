#!/bin/bash
set -x
set -eo pipefail

############################################ Model training parameters ############################################

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
############################################ Model Training Parameters ############################################

MEGATRON_PATH=${megatron_path:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH}
TOKENIZER_PATH=${TOKENIZER_PATH}
CHECKPOINT_PATH=${CHECKPOINT_PATH}
TENSORBOARD_PATH=${TENSORBOARD_PATH}

GPUS_PER_NODE=8

# Change for multinode config
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

# Parse environment variables parameters, environment variables should be space-separated parameter strings
# Expand directly into array, bash will split by space
MODEL_ARGS=(
    ${MODEL_ARGS}
)

MODEL_CONFIG_ARGS=(
    ${MODEL_CONFIG_ARGS}
)

DATA_ARGS=(
    ${DATA_ARGS}
)

TRAINING_ARGS=(
    ${TRAINING_ARGS}
)

MODEL_PARALLEL_ARGS=(
    ${MODEL_PARALLEL_ARGS}
)

LOGGING_ARGS=(
    ${LOGGING_ARGS}
)

MOE_ARGS=(
    ${MOE_ARGS}
)

SFT_ARGS=(
    ${SFT_ARGS}
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME} 
    )
fi

############################################ Model Training Parameters ############################################

export PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH

command="torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${MOE_ARGS[@]}
    "
    
if [[ "${use_nccl}" == "false" ]]; then
    export LD_LIBRARY_PATH=${BCCL_PATH}:$LD_LIBRARY_PATH
fi

echo ""
echo "Execute command: $command"

if [[ "${dry_run}" != "true" ]]; then
    eval $command
fi

