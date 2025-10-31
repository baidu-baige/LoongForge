#!/bin/bash
set -x
set -eo pipefail

############################################ 模型训练参数 ############################################

MEGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/workspace/AIAK-Training-Omni"}
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

# or you can setup llama2-7b by using the following command
MODEL_ARGS=(
    ${MODEL_ARGS}
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

if [[ "${model_name}" =~ "mixtral" ]]; then
    MOE_ARGS=(
        ${MOE_ARGS}
    )
fi

if [[ "${model_name}" =~ "cogvlm2" ]]; then
    IMG_ARGS=(
        ${IMG_ARGS}
    )
fi

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME} 
    )
fi

if [[ "${TRANING_MODEL}" != "" ]] && [[ "${TRANING_MODEL}" == "sft" ]]; then
    SFT_ARGS=(
        ${SFT_ARGS}
    )
fi

############################################ 模型训练参数 ############################################

export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH

command="torchrun ${DISTRIBUTED_ARGS[@]} \
    $AIAK_TRAINING_PATH/aiak_training_omni/train.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}"

if [[ "${model_name}" =~ "mixtral" ]]; then
    command="${command} ${MOE_ARGS[@]}"
fi

if [[ "${model_name}" =~ "cogvlm2" ]]; then
    command="${command} ${IMG_ARGS[@]}"
fi

if [[ "${use_nccl}" == "false" ]]; then
    export LD_LIBRARY_PATH=${BCCL_PATH}:$LD_LIBRARY_PATH
fi

echo ""
echo "执行命令: $command"

if [[ "${dry_run}" != "true" ]]; then
    eval $command
fi