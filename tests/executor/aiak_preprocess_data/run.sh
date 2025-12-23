#!/bin/bash
set -x
set -eo pipefail

############################################ 预处理数据集参数 ############################################

MEGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/mnt/cluster/cyw/E2E2/AIAK-Training-Omni"}
PREPROCESS_DATA_PATH="${PREPROCESS_DATA_PATH}"

PREPROCESS_DATA_ARGS=(
    ${PREPROCESS_DATA_ARGS}
)

export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH

# 根据脚本类型选择执行方式
if [[ "${PREPROCESS_DATA_PATH}" == *.sh ]]; then
    # shell 脚本（如 offline_packing）
    command="bash $PREPROCESS_DATA_PATH ${PREPROCESS_DATA_ARGS[@]}"
else
    # python 脚本
    command="python $PREPROCESS_DATA_PATH ${PREPROCESS_DATA_ARGS[@]}"
fi

echo ""
echo "执行命令: $command"
if [[ "${dry_run}" != "true" ]]; then
    eval $command
fi