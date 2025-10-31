#!/bin/bash
set -x
set -eo pipefail

############################################ 预处理数据集参数 ############################################

MEGATRON_PATH=${megatron_path:-"/workspace/AIAK-Megatron"}
AIAK_TRAINING_PATH=${aiak_training_path:-"/workspace/AIAK-Training-Omni"}
PREPROCESS_DATA_PATH="${PREPROCESS_DATA_PATH}"

PREPROCESS_DATA_ARGS=(
    ${PREPROCESS_DATA_ARGS}
)

export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH
command="python $PREPROCESS_DATA_PATH \
    ${PREPROCESS_DATA_ARGS[@]}"

echo ""
echo "执行命令: $command"
if [[ "${dry_run}" != "true" ]]; then
    eval $command
fi