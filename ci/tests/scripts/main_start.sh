#!/bin/bash

set -eo pipefail

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9999}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1

node_nums=1
gpu_nums=8
model_names="llama-2-7b"
TIMEOUT=3600
ckpt_loss_diff=0.02

# check_correctness_task check_perfness_task check_precess_data_task
tasks="check_perfness_task"

# pretrain sft
training_type="pretrain sft"

extra_param="--node_nums ${node_nums} \
            --gpu_nums ${gpu_nums} \
            --models ${model_names} \
            --ckpt_loss_diff ${ckpt_loss_diff} \
            --tasks ${tasks} \
            --timeout ${TIMEOUT}"

# extra_param=" $extra_param --dry_run"
extra_param=" $extra_param --training_type ${training_type}"

if [ "${KUBERNETES_SERVICE_HOST}" != "" ]; then
  mkdir -p /workspace/logs
fi
python3 main.py ${extra_param}