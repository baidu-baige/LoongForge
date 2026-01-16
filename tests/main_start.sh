#!/bin/bash

set -eo pipefail

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9999}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1

node_nums=1
gpu_nums=8


# 模型选择配置 - 选择以下方式之一，注释掉其他方式
# 方式1: 运行 configs/ 目录下的单个或多个模型
# model_names="deepseek_v2_lite"
# model_names="llama2_7b"
# model_names="llama3_8b"
# model_names="qwen3_14b"
# model_names="qwen2.5_vl_7b"
# model_names="llavaov_1.5_4b"
# model_names="internvl2.5_8b"
model_names="internvl3.5_30b_a3b"
# model_names="qwen2.5_vl_7b llama3_8b"  # 多个模型用空格分隔
# optional_subdir=""
include_optional=false

# 方式2: 同时运行 configs/ 和 optional_configs/ 下的模型（混合）
# model_names="internvl2.5_8b internvl2.5/internvl2.5_8b"
# optional_subdir=""
# include_optional=true


# 方式3: 只运行 optional_configs/ 下某个子目录的所有模型
# model_names="NONE"  
# extra_models=""
# optional_subdir="internvl2.5"
# include_optional=true

# 测试配置
TIMEOUT=3600

# 容差参数
accuracy_relative_tolerance=0.02
performance_relative_tolerance=0.05

# check_correctness_task check_perfness_task check_precess_data_task
tasks="check_correctness_task"
# tasks="check_perfness_task"
# tasks="check_precess_data_task"

# pretrain sft
training_type="pretrain sft"

# 构建参数
extra_param="--node_nums ${node_nums} \
            --gpu_nums ${gpu_nums} \
            --accuracy_relative_tolerance ${accuracy_relative_tolerance} \
            --performance_relative_tolerance ${performance_relative_tolerance} \
            --tasks ${tasks} \
            --timeout ${TIMEOUT}"

# 添加可选子目录参数（会自动启用 include_optional）
if [ -n "${optional_subdir}" ]; then
    extra_param="${extra_param} --optional_subdir ${optional_subdir}"
    include_optional=true
fi

# 添加可选配置目录
if [ "${include_optional}" = true ]; then
    extra_param="${extra_param} --include_optional"
fi

# 添加模型参数
if [ "${model_names}" = "NONE" ]; then
    # model_names="NONE" 表示不运行 configs/ 下的模型，只运行 optional models
    :
elif [ -n "${model_names}" ]; then
    extra_param="${extra_param} --models ${model_names}"
fi

# 添加额外的 optional_configs 下的模型
if [ -n "${extra_models}" ]; then
    extra_param="${extra_param} --extra_models ${extra_models}"
fi

# extra_param=" $extra_param --dry_run"
extra_param=" $extra_param --training_type ${training_type}"

if [ "${KUBERNETES_SERVICE_HOST}" != "" ]; then
  mkdir -p /workspace/logs
fi

# 列出所有可用模型（可选，用于调试）
# python3 main.py --list_available_models 
python3 main.py ${extra_param}