#!/bin/bash

set -eo pipefail

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9999}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1

node_nums=1
gpu_nums=8


# Model selection configuration - choose one of the following methods, comment out others
# Method 1: Run single or multiple models in configs/ directory
# model_names="deepseek_v2_lite"
# model_names="llama2_7b"
# model_names="llama3_8b"
# model_names="qwen3_14b"
# model_names="qwen2.5_vl_7b"
# model_names="llavaov_1.5_4b"
# model_names="internvl2.5_8b"
model_names="internvl3.5_30b_a3b"
# model_names="qwen2.5_vl_7b llama3_8b"  # Multiple models separated by spaces
# optional_subdir=""
include_optional=false

# Method 2: Run models in both configs/ and optional_configs/ simultaneously (mixed)
# model_names="internvl2.5_8b internvl2.5/internvl2.5_8b"
# optional_subdir=""
# include_optional=true


# Method 3: Run only all models in optional_configs/ under a specific subdirectory
# model_names="NONE"  
# extra_models=""
# optional_subdir="internvl2.5"
# include_optional=true

# Test configuration
TIMEOUT=3600

# Tolerance parameters
accuracy_relative_tolerance=0.02
performance_relative_tolerance=0.05

# check_correctness_task check_perfness_task check_precess_data_task
tasks="check_correctness_task"
# tasks="check_perfness_task"
# tasks="check_precess_data_task"

# pretrain sft
training_type="pretrain sft"

# Build parameters
extra_param="--node_nums ${node_nums} \
            --gpu_nums ${gpu_nums} \
            --accuracy_relative_tolerance ${accuracy_relative_tolerance} \
            --performance_relative_tolerance ${performance_relative_tolerance} \
            --tasks ${tasks} \
            --timeout ${TIMEOUT}"

# Add optional subdirectory parameter (will automatically enable include_optional)
if [ -n "${optional_subdir}" ]; then
    extra_param="${extra_param} --optional_subdir ${optional_subdir}"
    include_optional=true
fi

# Add optional config directory
if [ "${include_optional}" = true ]; then
    extra_param="${extra_param} --include_optional"
fi

# Add model parameters
if [ "${model_names}" = "NONE" ]; then
    # model_names="NONE" means not running models in configs/, only running optional models
    :
elif [ -n "${model_names}" ]; then
    extra_param="${extra_param} --models ${model_names}"
fi

# Add additional optional_configs/ models
if [ -n "${extra_models}" ]; then
    extra_param="${extra_param} --extra_models ${extra_models}"
fi

# extra_param=" $extra_param --dry_run"
extra_param=" $extra_param --training_type ${training_type}"

if [ "${KUBERNETES_SERVICE_HOST}" != "" ]; then
  mkdir -p /workspace/logs
fi

# List all available models (optional, for debugging)
# python3 main.py --list_available_models 
python3 main.py ${extra_param}