#!/bin/bash

set -eo pipefail

# Automatically execute the dataset download script
AKSK_FILE="$1"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "${SCRIPT_DIR}/download_datasets.sh" ]; then
    echo "Running data preparation script: ${SCRIPT_DIR}/download_datasets.sh"
    bash "${SCRIPT_DIR}/download_datasets.sh" "$AKSK_FILE"
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9999}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1

node_nums=1
gpu_nums=8

# Mode 1: Run single or multiple models under the configs/ directory
# model_names="deepseek_v2_lite"                # Run single or multiple models under configs/ directory
# model_names="llama2_7b llama3_8b"             # Separate multiple models with spaces
# model_names=""                                  # Leave empty to run all models under configs/ by default
# optional_subdir=""
# include_optional=false

# Mode 2: Run models under both configs/ and optional_configs/ simultaneously (Mixed)
# model_names="internvl2.5_8b internvl2.5/internvl2.5_8b"
# optional_subdir=""
# include_optional=true     

# Mode 3: Run all models in a specific subdirectory under optional_configs/ (e.g., internvl2.5)
# model_names="NONE"  
# optional_subdir="internvl2.5"
# extra_models="internvl2.5/internvl2.5_8b"     # Run specific models under optional_configs/
# include_optional=true

# Mode 4: Run all models under optional_configs/
# model_names="NONE"
# optional_subdir=""
# include_optional=true

# Test Configuration
TIMEOUT=3600

# Accuracy and performance tolerance parameters
accuracy_relative_tolerance=0.02
performance_relative_tolerance=0.05

# Test tasks
tasks="check_correctness_task check_precess_data_task"

# pretrain sft
training_type="pretrain sft"

# Build parameters
extra_param="--node_nums ${node_nums} \
            --gpu_nums ${gpu_nums} \
            --accuracy_relative_tolerance ${accuracy_relative_tolerance} \
            --performance_relative_tolerance ${performance_relative_tolerance} \
            --tasks ${tasks} \
            --timeout ${TIMEOUT}"

# Add optional subdir parameter (will automatically enable include_optional)
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
    # model_names="NONE" means do not run models under configs/, only run optional models
    :
elif [ -n "${model_names}" ]; then
    extra_param="${extra_param} --models ${model_names}"
fi

# Add extra models under optional_configs
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