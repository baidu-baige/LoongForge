#!/bin/bash

set -eo pipefail

# Automatically execute the dataset download script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "${SCRIPT_DIR}/download_datasets.sh" ]; then
    echo "Running data preparation script: ${SCRIPT_DIR}/download_datasets.sh"
    bash "${SCRIPT_DIR}/download_datasets.sh" "$@"
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9999}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Resume settings (optional)
if [ -z "${RESUME_STATE_FILE}" ]; then
    COMMON_YAML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs/common.yaml"
    TRAINING_LOG_PATH=""
    PFS_PATH=""
    if [ -f "${COMMON_YAML}" ]; then
        PFS_PATH=$(awk -F': *' '$1=="pfs_path" {print $2}' "${COMMON_YAML}")
        TRAINING_LOG_PATH=$(awk -F': *' '$1=="training_log_path" {print $2}' "${COMMON_YAML}")
    fi
    if [ -n "${PFS_PATH}" ] && [[ "${TRAINING_LOG_PATH}" == *"$pfs_path"* ]]; then
        TRAINING_LOG_PATH=${TRAINING_LOG_PATH//\$pfs_path/${PFS_PATH}}
    fi
    export RESUME_STATE_FILE="${TRAINING_LOG_PATH}/e2e_resume_state.json"
fi

export RESUME_POLICY=${RESUME_POLICY:-skip_completed}

node_nums=1
gpu_nums=8

# Mode 1: Run one/multiple/all models under tests/configs (space-separated; all models: model_names="")
# model_names="qwen3_14b" 
# model_names="deepseek_v2_lite llama3_8b qwen3_14b"  
model_names=""                                                   
optional_subdir=""
include_optional=false

# Mode 2: Run one/multiple/all models under tests/optional/ (space-separated; all models: model_names="NONE")
# model_names="deepseek_v2/deepseek_v2_lite"
# model_names="llama2/llama2_7b internvl2.5/internvl2.5_8b"
# model_names="NONE"                                               
# optional_subdir=""
# include_optional=true   

# Mode 3: Mixed run of multiple models from tests/configs and tests/optional/ (space-separated)
# model_names="qwen3_14b"
# extra_models="internvl2.5/internvl2.5_8b"                              
# optional_subdir=""
# include_optional=true 

# Mode 4: Run all models in a series under tests/optional/ (e.g., internvl2.5 series)
# model_names="NONE"  
# optional_subdir="internvl2.5"
# include_optional=true

# Mode 5: Run all models under tests/configs and optional_configs
# model_names=""                                                   
# optional_subdir=""
# include_optional=true

# Test Configuration
TIMEOUT=3600

# Accuracy and performance tolerance parameters
accuracy_relative_tolerance=0.02
performance_relative_tolerance=0.05
check_loss_only=true
chip="A800"
# auto_collect_baseline=true

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

if [ "${check_loss_only}" = true ]; then
    extra_param="${extra_param} --check_loss_only"
fi

if [ "${chip}" != "default" ]; then
    extra_param="${extra_param} --chip ${chip}"
fi

if [ "${auto_collect_baseline}" = true ]; then
    extra_param="${extra_param} --auto_collect_baseline"
fi

# extra_param=" $extra_param --dry_run"
extra_param=" $extra_param --training_type ${training_type}"

# Resume settings (optional)
if [ -n "${RESUME_STATE_FILE}" ]; then
    extra_param=" $extra_param --resume_state_file ${RESUME_STATE_FILE}"
fi
if [ -n "${RESUME_POLICY}" ]; then
    extra_param=" $extra_param --resume_policy ${RESUME_POLICY}"
fi

if [ "${KUBERNETES_SERVICE_HOST}" != "" ]; then
  mkdir -p /workspace/logs
fi

LOG_DIR="${TRAINING_LOG_PATH}"

# invoke the python test driver.  capture its exit code so we can decide
# whether the run completed normally and therefore the logs directory can be
# archived.  if the process fails or is killed, we leave the directory alone
# to allow resume on the next invocation.
python3 main.py ${extra_param}
ret=$?

# if everything finished successfully, rename the logs folder to include a
# timestamp.  this both preserves the results and clears the canonical
# "logs" location for the next fresh run.
if [ $ret -eq 0 ] && [ -d "${LOG_DIR}" ]; then
    ts=$(date '+%Y%m%d_%H%M%S')
    mv "${LOG_DIR}" "${LOG_DIR}_${ts}"
    echo "[resume] run completed, archived logs to ${LOG_DIR}_${ts}"
fi

exit $ret