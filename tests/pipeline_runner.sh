#!/bin/bash

set -eo pipefail

# Initialize environment variables
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9999}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Initialize node parameters
node_nums=1
gpu_nums=8
chip="A800" #A800, H800, BZZ

# Set metric tolerances
accuracy_relative_tolerance=0.02
performance_relative_tolerance=0.05

# Set test tasks and training types
tasks="check_correctness_task check_precess_data_task"
training_type="pretrain sft"

# Set test mode
test_mode="developer_mode"
model_in_configs=""
model_in_optional_configs=""

# Other parameters
timeout=3600
AK="default"
SK="default"
skip_env=false # Whether to skip environment preparation

# Resume parameters (can be preset via env vars, or auto-read from common.yaml by this script)
check_loss_only=true
auto_collect_baseline=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                timeout="$2"
                shift 2
            else
                shift
            fi
            ;;

        --chip)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                chip="$2"
                shift 2
            else
                shift
            fi
            ;;

        --precision)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                accuracy_relative_tolerance="$2"
                shift 2
            else
                shift
            fi
            ;;

        --performance)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                performance_relative_tolerance="$2"
                shift 2
            else
                shift
            fi
            ;;

        --release_mode)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                test_mode="$2"
                shift 2
            else
                shift
            fi
            ;;

        --model_in_configs)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                model_in_configs="$2"
                shift 2
            else
                shift
            fi
            ;;

        --model_in_optional_configs)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                model_in_optional_configs="$2"
                shift 2
            else
                shift
            fi
            ;;

        --time_flag)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                TIME_FLAG="$2"
                shift 2
            else
                TIME_FLAG=`date "+%Y%m%d%H%M%S"`
                shift
            fi
            ;;

        --training_type)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                training_type="$2"
                shift 2
            else
                shift
            fi
            ;;

        --tasks)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                tasks="$2"
                shift 2
            else
                shift
            fi
            ;;

        --ak)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                AK="$2"
                shift 2
            else
                shift
            fi
            ;;

        --sk)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                SK="$2"
                shift 2
            else
                shift
            fi
            ;;

        --skip_env)
            skip_env=true
            shift
            ;;

        --check_loss_only)
            check_loss_only=true
            shift
            ;;

        --auto_collect_baseline)
            auto_collect_baseline=true
            shift
            ;;

        --resume_state_file)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                RESUME_STATE_FILE="$2"
                shift 2
            else
                shift
            fi
            ;;

        --resume_policy)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                RESUME_POLICY="$2"
                shift 2
            else
                shift
            fi
            ;;

        *)
            echo "Unknown argument: $1"
            echo "Supported arguments: --timeout --chip --precision --performance --release_mode --model_in_configs --model_in_optional_configs --time_flag --training_type --tasks --ak --sk --skip_env --check_loss_only --auto_collect_baseline --resume_state_file --resume_policy"
            exit 1
            ;;
    esac
done

# Auto-read Resume config from configs/common.yaml (only when not preset via env vars or arguments)
if [ -z "${RESUME_STATE_FILE}" ]; then
    SCRIPT_DIR_TMP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    COMMON_YAML="${SCRIPT_DIR_TMP}/configs/common.yaml"
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

# Set model parameters based on test mode
case "${test_mode}" in
    mode1|"")
        echo "Run all models under configs only"
        model_names=""
        include_optional=false
        optional_subdir=""
        extra_models=""
        download_mode="default"
        ;;
    mode2)
        echo "Run all models under configs and optional_configs"
        model_names=""
        include_optional=true
        optional_subdir=""
        extra_models=""
        download_mode="default optional"
        ;;
    *)
        echo "Using developer mode, custom test models"
        download_mode=""

        # Check if at least one model-related parameter is specified
        if [ -z "${model_in_configs}" ] && [ -z "${model_in_optional_configs}" ]; then
            echo "Error: developer mode requires at least --model_in_configs or --model_in_optional_configs"
            echo "Please specify a model to test, or use another test mode (mode1, mode2, mode3)"
            exit 1
        fi

        # Handle model selection under configs/
        if [ -n "${model_in_configs}" ]; then
            # Specify specific models, multiple models separated by spaces
            model_names="${model_in_configs}"
            download_mode+=" default"
        else
            model_names=""
        fi

        # Handle model selection under optional_configs/
        if [ -n "${model_in_optional_configs}" ]; then
            # Determine input type: path structure (char+/+char) or subdirectory name
            if [[ "${model_in_optional_configs}" =~ ^[^/]+/[^/]+$ ]]; then
                # If path structure (e.g., "internvl2.5/internvl2.5_8b"), assign to extra_models
                extra_models="${model_in_optional_configs}"
                optional_subdir=""
            else
                # If subdirectory name (e.g., "internvl2.5"), assign to optional_subdir
                optional_subdir="${model_in_optional_configs}"
                extra_models=""
            fi
            include_optional=true
            download_mode+=" optional"
        else
            include_optional=false
            optional_subdir=""
            extra_models=""
        fi

        # Remove extra leading spaces
        download_mode="${download_mode#"${download_mode%%[![:space:]]*}"}"
        ;;
esac

# Build download parameters
if [[ ! "$skip_env" == true ]]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  if [[ -f "${SCRIPT_DIR}/prepare_env.sh" ]]; then
      echo "Running environment preparation script: ${SCRIPT_DIR}/prepare_env.sh"

      # Build argument array
      args=("${SCRIPT_DIR}/prepare_env.sh")

      # Only add this parameter when download_mode is not empty
      if [[ -n "$download_mode" ]]; then
          args+=("--download_mode" "${download_mode}")
      else
          echo "Warning: no download mode specified, will skip data download or use default mode" >&2
          exit 1
      fi

      # Add optional parameters
      if [[ "$AK" != "default" && "$SK" != "default" ]]; then
          args+=("--ak" "$AK" "--sk" "$SK")
      fi

      echo "Argument array: ${args[@]}"

      # Safe execution
      if ! bash "${args[@]}"; then
          echo "Error: data download failed" >&2
          exit 1
      fi
  else
      echo "Warning: download script ${SCRIPT_DIR}/prepare_env.sh not found, skipping data download"
  fi
fi

# Build test parameters
# 1. Base required parameters
args=(
    "--node_nums" "${node_nums}"
    "--gpu_nums" "${gpu_nums}"
    "--timeout" "${timeout}"
    "--chip" "${chip}"
    "--accuracy_relative_tolerance" "${accuracy_relative_tolerance}"
    "--performance_relative_tolerance" "${performance_relative_tolerance}"
)

# 2. Handle tasks (nargs='+')
if [[ -n "${tasks}" ]]; then
    tasks=$(echo "${tasks}" | tr ',' ' ')
    args+=("--tasks" ${tasks})
fi

# 3. Handle training_type (nargs='+')
if [[ -n "${training_type}" ]]; then
    training_type=$(echo "${training_type}" | tr ',' ' ')
    args+=("--training_type" ${training_type})
fi

# 4. Handle models (nargs='+')
# Note: skip passing model_names when it is "NONE"
if [[ -n "${model_names}" && "${model_names}" != "NONE" ]]; then
    args+=("--models" ${model_names})
fi

# 5. Handle include_optional (action='store_true')
if [[ "${include_optional}" == "true" ]]; then
    args+=("--include_optional")
fi

# 6. Handle optional_subdir (type=str, default=None)
# Only pass when the variable is non-empty, to avoid empty string causing path logic errors in Python
if [[ -n "${optional_subdir}" ]]; then
    args+=("--optional_subdir" "${optional_subdir}")
fi

# 7. Handle extra_models (nargs='*')
if [[ -n "${extra_models}" ]]; then
    args+=("--extra_models" ${extra_models})
fi

# 8. Handle check_loss_only (action='store_true')
if [[ "${check_loss_only}" == "true" ]]; then
    args+=("--check_loss_only")
fi

# 9. Handle auto_collect_baseline (action='store_true')
if [[ "${auto_collect_baseline}" == "true" ]]; then
    args+=("--auto_collect_baseline")
fi

# 10. Handle resume_state_file
if [[ -n "${RESUME_STATE_FILE}" ]]; then
    args+=("--resume_state_file" "${RESUME_STATE_FILE}")
fi

# 11. Handle resume_policy
if [[ -n "${RESUME_POLICY}" ]]; then
    args+=("--resume_policy" "${RESUME_POLICY}")
fi

echo "Final execution arguments: ${args[@]}"

# 12. Safe execution
if [ "${KUBERNETES_SERVICE_HOST}" != "" ]; then
  mkdir -p /workspace/logs
fi

# List all available models (optional, for debugging)
# python3 main.py --list_available_models

LOG_DIR="${TRAINING_LOG_PATH:-}"

python3 /workspace/LoongForge/tests/main.py "${args[@]}"
ret=$?


exit $ret
