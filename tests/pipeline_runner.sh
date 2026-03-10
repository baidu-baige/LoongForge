#!/bin/bash

set -eo pipefail

# 初始化环境变量
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-9999}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 初始化节点参数
node_nums=1
gpu_nums=8
chip="A800" #A800, H800, BZZ

# 设置指标参数
accuracy_relative_tolerance=0.02
performance_relative_tolerance=0.05

# 设置测试任务和训练类型
tasks="check_correctness_task check_precess_data_task"
training_type="pretrain sft"

# 设置测试模式
test_mode="developer_mode"
model_in_configs=""
model_in_optional_configs=""

# 其他参数
timeout=3600
AK="default"
SK="default"
skip_env=false #是否跳过环境准备

# 解析参数
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

        *)
            echo "未知参数: $1"
            echo "支持的参数: --timeout --precision --performance --release_mode --model_in_configs --model_in_optional_configs --time_flag --training_type --tasks --ak --sk"
            exit 1
            ;;
    esac
done

# 根据测试模式设置模型参数
case "${test_mode}" in
    mode1|"")
        echo "只运行configs下的所有模型"
        model_names=""
        include_optional=false
        optional_subdir=""
        extra_models=""
        download_mode="default"
        ;;
    mode2)
        echo "运行configs、optional_configs下的所有模型"
        model_names=""
        include_optional=true
        optional_subdir=""
        extra_models=""
        download_mode="default optional"
        ;;
    *)
        echo "使用开发者模式，自定义测试模型"
        download_mode=""

        # 检查是否至少指定了一个模型相关参数
        if [ -z "${model_in_configs}" ] && [ -z "${model_in_optional_configs}" ]; then
            echo "错误：开发者模式必须至少指定 --model_in_configs 或 --model_in_optional_configs 参数"
            echo "请指定要测试的模型，或使用其他测试模式（mode1, mode2, mode3）"
            exit 1
        fi

        # 处理 configs/ 目录下的模型选择
        if [ -n "${model_in_configs}" ]; then
            # 指定具体模型，多个模型用空格分隔
            model_names="${model_in_configs}"
            download_mode+=" default"
        else
            model_names=""
        fi

        # 处理 optional_configs/ 目录下的模型选择
        if [ -n "${model_in_optional_configs}" ]; then
            # 判断输入类型：路径结构（字符+/+字符）还是子目录名
            if [[ "${model_in_optional_configs}" =~ ^[^/]+/[^/]+$ ]]; then
                # 如果是路径结构（如"internvl2.5/internvl2.5_8b"），赋值给extra_models
                extra_models="${model_in_optional_configs}"
                optional_subdir=""
            else
                # 如果是子目录名（如"internvl2.5"），赋值给optional_subdir
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

        # 去除多余的空格
        download_mode="${download_mode#"${download_mode%%[![:space:]]*}"}"
        ;;
esac

# 构建下载参数
if [[ ! "$skip_env" == true ]]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  if [[ -f "${SCRIPT_DIR}/prepare_env.sh" ]]; then
      echo "执行测试数据准备脚本: ${SCRIPT_DIR}/prepare_env.sh"

      # 构建参数数组
      args=("${SCRIPT_DIR}/prepare_env.sh")

      # 只有当download_mode不为空时才添加这个参数
      if [[ -n "$download_mode" ]]; then
          args+=("--download_mode" "${download_mode}")
      else
          echo "警告：未指定下载模式，将跳过数据下载或使用默认模式" >&2
          exit 1
      fi

      # 添加可选参数
      if [[ "$AK" != "default" && "$SK" != "default" ]]; then
          args+=("--ak" "$AK" "--sk" "$SK")
      fi

      echo "参数数组: ${args[@]}"

      # 安全执行
      if ! bash "${args[@]}"; then
          echo "错误：数据下载失败" >&2
          exit 1
      fi
  else
      echo "警告：未找到下载脚本 ${SCRIPT_DIR}/prepare_env.sh，跳过数据下载"
  fi
fi

# 构建测试参数
# 1. 基础必选参数
args=(
    "--node_nums" "${node_nums}"
    "--gpu_nums" "${gpu_nums}"
    "--timeout" "${timeout}"
    "--chip" "${chip}"
    "--accuracy_relative_tolerance" "${accuracy_relative_tolerance}"
    "--performance_relative_tolerance" "${performance_relative_tolerance}"
)

# 2. 处理 tasks (nargs='+')
if [[ -n "${tasks}" ]]; then
    tasks=$(echo "${tasks}" | tr ',' ' ')
    args+=("--tasks" ${tasks})
fi

# 3. 处理 training_type (nargs='+')
if [[ -n "${training_type}" ]]; then
    training_type=$(echo "${training_type}" | tr ',' ' ')
    args+=("--training_type" ${training_type})
fi

# 4. 处理 models (nargs='+')
# 注意：逻辑中 model_names 为 "NONE" 时不传递
if [[ -n "${model_names}" && "${model_names}" != "NONE" ]]; then
    args+=("--models" ${model_names})
fi

# 5. 处理 include_optional (action='store_true')
if [[ "${include_optional}" == "true" ]]; then
    args+=("--include_optional")
fi

# 6. 处理 optional_subdir (type=str, default=None)
# 只有当变量不为空时才传递，避免 Python 接收到 "" 导致路径逻辑错误
if [[ -n "${optional_subdir}" ]]; then
    args+=("--optional_subdir" "${optional_subdir}")
fi

# 7. 处理 extra_models (nargs='*')
if [[ -n "${extra_models}" ]]; then
    args+=("--extra_models" ${extra_models})
fi

echo "最终执行参数: ${args[@]}"

# 8. 安全执行
if [ "${KUBERNETES_SERVICE_HOST}" != "" ]; then
  mkdir -p /workspace/logs
fi

# List all available models (optional, for debugging)
# python3 main.py --list_available_models

python3 /workspace/OmniTraining/tests/main.py "${args[@]}"