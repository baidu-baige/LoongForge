#!/bin/bash
set -e

#PFS_DIR="/mnt/cluster/loongforge"
PFS_DIR="/mnt/cfs/LoongForgeCI"
BCECMD_DIR="${PFS_DIR}/linux-bcecmd-0.5.1"
REAL_BCECMD="${BCECMD_DIR}/bcecmd"
BCECMD_CMD="${REAL_BCECMD} --conf-path ${BCECMD_DIR}"
DOWNLOAD_MODE="default" # default/optional

CHECK_SHARED_STORAGE_DIR="${PFS_DIR}/check_shared_storage"
CHECK_PORT=29999

checkpoint_dir=${PFS_DIR}/megatron_checkpoint
huggingface_dir=${PFS_DIR}/huggingface.co
datasets_dir=${PFS_DIR}/omni_datasets/LoongForge

mkdir -p $checkpoint_dir $huggingface_dir $datasets_dir
mkdir -p $PFS_DIR
mkdir -p $CHECK_SHARED_STORAGE_DIR

# 确保bcecmd配置存在
ensure_bcecmd_config() {
    local bcecmd_dir="${BCECMD_DIR}"

    if [[ ! -d "$bcecmd_dir" ]]; then
        echo "bcecmd配置不存在，开始下载和配置..."

        local ENV="bj"
        local DATASET_PREFIX="cce-ai-datasets"

        cd $PFS_DIR

        echo "下载bcecmd工具..."
        if ! wget "https://${DATASET_PREFIX}.${ENV}.bcebos.com/aihc-qa/aihc_cluster_test/linux-bcecmd-0.5.1.zip"; then
            echo "错误：下载bcecmd工具失败！"
            exit 1
        fi

        echo "解压bcecmd工具..."
        if ! unzip linux-bcecmd-0.5.1.zip; then
            echo "错误：解压bcecmd工具失败！"
            exit 1
        fi

        cd "$bcecmd_dir"

        echo "配置bcecmd凭据..."
        echo "[Defaults]" > credentials
        echo "Ak = xxx" >> credentials
        echo "Sk = xxx" >> credentials
        echo "Sts =" >> credentials

        echo "配置bcecmd参数..."
        echo "[Defaults]" > config
        echo "Domain = ${ENV}.bcebos.com" >> config
        echo "Region = ${ENV}" >> config
        echo "AutoSwitchDomain = yes" >> config
        echo "BreakpointFileExpiration =" >> config
        echo "Https =" >> config
        echo "MultiUploadThreadNum =" >> config
        echo "MultiDownloadThreadNum =" >> config
        echo "SyncProcessingNum =" >> config
        echo "MultiUploadPartSize =" >> config
        echo "ProxyHost =" >> config

        echo "bcecmd配置完成"
    else
        echo "bcecmd配置已存在"
    fi
}

ensure_bcecmd_config

# === 分布式训练检测 ===
detect_distributed_env() {
    if [[ "${WORLD_SIZE:-1}" -gt 1 ]]; then
        echo "检测到分布式训练: WORLD_SIZE=$WORLD_SIZE"
        if [[ "${RANK:-0}" == "0" ]]; then
            echo "当前节点: Master (RANK=0)"
            IS_MASTER=true
        else
            echo "当前节点: Worker (RANK=$RANK)"
            IS_MASTER=false
        fi
    else
        echo "检测到单机训练模式"
        IS_MASTER=true
    fi
}

detect_distributed_env

while [[ $# -gt 0 ]]; do
    case $1 in
        --download_mode)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                DOWNLOAD_MODE="$2"
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

        --time_flag)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                TIME_FLAG="$2"
                shift 2
            else
                TIME_FLAG="tmp_file"
                shift
            fi
            ;;

        *)
            echo "未知参数: $1"
            echo "支持的参数: --download_mode, --ak, --sk, --min_space, --time_flag"
            exit 1
            ;;
    esac
done

# === 检测共享存储 ===
detect_shared_storage() {
    local test_file
    test_file="$CHECK_SHARED_STORAGE_DIR/shared_storage_test_$TIME_FLAG.txt"
    echo "当前节点: RANK=$RANK, IS_MASTER=$IS_MASTER"

    # Master节点创建测试文件
    if [[ "$IS_MASTER" == "true" ]]; then
        echo "Master节点创建共享存储测试文件..."
        echo "shared_storage_test_$TIME_FLAG" > "$test_file"
        sync
        echo "测试文件已创建: $test_file"

        echo "Starting master HTTP service on port..."
        if command -v python3 >/dev/null 2>&1; then
            python3 -m http.server $CHECK_PORT &
            MASTER_PID=$!
        elif command -v python >/dev/null 2>&1; then
            python -m SimpleHTTPServer $CHECK_PORT &
            MASTER_PID=$!
        else
            echo "Python not found, cannot start HTTP service"
            exit 1
        fi

        echo "Master HTTP service started (PID: $MASTER_PID)"
    else
        # Worker节点：等待Master节点就绪
        echo "Worker节点等待Master节点..."
        local max_wait=36000
        local waited=0

        while [[ $waited -lt $max_wait ]]; do
            if timeout 5 bash -c "echo > /dev/tcp/$MASTER_ADDR/$CHECK_PORT" 2>/dev/null; then
                echo "Master节点就绪!"
                break
            fi
            echo "等待Master节点...(${waited}秒)"
            sleep 5
            waited=$((waited+5))
        done

        if [[ $waited -ge $max_wait ]]; then
            echo "错误：等待Master节点超时"
            SHARED_STORAGE=false
            export SHARED_STORAGE
            return 1
        fi
        sleep 5
    fi

    # 所有节点检查文件是否存在
    if [[ "$IS_MASTER" == "true" ]]; then
        SHARED_STORAGE=true
    else
        if [[ -f "$test_file" ]]; then
            echo "节点 $RANK 检测到共享存储: $test_file"
            SHARED_STORAGE=true

            # 记录检测到的文件内容（用于调试）
            echo "测试文件内容: $(cat "$test_file" 2>/dev/null || echo '无法读取')"
        else
            echo "节点 $RANK 未检测到共享存储"
            SHARED_STORAGE=false
        fi
    fi

    export SHARED_STORAGE
}

bcecmd() {
    local retries=20
    local count=0
    local wait_time=3
    local file_bos_dir="$1"
    local file_local_dir="$2"

    if [[ -z "$AK" || -z "$SK" ]]; then
        echo "AK或SK未提供"
        exit 1
    else
        current_ak="$AK"
        current_sk="$SK"
        echo "使用传入的AK、SK: $current_ak, $current_sk"
    fi

    local temp_config_dir
    temp_config_dir=$(mktemp -d "/tmp/bcecmd_XXXXXX")

    cp "$BCECMD_DIR/config" "$temp_config_dir/" 2>/dev/null || true
    echo "[Defaults]" > "$temp_config_dir/credentials"
    echo "Ak = $current_ak" >> "$temp_config_dir/credentials"
    echo "Sk = $current_sk" >> "$temp_config_dir/credentials"
    echo "Sts =" >> "$temp_config_dir/credentials"
    local temp_bos_cmd="$REAL_BCECMD --conf-path $temp_config_dir"

    if [[ -z "${SHARED_STORAGE:-}" ]]; then
        detect_shared_storage
    fi

    # 根据存储类型决定下载策略
    if [[ "$SHARED_STORAGE" == "true" ]]; then
        # 共享存储模式：只有master下载，worker等待
        if [[ "$IS_MASTER" == "true" ]]; then
            echo "共享存储模式: Master节点下载文件..."
            while [ $count -lt $retries ]; do
                # Temporarily disable set -e to handle errors manually
                set +e
                $temp_bos_cmd bos sync $file_bos_dir $file_local_dir
                local status=$?
                set -e

                if [ $status -eq 0 ]; then
                    return 0
                fi

                count=$((count + 1))
                echo "========================================================================"
                echo "[WARNING] bcecmd execution failed (Exit code: $status)."
                echo "Retrying in $wait_time seconds... (Attempt $count/$retries)"
                echo "========================================================================"
                sleep $wait_time
            done
            echo "[ERROR] bcecmd failed after $retries attempts."
            exit 1
        else
            echo "共享存储模式下Worker节点跳过下载..."
            return 0
        fi
    else
        echo "独立存储模式: 所有节点独立下载文件..."
        while [ $count -lt $retries ]; do
            # Temporarily disable set -e to handle errors manually
            set +e
            $temp_bos_cmd bos sync $file_bos_dir $file_local_dir
            local status=$?
            set -e

            if [ $status -eq 0 ]; then
                return 0
            fi

            count=$((count + 1))
            echo "========================================================================"
            echo "[WARNING] bcecmd execution failed (Exit code: $status)."
            echo "Retrying in $wait_time seconds... (Attempt $count/$retries)"
            echo "========================================================================"
            sleep $wait_time
        done
        echo "[ERROR] bcecmd failed after $retries attempts."
        exit 1
    fi
}

# 支持多个下载模式，用空格分隔
case " $DOWNLOAD_MODE " in
    *" default "*)
        echo "Running default dataset download..."

        # deepseek_v2_lite
        bcecmd bos:/ai-data/deepseek-ai/DeepSeek-V2-Lite ${huggingface_dir}/deepseek-ai/DeepSeek-V2-Lite
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/deepseek_v2_lite/pile_test/ ${datasets_dir}/deepseek_v2_lite/pile_test/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/deepseek_v2_lite/tp2pp1ep8  ${checkpoint_dir}/deepseek_v2_lite/

        # llama2_7b
        bcecmd bos:/ai-data/Llama-2-7b-hf-test ${huggingface_dir}/meta-llama/Llama-2-7b-hf
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/llama2/pile_test/ ${datasets_dir}/llama2/pile_test/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llama2_7b/tp1pp1  ${checkpoint_dir}/llama2_7b/

        # llama3_8b
        bcecmd bos:/ai-data/meta-llama/Meta-Llama-3-8B ${huggingface_dir}/meta-llama/Meta-Llama-3-8B
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llama3_8b/tp2pp2/  ${checkpoint_dir}/llama3_8b/

        # qwen3_14b
        bcecmd bos:/ai-data/Qwen/Qwen3-14B ${huggingface_dir}/Qwen/Qwen3-14B
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen3/pile_test/ ${datasets_dir}/qwen3/pile_test/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_14b/tp2pp4/  ${checkpoint_dir}/qwen3_14b/

        # qwen2.5_vl_7b
        bcecmd bos:/ai-data/Qwen/Qwen2.5-VL-7B-Instruct ${huggingface_dir}/Qwen/Qwen2.5-VL-7B-Instruct/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2.5_vl_7b/pretrain/llava_details-minigpt4_3500_formate/wds ${datasets_dir}/qwen2.5_vl_7b/pretrain/llava_details-minigpt4_3500_formate/wds
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2.5_vl_7b/sft/test_packing/test_packed_wds_4k/ ${datasets_dir}/qwen2.5_vl_7b/sft/test_packing/test_packed_wds_4k/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen2.5_vl_7b/tp4pp2vpp2/  ${checkpoint_dir}/qwen2.5_vl_7b/

        # llavaov_1.5_4b
        bcecmd bos:/ai-data/lmms-lab/LLaVA-OneVision-1.5-4B-stage0 ${huggingface_dir}/LLaVA-OneVision-1.5-4B-stage0/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/llavaov_1.5_4b/test_packed_wds_4k/ ${datasets_dir}/llavaov_1.5_4b/test_packed_wds_4k/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/llavaov_1.5_4b/tp2pp2/  ${checkpoint_dir}/llavaov_1.5_4b/

        # internvl2.5_8b
        bcecmd bos:/ai-data/InternVL2_5-8B ${huggingface_dir}/internvl/InternVL2_5-8B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_image ${datasets_dir}/internvl2.5_8b/webdataset_image
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_8b/tp4pp2vpp2/  ${checkpoint_dir}/internvl2.5_8b/

        # internvl3.5_30b_a3b
        bcecmd bos:/ai-data/OpenGVLab/InternVL3_5-30B-A3B ${huggingface_dir}/internvl/InternVL3_5-30B-A3B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_video ${datasets_dir}/internvl3.5_30b_a3b/webdataset_video
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_30b_a3b/tp2pp2ep4/  ${checkpoint_dir}/internvl3.5_30b_a3b/

        # offline packing
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/offline_packing/test_wds/ ${datasets_dir}/offline_packing_data/

    ;;
    *" optional "*)
        echo "Running optional dataset download..."

        # internvl2.5_8b
        bcecmd bos:/ai-data/InternVL2_5-8B ${huggingface_dir}/internvl/InternVL2_5-8B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/internvl/webdataset_image ${datasets_dir}/internvl2.5_8b/webdataset_image
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_8b/tp4pp1/  ${checkpoint_dir}/internvl2.5/internvl2.5_8b/

        # internvl2.5_26b
        bcecmd bos:/ai-data/OpenGVLab/InternVL2_5-26B ${huggingface_dir}/internvl/InternVL2_5-26B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_26b/tp4pp1/  ${checkpoint_dir}/internvl2.5/internvl2.5_26b/

        # internvl2.5_38b
        bcecmd bos:/ai-data/InternVL2_5-38B ${huggingface_dir}/internvl/InternVL2_5-38B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_38b/tp4pp2/  ${checkpoint_dir}/internvl2.5/internvl2.5_38b/

        # internvl2.5_78b
        bcecmd bos:/ai-data/OpenGVLab/InternVL2_5-78B ${huggingface_dir}/internvl/InternVL2_5-78B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl2.5_78b/tp8pp4/  ${checkpoint_dir}/internvl2.5/internvl2.5_78b/

        # internvl3.5_8b
        bcecmd bos:/ai-data/jhc/OpenGVLab/InternVL3_5-8B ${huggingface_dir}/internvl/InternVL3_5-8B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_8b/tp4pp1/  ${checkpoint_dir}/internvl3.5/internvl3.5_8b/

        # internvl3.5_14b
        bcecmd bos:/ai-data/OpenGVLab/InternVL3_5-14B ${huggingface_dir}/internvl/InternVL3_5-14B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_14b/tp4pp1/  ${checkpoint_dir}/internvl3.5/internvl3.5_14b/

        # internvl3.5_30b_a3b
        bcecmd bos:/ai-data/OpenGVLab/InternVL3_5-30B-A3B ${huggingface_dir}/internvl/InternVL3_5-30B-A3B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_30b_a3b/tp4pp1ep8etp1/  ${checkpoint_dir}/internvl3.5/internvl3.5_30b_a3b/

        # internvl3.5_38b
        bcecmd bos:/ai-data/OpenGVLab/InternVL3_5-38B ${huggingface_dir}/internvl/InternVL3_5-38B
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/internvl3.5_38b/tp4pp2/  ${checkpoint_dir}/internvl3.5/internvl3.5_38b/

        # qwen2_7b
        bcecmd bos:/ai-data/huggingface.co/Qwen/Qwen2-7B-Instruct ${huggingface_dir}/Qwen/Qwen2-7B-Instruct/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen2/pile_test/ ${datasets_dir}/qwen2/pile_test/

        # qwen2.5_7b
        bcecmd bos:/ai-data/Qwen2.5-7B-Instruct ${huggingface_dir}/Qwen/Qwen2.5-7B-Instruct/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen/pile_test/ ${datasets_dir}/qwen2.5/pile_test/

        # qwen2.5_vl_7b
        bcecmd bos:/ai-data/Qwen/Qwen2.5-VL-7B-Instruct ${huggingface_dir}/Qwen/Qwen2.5-VL-7B-Instruct/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen2.5_vl_7b/tp1pp1/  ${checkpoint_dir}/qwen2.5_vl/qwen2.5_vl_7b/

        # qwen3_8b
        bcecmd bos:/ai-data/Qwen/Qwen3-8B ${huggingface_dir}/Qwen/Qwen3-8B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_8b/tp1pp1/  ${checkpoint_dir}/qwen3/qwen3_8b/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen3/pile_test/ ${datasets_dir}/qwen3/pile_test/

        # qwen3_30b_a3b
        bcecmd bos:/ai-data/Qwen/Qwen3-30B-A3B ${huggingface_dir}/Qwen/Qwen3-30B-A3B/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_30b_a3b/tp2pp2ep4/  ${checkpoint_dir}/qwen3/qwen3_30b_a3b/

        # qwen3_vl_30b_a3b
        bcecmd bos:/ai-data/Qwen3-VL-30B-A3B-Instruct ${huggingface_dir}/Qwen/Qwen3-VL-30B-A3B-Instruct/
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/qwen_vl/qwen3vl_30b_a3b_data/ ${datasets_dir}/qwen3_vl/qwen3vl_30b_a3b_data
        bcecmd bos:/aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/megatron_checkpoint/qwen3_vl_30b_a3b/tp1pp2ep8/  ${checkpoint_dir}/qwen3_vl/qwen3_vl_30b_a3b/

    ;;
    *)
        echo "Unknown download mode: $DOWNLOAD_MODE"
        echo "Supported modes: default optional"
        exit 1
        ;;
esac

echo "========================================"
echo "测试环境部署完毕"
echo "========================================"
