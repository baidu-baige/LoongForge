#!/bin/bash
set -eo pipefail

root_path=$(dirname "$(readlink -f "$0")")
export scripts_root_path=${root_path}
source ${scripts_root_path}/common/common.sh

# 如若使用其他机器使用，则需要配置kubeconfig和kubectl、kubectl_view_allocations_path目录地址
kubectl_path="/usr/local/bin"
kubectl_view_allocations_path="/usr/local/bin"

# "/ssd2/leoli/kubeconfigs/configs/hwl-cce.config"、"/ssd2/leoli/kubeconfigs/configs/bjtest_ztl-ai-1.24.config"
kubeconfig_path=${kubeconfig_path:-"/ssd2/leoli/kubeconfigs/configs/hwl-cce.config"}
# "baidu.com/a800_80g_cgpu"、"baidu.com/a100_80g_cgpu"、"baidu.com/a10_24g_cgpu"
gpu_resource=${gpu_resource:-"baidu.com/a800_80g_cgpu"}
gpu_nums="8"

export PATH=$PATH:${kubectl_path}:${kubectl_view_allocations_path}
export KUBECONFIG="$kubeconfig_path"
export GPU_RESOURCE="$gpu_resource"
export GPU_NUMS="$gpu_nums"

# K8S 集群方式存在于pfs 数据的根目录
export TRAIN_DATA_DIR="/mnt/pfs/leoli"
export IMAGE=${IMAGE:-"registry.baidubce.com/hac_test/aiak-transformer:dev_20240204_130244"}
# export SPECIFIC_PYTORCHJOB_COMMAND="sleep 1d" # 可用于单机调试
export TIMEOUT=${TIMEOUT:-"7200"}
export SCHEDULE_TIMEOUT=${SCHEDULE_TIMEOUT:-"600"}
export CHECK_PYTORCHJOB_TIMEOUT=${CHECK_PYTORCHJOB_TIMEOUT:-"86400"} # 分布式训练，默认1天
# export specific_model_name="llama-2-7b" # 运行指定模型
# export NAME_PREFIX_AGILE="a800-test-lijipeng-aiak-transformer"
export accuracy_relative_tolerance=${accuracy_relative_tolerance:-"0.02"}
export performance_relative_tolerance=${performance_relative_tolerance:-"0.05"}
# check_correctness_task、check_perfness_task
export tasks=${tasks:-"check_perfness_task"}
export use_nccl=${use_nccl:-"false"}
export training_type=${training_type:-"pretrain sft"}

# export BOS_SYNC_AIAK_TRANSFORMER_ADDR=${BOS_SYNC_AIAK_TRANSFORMER_ADDR:-""}

run_all_ipipe_case