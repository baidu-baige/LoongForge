#!/bin/bash

##########################################
# 精简版：仅保留AIAK-Training-Omni基础环境安装
# 去除所有外部库（AIAK-Megatron/TransformerEngine/BCCL等）依赖，后续需求说是用 patch 方式，暂不清楚怎么操作
##########################################
set -exuo pipefail

############################################################# 基础环境开始 ###########################################################

function install_pip_config() {
    # 仅保留pip源配置（如果不需要可注释）
    cp pip.conf /etc/pip.conf
    cp pip.conf /usr/pip.conf
    mkdir -p /root/.pip && cp pip.conf /root/.pip/pip.conf
    mkdir -p /etc/xdg/pip && cp pip.conf /etc/xdg/pip/pip.conf
}

function install_base_env() {
    local COMPILE_ENV
    read COMPILE_ENV<<< $*

    cd ${CURRENT_DIR}
    # 清理pip配置
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf

    # 安装基础依赖
    install_requirements

    # pip config（可选，不需要可注释）
    mkdir -p /tmp && chmod -R 777 /tmp
    install_pip_config

    # 设置系统源和时区（基础环境，保留）
    rm -rf /etc/apt/sources.list && cp /workspace/AIAK-Training-Omni/docker/sources.list /etc/apt/sources.list
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install bc tree pwgen nodejs iproute2 -y
}

function install_requirements() {
    # 移除镜像中的constraints，以便升级库
    source_file="/etc/pip/constraint.txt"
    if [ -f "$source_file" ]; then
        backup_file="${source_file%.*}_backup.${source_file##*.}"
        mv -v "$source_file" "$backup_file"
        > "$source_file"
    fi

    # 仅保留AIAK-Training-Omni自身的requirements
    requirements_file="/workspace/AIAK-Training-Omni/requirements.txt"
    if [[ "$COMPILE_ENV" == "p800" ]];then
        requirements_file="/workspace/AIAK-Training-Omni/requirements_xpu.txt"
        echo "alias ll='ls -alF'" > /etc/profile.d/alias.sh
        echo "source /etc/profile.d/alias.sh" >> ~/.bashrc
    fi

    # 安装基础pip包（仅保留必要的）
    pip install wandb -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
    pip install swanlab==0.6.1 -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com 
    
    # ======================== 核心修复：-e . 路径问题 ========================
    local original_pwd=$(pwd)
    cd /workspace/AIAK-Training-Omni/
    pip install -r ${requirements_file} -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com
    cd ${original_pwd}

    # 临时卸载nvidia-modelopt（保留，避免兼容性告警）
    echo "y" | pip uninstall nvidia-modelopt || true
}

# 注释/删除所有依赖外部库的函数（encrypt_code/TransformerEngine/BCCL等）
# function encrypt_code() { ... }
# function install_transformerEngine() { ... }
# function install_AIAK_ACCELERATOR() { ... }
# function install_bccl_env() { ... }
# function install_community_megatron() { ... }
# function install_aiak_training_llm() { ... }
# function download_megatron_binary_compile() { ... }
# function install_deepep() { ... }
# function upgrade_cudnn_9_16_0() { ... }
# function install_aiak_fp8_quant() { ... }
# function update_flash_attn3_import_path_for_te() { ... }
# function download_xpytorch() { ... }

function clear_unused_file() {
    # 仅清理AIAK-Training-Omni自身的无用文件，删除外部库清理逻辑
    rm -rf /tmp/* ~/.bash_history
    rm -rf /workspace/AIAK-Training-Omni/.git
    rm -rf /workspace/AIAK-Training-Omni/build.sh
    rm -rf /workspace/AIAK-Training-Omni/ci.yml
    rm -rf /workspace/AIAK-Training-Omni/docker/ci || true
}

function install_aihclite_jupyter() {
    # 简化jupyter配置（仅创建目录，避免依赖外部脚本）
    mkdir -p /root/.jupyter
    touch /root/.jupyter/enterpoint.sh
    chmod +x /root/.jupyter/enterpoint.sh
}

# ======================== 参数解析（精简版） ========================
COMPILE_ENV=$1
BINARY_REPLACE=$2
ENCRYPT_FLAG=$3
INSTALL_BCCL_FLAG=$4
INSTALL_BCCL_ADDR=$5
INSTALL_TransformerEngine_FLAG=$6
INSTALL_AIAK_ACCELERATOR_FLAG=$7
MEGATRON_TYPE=$8

echo "Received COMPILE_ENV: ${COMPILE_ENV}"

VERSION=""
TOKEN=""
if [ -n "${9:-}" ]; then
    VERSION=$9
fi
if [ -n "${10:-}" ]; then
    TOKEN=${10}
fi

CURRENT_DIR=$(cd `dirname $0`; pwd)
BCCL_VERSION=${BCCL_VERSION-"1.2.7.2"}
IREPO_TOKEN=${BCCL_IREPO_TOKEN-"1bebc022-2d71-41a7-896c-53b32131285f"}

# ======================== 核心流程（仅保留基础环境） ========================
# 跳过p800环境的xpytorch下载（依赖外部包）
# if [[ "$COMPILE_ENV" == "p800" ]];then ... fi

# 仅安装基础环境，跳过二进制替换/加密/BCCL/TransformerEngine等
if  [[ "$BINARY_REPLACE" == "false" ]];then
    install_base_env $COMPILE_ENV
fi

# 跳过加密（依赖AIAK-Megatron，目录不存在）
# if  [[ "$ENCRYPT_FLAG" == "true" ]];then encrypt_code; fi

# 跳过BCCL安装
# if  [[ "$INSTALL_BCCL_FLAG" == "true" ]];then install_bccl_env "$BCCL_DOWNLOAD_ADDR"; fi

# 跳过TransformerEngine安装
# if  [[ "$INSTALL_TransformerEngine_FLAG" == "true" ]];then install_transformerEngine; fi

# 跳过AIAK-ACCELERATOR安装
# if  [[ "$INSTALL_AIAK_ACCELERATOR_FLAG" == "true" ]];then install_AIAK_ACCELERATOR; fi

# 跳过Megatron安装（无论community/aiak）
# if  [[ "$MEGATRON_TYPE" == "community" ]];then install_community_megatron; else install_aiak_training_llm; fi

# 简化jupyter安装
install_aihclite_jupyter

# 清理无用文件
clear_unused_file

############################################################# 基础环境结束 ###########################################################