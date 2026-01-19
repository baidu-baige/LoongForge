#!/bin/bash

##########################################
# Simplified version: Only keep AIAK-Training-Omni basic environment installation
# Remove all external library dependencies (AIAK-Megatron/TransformerEngine/BCCL, etc.), subsequent requirement says to use patch method, operation method is unclear yet
##########################################
set -exuo pipefail

############################################################# Basic environment start ###########################################################

function install_pip_config() {
    # Only keep pip source configuration (can comment out if not needed)
    cp pip.conf /etc/pip.conf
    cp pip.conf /usr/pip.conf
    mkdir -p /root/.pip && cp pip.conf /root/.pip/pip.conf
    mkdir -p /etc/xdg/pip && cp pip.conf /etc/xdg/pip/pip.conf
}

function install_base_env() {
    local COMPILE_ENV
    read COMPILE_ENV<<< $*

    cd ${CURRENT_DIR}
    # Clean pip configuration
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf

    # Install basic dependencies
    install_requirements

    # pip config (optional, can comment out if not needed)
    mkdir -p /tmp && chmod -R 777 /tmp
    install_pip_config

    # Set system source and timezone (basic environment, keep)
    rm -rf /etc/apt/sources.list && cp /workspace/AIAK-Training-Omni/docker/sources.list /etc/apt/sources.list
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install bc tree pwgen nodejs iproute2 -y
}

function install_requirements() {
    # Remove constraints from image to upgrade libraries
    source_file="/etc/pip/constraint.txt"
    if [ -f "$source_file" ]; then
        backup_file="${source_file%.*}_backup.${source_file##*.}"
        mv -v "$source_file" "$backup_file"
        > "$source_file"
    fi

    # Only keep AIAK-Training-Omni's own requirements
    requirements_file="/workspace/AIAK-Training-Omni/requirements.txt"
    if [[ "$COMPILE_ENV" == "p800" ]];then
        requirements_file="/workspace/AIAK-Training-Omni/requirements_xpu.txt"
        echo "alias ll='ls -alF'" > /etc/profile.d/alias.sh
        echo "source /etc/profile.d/alias.sh" >> ~/.bashrc
    fi

    # Install basic pip packages (only keep necessary ones)
    pip install wandb -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
    pip install swanlab==0.6.1 -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com 
    
    # ======================== Core fix: -e . path issue ========================
    local original_pwd=$(pwd)
    cd /workspace/AIAK-Training-Omni/
    pip install -r ${requirements_file} -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com
    cd ${original_pwd}

    # Temporarily uninstall nvidia-modelopt (keep, avoid compatibility warnings)
    echo "y" | pip uninstall nvidia-modelopt || true
}

# Comment/delete all functions that depend on external libraries (encrypt_code/TransformerEngine/BCCL, etc.)
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
    # Only clean up unused files from AIAK-Training-Omni, remove external library cleanup logic
    rm -rf /tmp/* ~/.bash_history
    rm -rf /workspace/AIAK-Training-Omni/.git
    rm -rf /workspace/AIAK-Training-Omni/build.sh
    rm -rf /workspace/AIAK-Training-Omni/ci.yml
    rm -rf /workspace/AIAK-Training-Omni/docker/ci || true
}

function install_aihclite_jupyter() {
    # Simplify jupyter configuration (only create directory, avoid external script dependencies)
    mkdir -p /root/.jupyter
    touch /root/.jupyter/enterpoint.sh
    chmod +x /root/.jupyter/enterpoint.sh
}

# ======================== Parameter parsing (simplified version) ========================
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

# ======================== Core process (only keep basic environment) ========================
# Skip p800 environment xpytorch download (depends on external packages)
# if [[ "$COMPILE_ENV" == "p800" ]];then ... fi

# Only install basic environment, skip binary replacement/encryption/BCCL/TransformerEngine etc.
if  [[ "$BINARY_REPLACE" == "false" ]];then
    install_base_env $COMPILE_ENV
fi

# Skip encryption (depends on AIAK-Megatron, directory does not exist)
# if  [[ "$ENCRYPT_FLAG" == "true" ]];then encrypt_code; fi

# Skip BCCL installation
# if  [[ "$INSTALL_BCCL_FLAG" == "true" ]];then install_bccl_env "$BCCL_DOWNLOAD_ADDR"; fi

# Skip TransformerEngine installation
# if  [[ "$INSTALL_TransformerEngine_FLAG" == "true" ]];then install_transformerEngine; fi

# Skip AIAK-ACCELERATOR installation
# if  [[ "$INSTALL_AIAK_ACCELERATOR_FLAG" == "true" ]];then install_AIAK_ACCELERATOR; fi

# Skip Megatron installation (whether community/aiak)
# if  [[ "$MEGATRON_TYPE" == "community" ]];then install_community_megatron; else install_aiak_training_llm; fi

# Simplify jupyter installation
install_aihclite_jupyter

# Clean up unused files
clear_unused_file

############################################################# Basic environment end ###########################################################