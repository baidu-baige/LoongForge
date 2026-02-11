#!/bin/bash
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
    # 添加 --no-cache-dir 和 -q 避免进度条线程问题
    pip install -q --no-cache-dir wandb -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
    pip install -q --no-cache-dir swanlab==0.6.1 -i http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com 
    
    # ======================== 核心修复：-e . 路径问题 ========================
    local original_pwd=$(pwd)
    cd /workspace/AIAK-Training-Omni/
    # 添加 -q --no-cache-dir 避免进度条线程问题
    # 添加 --root-user-action=ignore 避免 root 用户警告
    pip install -q --no-cache-dir --root-user-action=ignore -r ${requirements_file} -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com
    cd ${original_pwd}

    # COMPILE_ENV: p800, hzz, b系列
    if [[ "$COMPILE_ENV" != "p800" ]];then
        # 当前流水只有h卡
        if [[ "$COMPILE_ENV" == "hzz" ]];then
            # 安装 FA3，适配 TE
            update_flash_attn3_import_path_for_te
            # 安装 Deepep
            install_deepep
            # 安装 AIAK FP8量化算子，用于支持权重转换
            install_aiak_fp8_quant
            # 升级 cudnn 到 9.16.0
            upgrade_cudnn_9_16_0
        elif [[ "$COMPILE_ENV" == "bzz" ]];then
            # 安装驱动（目前占位）
            echo "B卡对应安装的东西"
        # 后续如果要继续增加
        # elif [[ "$COMPILE_ENV" == "xxxx" ]];then  
        #     # 新增xxxx分支
        #     echo "新增xxxx分支"
        else
        # 默认分支：处理未预期的环境值，非0退出提示错误
            echo "[ERROR] 不支持的COMPILE_ENV值: $COMPILE_ENV"
            exit 1
        fi
    fi

    # 临时卸载nvidia-modelopt（保留，避免兼容性告警）
    echo "y" | pip uninstall nvidia-modelopt || true
}

function update_flash_attn3_import_path_for_te() {
    # 兼容 te 对 fa 算子的调用方式
    python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    flash_attn_3_path=$python_path/flash_attn_3/
    mkdir -p $flash_attn_3_path

    if [ -f "$python_path/flash_attn_interface.py" ]; then
        cp $python_path/flash_attn_interface.py $flash_attn_3_path
        echo "install fa3 success!"
    else
        echo "$python_path/flash_attn_interface.py not found, fa3 may not be installed!"
    fi
}

function install_deepep() {
    cd ${CURRENT_DIR}
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2506_torch28_cuda129_tev29/bzz_deepep/nvshmem.tar.gz

    tar -zxvf nvshmem.tar.gz
    cp -rf nvshmem /usr/local/nvshmem

    export NVSHMEM_DIR=/usr/local/nvshmem/
    export LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH
    export PATH=${NVSHMEM_DIR}/bin:$PATH

    rm -rf nvshmem
    rm -f nvshmem.tar.gz

    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2506_torch28_cuda129_tev29/bzz_deepep/deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl

    TORCH_CUDA_ARCH_LIST="10.0" pip install deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl
    rm -rf deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl
}

function upgrade_cudnn_9_16_0() {
    cd ${CURRENT_DIR}
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2506_torch28_cuda129_tev29/cudnn-local-repo-ubuntu2404-9.16.0_1.0-1_amd64.deb
    dpkg -i cudnn-local-repo-ubuntu2404-9.16.0_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu2404-9.16.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cudnn9-cuda-12
    dpkg -l |grep cudnn
    rm -f cudnn-local-repo-ubuntu2404-9.16.0_1.0-1_amd64.deb
}

function install_aiak_fp8_quant() {
    cd ${CURRENT_DIR}

    # 基于 https://console.cloud.baidu-int.com/devops/icode/repos/baidu/hac-aiacc/AIAK-FP8-Quantization/commits/2c5bdacec6f850fef16e016e89bdf5624aaeb4e3/setup.py
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2506_torch28_cuda129_tev29/AIAK-FP8-Quantization.tar.gz

    tar -zxvf AIAK-FP8-Quantization.tar.gz

    cd ${CURRENT_DIR}/AIAK-FP8-Quantization/

    python setup.py clean
    python setup.py build_ext --inplace --force
    pip install .

    cd ${CURRENT_DIR}

    rm -rf AIAK-FP8-Quantization/
    rm -f AIAK-FP8-Quantization.tar.gz
}

function clear_unused_file() {
    # 仅清理AIAK-Training-Omni自身的无用文件，删除外部库清理逻辑
    rm -rf /tmp/* ~/.bash_history
    rm -rf /workspace/AIAK-Training-Omni/.git
    rm -rf /workspace/AIAK-Training-Omni/build.sh
    rm -rf /workspace/AIAK-Training-Omni/ci.yml
    rm -rf /workspace/AIAK-Training-Omni/docker/ci || true
    
    rm -rf /workspace/TransformerEngine

    rm -rf /workspace/AIAK-Megatron/.git
    rm -rf /workspace/AIAK-Megatron/.github
    rm -rf /workspace/AIAK-Megatron/output

    rm /workspace/README.md
    rm /workspace/license.txt
    rm -rf /workspace/docker-examples
    rm -rf /workspace/tutorials

    rm -rf /workspace/lerobot
}

function apply_transformerEngine() {
    local transformerEngine_dir=/workspace/TransformerEngine
    local tag=v2.9
    
    # 1. Clone TransformerEngine from community
    export http_proxy=http://agent.baidu.com:8891
    export https_proxy=http://agent.baidu.com:8891
    export GIT_SSL_NO_VERIFY=1
    
    cd /workspace
    git clone https://github.com/NVIDIA/TransformerEngine.git
    cd $transformerEngine_dir
    
    # 2. Fetch all tags and checkout to target tags
    git fetch --all --tags
    git checkout $tag -b aiak_$tag
    git restore .
    
    unset http_proxy https_proxy
    
    # 3. Apply patches from omni
    if [ -f "/workspace/AIAK-Training-Omni/tools/apply_patches/apply_two_repo.sh" ]; then
        bash /workspace/AIAK-Training-Omni/tools/apply_patches/apply_two_repo.sh \
            /workspace/AIAK-Training-Omni/patches/TransformerEngine \
            $transformerEngine_dir
    else
        echo "Warning: apply_two_repo.sh not found, skipping patch application"
    fi

    # 4. Install dependencies (offline packages)
    thirdparty_path="${transformerEngine_dir}/3rdparty"
    rm -rf ${thirdparty_path} && mkdir -p ${thirdparty_path}
    cd ${thirdparty_path}

    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2506_torch28_cuda129_tev29/cudnn-frontend.tar.gz && tar -zxvf cudnn-frontend.tar.gz
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2506_torch28_cuda129_tev29/cutlass.tar.gz && tar -zxvf cutlass.tar.gz
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/aiak-training-llm/ngc2506_torch28_cuda129_tev29/googletest.tar.gz && tar -zxvf googletest.tar.gz

    rm -rf cudnn-frontend.tar.gz googletest.tar.gz cutlass.tar.gz

    # Install nvidia-mathdx, TE dependency
    pip install nvidia-mathdx==25.1.1 -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com

    # 5. Compile and install
    cd $transformerEngine_dir
    ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
    export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI"
    export NVTE_WITH_USERBUFFERS=1 && \
    export MPI_HOME=/usr/local/mpi && \
    export NVTE_FRAMEWORK=pytorch && \
    pip3 install --no-build-isolation .

    echo "TransformerEngine install completed"

    cd /workspace
    
    rm -rf $transformerEngine_dir
}

function apply_megatron() {
    local megatron_dir=/workspace/AIAK-Megatron
    local tag=core_v0.15.0
    
    # 1. Clone Megatron-LM from community
    export http_proxy=http://agent.baidu.com:8891
    export https_proxy=http://agent.baidu.com:8891
    export no_proxy=mirrors.baidubce.com
    export GIT_SSL_NO_VERIFY=1
    
    cd /workspace
    git clone https://github.com/NVIDIA/Megatron-LM.git
    mv Megatron-LM AIAK-Megatron
    cd $megatron_dir
    
    # 2. Fetch all tags and checkout to v0.15
    git fetch --all --tags
    git checkout $tag -b aiak_$tag
    git restore .
    
    unset http_proxy https_proxy
    
    # 3. Apply patches from omni repo
    if [ -f "/workspace/AIAK-Training-Omni/tools/apply_patches/apply_two_repo.sh" ]; then
        bash /workspace/AIAK-Training-Omni/tools/apply_patches/apply_two_repo.sh \
            /workspace/AIAK-Training-Omni/patches/Megatron-LM \
            $megatron_dir
    else
        echo "Warning: apply_two_repo.sh not found, skipping patch application"
    fi
    
    echo "Megatron-LM setup completed"
}


function install_lerobot() {
    echo "Installing lerobot..."
    
    export http_proxy=http://agent.baidu.com:8891
    export https_proxy=http://agent.baidu.com:8891
    export no_proxy=mirrors.baidubce.com
    export GIT_SSL_NO_VERIFY=1
    
    cd /workspace
    if [ -d "lerobot" ]; then
        rm -rf lerobot
    fi
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    
    # 移除 evdev 依赖，避免编译错误
    if [ -f "requirements-ubuntu.txt" ]; then
        sed -i '/evdev/d' requirements-ubuntu.txt
    fi
    
    # install [pi] extras
    pip install --no-build-isolation ".[pi]"
}


function install_aihclite_jupyter() {
    # 简化jupyter配置（仅创建目录，避免依赖外部脚本）
    mkdir -p /root/.jupyter
    touch /root/.jupyter/enterpoint.sh
    chmod +x /root/.jupyter/enterpoint.sh
}

function download_xpytorch() {
    # 为 p800 安装 xpytorch
    local XPYTORCH_URL=$1
    XPYTORCH_FILE=$(basename "${XPYTORCH_URL}")

    echo "下载 xpytorch 安装包: ${XPYTORCH_URL}"
    wget "${XPYTORCH_URL}"
    bash "${XPYTORCH_FILE}"
    rm "${XPYTORCH_FILE}"
}

# ======================== 参数解析（精简版） ========================
COMPILE_ENV=$1
BINARY_REPLACE=$2
INSTALL_TransformerEngine_FLAG=$3
MEGATRON_TYPE=$4
INSTALL_LEROBOT=${5:-"false"}

echo "Received COMPILE_ENV: ${COMPILE_ENV}"

XPYTORCH_URL_ARG=""
if [ -n "${5:-}" ]; then
    XPYTORCH_URL_ARG=$5
fi

CURRENT_DIR=$(cd `dirname $0`; pwd)

# ======================== 核心流程（仅保留基础环境） ========================

# 仅安装基础环境，跳过二进制替换/加密/BCCL/TransformerEngine等

# 如果是 p800 镜像，激活当前 conda 环境，download xpytorch
if [[ "$COMPILE_ENV" == "p800" ]];then
    . /root/miniconda/etc/profile.d/conda.sh
    conda activate python310_torch25_cuda

    if [[ -n "$XPYTORCH_URL_ARG" ]]; then
        XPYTORCH_URL="$XPYTORCH_URL_ARG"
        echo "使用指定的 xpytorch 版本: $XPYTORCH_URL"
    else
        xpytorch_info="/workspace/AIAK-Training-LLM/xpytorch_info.txt"

        if [[ ! -f "$xpytorch_info" ]]; then
            echo "xpytorch配置文件不存在: $xpytorch_info"
            exit 1
        fi

        DEFAULT_XPYTORCH_URL=$(cat "$xpytorch_info" | tr -d '[:space:]')

        if [[ -z "$DEFAULT_XPYTORCH_URL" ]]; then
            echo "Error: xpytorch_info.txt 文件内容为空"
            exit 1
        fi

        XPYTORCH_URL="$DEFAULT_XPYTORCH_URL"
        echo "使用默认的 xpytorch 版本: $XPYTORCH_URL"
    fi

    download_xpytorch "$XPYTORCH_URL"
fi

if  [[ "$BINARY_REPLACE" == "false" ]];then
    install_base_env $COMPILE_ENV
fi

# 跳过TransformerEngine安装
if  [[ "$INSTALL_TransformerEngine_FLAG" == "true" ]];then apply_transformerEngine; fi

# 跳过Megatron安装（无论community/aiak）
if  [[ "$MEGATRON_TYPE" == "community" ]];then apply_megatron; fi


# 安装lerobot
if [[ "$INSTALL_LEROBOT" == "true" ]]; then
    install_lerobot
else
    echo "Skipping lerobot installation (INSTALL_LEROBOT=${INSTALL_LEROBOT})"
fi


# 简化jupyter安装
install_aihclite_jupyter

# 清理无用文件
clear_unused_file

############################################################# 基础环境结束 ###########################################################