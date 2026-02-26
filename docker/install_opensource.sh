#!/bin/bash
set -exuo pipefail

function install_pip_config() {
    if [ -f "pip.conf" ]; then
        cp pip.conf /etc/pip.conf
        cp pip.conf /usr/pip.conf
        mkdir -p /root/.pip && cp pip.conf /root/.pip/pip.conf
        mkdir -p /etc/xdg/pip && cp pip.conf /etc/xdg/pip/pip.conf
    fi
}

function install_base_env() {
    local COMPILE_ENV
    read COMPILE_ENV<<< $*

    cd ${CURRENT_DIR}

    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf

    install_requirements

    mkdir -p /tmp && chmod -R 777 /tmp
    install_pip_config

    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install bc tree pwgen nodejs iproute2 lsb-release -y
}

function install_requirements() {
    source_file="/etc/pip/constraint.txt"
    if [ -f "$source_file" ]; then
        backup_file="${source_file%.*}_backup.${source_file##*.}"
        mv -v "$source_file" "$backup_file"
        > "$source_file"
    fi

    requirements_file="/workspace/OmniTraining/requirements.txt"

    pip install -q --no-cache-dir wandb
    pip install -q --no-cache-dir swanlab==0.6.1
    
    local original_pwd=$(pwd)
    cd /workspace/OmniTraining/
    pip install -q --no-cache-dir --root-user-action=ignore -r ${requirements_file}
    cd ${original_pwd}

    # COMPILE_ENV
    if [[ "$COMPILE_ENV" != "p800" ]];then
        if [[ "$COMPILE_ENV" == "hzz" ]];then
            install_flash_attn
            update_flash_attn3_import_path_for_te
            install_deepep
            #install_aiak_fp8_quant
            upgrade_cudnn_9_16_0
        elif [[ "$COMPILE_ENV" == "bzz" ]];then
            echo "For NVIDIA Blackwell, we do not need to install anything."
        else
            echo "[ERROR] Not supported COMPILE_ENV: $COMPILE_ENV"
            exit 1
        fi
    fi

    echo "y" | pip uninstall nvidia-modelopt || true
}

function update_flash_attn3_import_path_for_te() {
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
    
    # 1. NVSHMEM installation
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get -y install nvshmem

    # 2. Deepep installation
    git clone https://github.com/deepseek-ai/DeepEP.git
    cd DeepEP
    git fetch --all --tags
    git checkout v1.2.1

    TORCH_CUDA_ARCH_LIST="9.0 10.0" pip install .
    rm -rf DeepEP
}

function upgrade_cudnn_9_16_0() {

    cd ${CURRENT_DIR}
    
    # use /etc/os-release to get ubuntu version
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        UBUNTU_VERSION=$(echo $VERSION_ID | tr -d '.')
    else
        echo "Error: /etc/os-release not found. Cannot determine Ubuntu version."
        exit 1
    fi

    wget -q https://developer.download.nvidia.com/compute/cudnn/9.16.0/local_installers/cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0_1.0-1_amd64.deb

    dpkg -i cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    
    apt-get update
    apt-get -y install cudnn-cuda-12
    
    dpkg -l | grep cudnn
    
    rm -f cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0_1.0-1_amd64.deb
}

function install_aiak_fp8_quant() {

    cd ${CURRENT_DIR}

    # 基于 https://console.cloud.baidu-int.com/devops/icode/repos/baidu/hac-aiacc/AIAK-FP8-Quantization/commits/2c5bdacec6f850fef16e016e89bdf5624aaeb4e3/setup.py
    wget -q https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/OmniTraining/ngc2506_torch28_cuda129_tev29/AIAK-FP8-Quantization.tar.gz

    tar -zxvf AIAK-FP8-Quantization.tar.gz

    cd ${CURRENT_DIR}/AIAK-FP8-Quantization/

    python setup.py clean
    python setup.py build_ext --inplace --force
    pip install .

    cd ${CURRENT_DIR}

    rm -rf AIAK-FP8-Quantization/
    rm -f AIAK-FP8-Quantization.tar.gz
}


function install_flash_attn() {

    cd ${CURRENT_DIR}

    # flash-attn2
    wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    pip install flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    rm flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    
    # flash-attn3
    wget -q https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.01.26-f6c4937/flash_attn_3-3.0.0b1%2B20260126.cu129torch280cxx11abitrue.438325-cp39-abi3-linux_x86_64.whl
    pip install flash_attn_3-3.0.0b1%2B20260126.cu129torch280cxx11abitrue.438325-cp39-abi3-linux_x86_64.whl
    rm flash_attn_3-3.0.0b1%2B20260126.cu129torch280cxx11abitrue.438325-cp39-abi3-linux_x86_64.whl
}

function clear_unused_file() {
    rm -rf /tmp/* ~/.bash_history
    rm -rf /workspace/OmniTraining/.git
    rm -rf /workspace/OmniTraining/build.sh
    rm -rf /workspace/OmniTraining/ci.yml
    rm -rf /workspace/OmniTraining/docker/ci || true
    
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
    cd /workspace
    git clone https://github.com/NVIDIA/TransformerEngine.git
    cd $transformerEngine_dir
    
    # 2. Fetch all tags and checkout to target tags
    git fetch --all --tags
    git checkout $tag -b aiak_$tag
    git restore .
    
    # 3. Apply patches from omni
    if [ -f "/workspace/OmniTraining/tools/apply_patches/apply_two_repo.sh" ]; then
        bash /workspace/OmniTraining/tools/apply_patches/apply_two_repo.sh \
            /workspace/OmniTraining/patches/TransformerEngine \
            $transformerEngine_dir
    else
        echo "Warning: apply_two_repo.sh not found, skipping patch application"
    fi

    # 4. Install dependencies\
    git submodule update --init --recursive
    
    # Install nvidia-mathdx, TE dependency
    pip install nvidia-mathdx==25.1.1

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
    cd /workspace
    git clone https://github.com/NVIDIA/Megatron-LM.git
    mv Megatron-LM AIAK-Megatron
    cd $megatron_dir
    
    # 2. Fetch all tags and checkout to v0.15
    git fetch --all --tags
    git checkout $tag -b aiak_$tag
    git restore .
    
    # 3. Apply patches from omni repo
    if [ -f "/workspace/OmniTraining/tools/apply_patches/apply_two_repo.sh" ]; then
        bash /workspace/OmniTraining/tools/apply_patches/apply_two_repo.sh \
            /workspace/OmniTraining/patches/Megatron-LM \
            $megatron_dir
    else
        echo "Warning: apply_two_repo.sh not found, skipping patch application"
    fi
    
    echo "Megatron-LM setup completed"
}

function install_lerobot() {
    echo "Installing lerobot..."
    
    cd /workspace
    if [ -d "lerobot" ]; then
        rm -rf lerobot
    fi
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    
    if [ -f "requirements-ubuntu.txt" ]; then
        sed -i '/evdev/d' requirements-ubuntu.txt
    fi
    
    # install [pi] extras
    pip install --no-build-isolation ".[pi]"
}

function install_aihclite_jupyter() {
    mkdir -p /root/.jupyter
    touch /root/.jupyter/enterpoint.sh
    chmod +x /root/.jupyter/enterpoint.sh
}


COMPILE_ENV=$1
INSTALL_TransformerEngine_FLAG=$2
INSTALL_LEROBOT=${3:-"false"}

echo "Received COMPILE_ENV: ${COMPILE_ENV}"

CURRENT_DIR=$(cd `dirname $0`; pwd)


install_base_env $COMPILE_ENV

if  [[ "$INSTALL_TransformerEngine_FLAG" == "true" ]];then 
    apply_transformerEngine
fi

apply_megatron

if [[ "$INSTALL_LEROBOT" == "true" ]]; then
    install_lerobot
else
    echo "Skipping lerobot installation (INSTALL_LEROBOT=${INSTALL_LEROBOT})"
fi

install_aihclite_jupyter

clear_unused_file