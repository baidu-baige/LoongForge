ARG BASE_IMAGE="nvcr.io/nvidia/pytorch:25.06-py3"
FROM ${BASE_IMAGE}

# =========================================================
# Build Arguments
# =========================================================
ARG COMPILE_ENV={{COMPILE_ENV}}
# LeRobot has dependency conflicts with the base environment; disabled by default.
# Enable only when training VLA models (e.g., Pi0.5, GR00T).
ARG ENABLE_LEROBOT="false"

# =========================================================
# Environment Variables
# =========================================================
ENV NVSHMEM_DIR=/usr/local/nvshmem
ENV LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}
ENV PATH=${NVSHMEM_DIR}/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# =========================================================
# Base System Configuration + APT Source Replacement
# =========================================================
COPY ./LoongForge /workspace/LoongForge

# ======================== Basic Environment Setup ========================
RUN set -exuo pipefail && \
    mkdir -p /tmp && chmod -R 777 /tmp && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get install -y bc tree pwgen nodejs iproute2

# ======================== Install Base Requirements ========================
RUN set -exuo pipefail && \
    cd /workspace/LoongForge/ && \
    pip install uv && \
    uv pip install --system --break-system-packages --no-cache -e ".[gpu]" && \
    echo "y" | pip uninstall nvidia-modelopt || true

# =========================================================
# Specific Environment Setup
# =========================================================
RUN set -exuo pipefail && \
    install_flash_attn() { \
        echo "Installing Flash Attention 2 & 3..." && \
        wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
        pip install flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
        rm flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
        wget -q https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.01.26-f6c4937/flash_attn_3-3.0.0b1%2B20260126.cu129torch280cxx11abitrue.438325-cp39-abi3-linux_x86_64.whl && \
        pip install flash_attn_3-3.0.0b1%2B20260126.cu129torch280cxx11abitrue.438325-cp39-abi3-linux_x86_64.whl && \
        rm flash_attn_3-3.0.0b1%2B20260126.cu129torch280cxx11abitrue.438325-cp39-abi3-linux_x86_64.whl; \
    }; \
    update_flash_attn3_path_for_te() { \
        echo "Updating Flash Attention 3 import path for TE..." && \
        python_path=$(python -c "import site; print(site.getsitepackages()[0])") && \
        mkdir -p ${python_path}/flash_attn_3/ && \
        if [ -f "${python_path}/flash_attn_interface.py" ]; then \
            cp ${python_path}/flash_attn_interface.py ${python_path}/flash_attn_3/ && \
            echo "Flash Attention 3 adapter installed successfully!"; \
        else \
            echo "Warning: flash_attn_interface.py not found"; \
        fi; \
    }; \
    install_deepep() { \
        local torch_cuda_arch_list="$1"; \
        local disable_sm90="$2" && \
        echo "Installing DeepEP..." && \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update && \
        apt-get -y install nvshmem && \
        rm cuda-keyring_1.1-1_all.deb && \
        git clone https://github.com/deepseek-ai/DeepEP.git && \
        cd DeepEP && \
        git fetch --all --tags && \
        git checkout v1.2.1 && \
        TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list}" DISABLE_SM90_FEATURES="${disable_sm90}" pip install --no-build-isolation . && \
        cd .. && rm -rf DeepEP && \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    upgrade_cudnn_9_16_0() { \
        echo "Upgrading cuDNN to 9.16.0..." && \
        UBUNTU_VERSION=$(cat /etc/os-release | grep VERSION_ID | cut -d '"' -f 2 | tr -d '.') && \
        wget -q https://developer.download.nvidia.com/compute/cudnn/9.16.0/local_installers/cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0_1.0-1_amd64.deb && \
        dpkg -i cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0_1.0-1_amd64.deb && \
        cp /var/cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
        apt-get update && \
        apt-get -y install cudnn9-cuda-12 && \
        dpkg -l | grep cudnn && \
        rm -f cudnn-local-repo-ubuntu${UBUNTU_VERSION}-9.16.0_1.0-1_amd64.deb && \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    install_deepgemm() { \
        echo "Installing DeepGEMM..." && \
        wget -q https://github.com/deepseek-ai/DeepGEMM/releases/download/v2.1.1.post3/deep_gemm-2.1.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
        pip install -v deep_gemm-2.1.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && \
        rm deep_gemm-2.1.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl; \
    }; \
    install_flashmla() { \
        echo "Installing FlashMLA..." && \
        git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && \
        cd flash-mla && \
        git submodule update --init --recursive && \
        pip install -v . && \
        rm -rf flash-mla; \
    }; \
    update_flash_attn3_path_for_te && \
    upgrade_cudnn_9_16_0 && \
    install_deepgemm && \
    install_flashmla && \
    if [ "${COMPILE_ENV}" = "ampere" ]; then \
        install_deepep "8.0" "1"; \
    elif [ "${COMPILE_ENV}" = "hopper" ]; then \
        install_deepep "9.0" "0"; \
    elif [ "${COMPILE_ENV}" = "blackwell" ]; then \
        install_deepep "10.0" "0"; \
    else \
        echo "[ERROR] Unsupported COMPILE_ENV value: ${COMPILE_ENV}" && \
        exit 1; \
    fi

# =========================================================
# TransformerEngine / Lerobot
# NOTE: Megatron is managed as a git submodule (third_party/Loong-Megatron)
#       and is already included via COPY ./LoongForge. No clone or patch needed.
# =========================================================
RUN set -exuo pipefail && \
    install_transformer_engine() { \
        local transformerEngine_dir=/workspace/TransformerEngine; \
        local TE_TAGS="v2.9" && \
        echo "Installing TransformerEngine ${TE_TAGS}..." && \
        cd /workspace && \
        git clone https://github.com/NVIDIA/TransformerEngine.git && \
        cd ${transformerEngine_dir} && \
        git fetch --all --tags && \
        git checkout ${TE_TAGS} -b loongforge_${TE_TAGS} && \
        git restore . && \
        if [ -f "/workspace/LoongForge/patches/apply_patches.sh" ]; then \
            NON_INTERACTIVE=true bash /workspace/LoongForge/patches/apply_patches.sh \
                /workspace/LoongForge/patches/TransformerEngine_${TE_TAGS} \
                ${transformerEngine_dir}; \
        fi && \
        git submodule update --init --recursive && \
        pip install nvidia-mathdx==25.1.1 && \
        ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))") && \
        export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" && \
        export NVTE_WITH_USERBUFFERS=1 && \
        export MPI_HOME=/usr/local/mpi && \
        export NVTE_FRAMEWORK=pytorch && \
        pip3 install --no-build-isolation . && \
        echo "TransformerEngine install completed" && \
        cd /workspace && rm -rf ${transformerEngine_dir}; \
    }; \
    install_lerobot() { \
        echo "Installing Lerobot..." && \
        cd /workspace && \
        if [ -d "lerobot" ]; then \
            rm -rf lerobot; \
        fi && \
        git clone https://github.com/huggingface/lerobot.git && \
        cd lerobot && \
        if [ -f "requirements-ubuntu.txt" ]; then \
            sed -i '/evdev/d' requirements-ubuntu.txt; \
        fi && \
        pip install --no-build-isolation ".[pi]" && \
        echo "Lerobot installation completed"; \
    }; \
    install_transformer_engine && \
    if [ "$ENABLE_LEROBOT" = "true" ]; then \
        install_lerobot; \
    else \
        echo "Skipping lerobot installation (ENABLE_LEROBOT=${ENABLE_LEROBOT})"; \
    fi


# =========================================================
# Megatron Symlink
# Scripts reference /workspace/Loong-Megatron; the actual location is
# /workspace/LoongForge/third_party/Loong-Megatron (git submodule).
# =========================================================
RUN ln -sf /workspace/LoongForge/third_party/Loong-Megatron /workspace/Loong-Megatron

# =========================================================
# Jupyter Configuration
# =========================================================
RUN mkdir -p /root/.jupyter && \
    touch /root/.jupyter/enterpoint.sh && \
    chmod +x /root/.jupyter/enterpoint.sh

# =========================================================
# Clean Up Unused Files (Consistent with install.sh)
# =========================================================
RUN rm -rf /tmp/* ~/.bash_history && \
    rm -rf /workspace/LoongForge/.git && \
    rm -rf /workspace/LoongForge/third_party/Loong-Megatron/.git && \
    rm -rf /workspace/LoongForge/build.sh && \
    rm -rf /workspace/LoongForge/ci.yml && \
    rm -rf /workspace/LoongForge/docker/ci && \
    rm -rf /workspace/*.tar.gz && \
    rm -f /workspace/README.md /workspace/license.txt && \
    rm -rf /workspace/docker-examples /workspace/tutorials /workspace/lerobot

WORKDIR /workspace