ARG BASE_IMAGE="nvcr.io/nvidia/pytorch:26.06-py3"
FROM ${BASE_IMAGE}

# =========================================================
# Build Arguments
# =========================================================
ARG COMPILE_ENV="hopper"
ARG FLASH_ATTN_VERSION="v2.8.3.post1"
ARG DEEPGEMM_VERSION="v2.1.1.post3"
ARG LEROBOT_VERSION="v0.5.0"

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
    apt-get install -y bc tree pwgen nodejs iproute2 && \
    apt-get install -y ffmpeg=7:6.1.1-3ubuntu5 && \
    rm -rf /var/lib/apt/lists/*

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
        local torch_cuda_arch_list="$1" && \
        local flash_attn_dir=/workspace/flash-attention && \
        echo "Building Flash Attention 2 & 3 ${FLASH_ATTN_VERSION} from source for CUDA arch ${torch_cuda_arch_list}..." && \
        cd /workspace && \
        git clone --recurse-submodules --shallow-submodules --depth 1 --branch "${FLASH_ATTN_VERSION}" \
            https://github.com/Dao-AILab/flash-attention.git "${flash_attn_dir}" && \
        cd "${flash_attn_dir}" && \
        MAX_JOBS=$(nproc) FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list}" \
            pip install --no-build-isolation --no-cache-dir . && \
        cd "${flash_attn_dir}/hopper" && \
        MAX_JOBS=$(nproc) FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list}" \
            pip install --no-build-isolation --no-cache-dir . && \
        cd /workspace && rm -rf "${flash_attn_dir}"; \
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
        local torch_cuda_arch_list="$1" && \
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
        cd /workspace && rm -rf DeepEP && \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    install_deepgemm() { \
        local deepgemm_dir=/workspace/DeepGEMM && \
        # Published wheels are tied to torch 2.8; build against the NGC image's PyTorch ABI. \
        echo "Building DeepGEMM ${DEEPGEMM_VERSION} from source..." && \
        cd /workspace && \
        git clone --recurse-submodules --shallow-submodules --depth 1 --branch "${DEEPGEMM_VERSION}" \
            https://github.com/deepseek-ai/DeepGEMM.git "${deepgemm_dir}" && \
        cd "${deepgemm_dir}" && \
        DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 pip install -v --no-build-isolation --no-cache-dir . && \
        cd /workspace && rm -rf "${deepgemm_dir}"; \
    }; \
    install_flashmla() { \
        echo "Installing FlashMLA..." && \
        git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla && \
        cd flash-mla && \
        git submodule update --init --recursive && \
        pip install -v . && \
        cd /workspace && rm -rf flash-mla; \
    }; \
    if [ "${COMPILE_ENV}" = "ampere" ]; then \
        torch_cuda_arch_list="8.0" && disable_sm90="1"; \
    elif [ "${COMPILE_ENV}" = "hopper" ]; then \
        torch_cuda_arch_list="9.0" && disable_sm90="0"; \
    elif [ "${COMPILE_ENV}" = "blackwell" ]; then \
        torch_cuda_arch_list="10.0" && disable_sm90="0"; \
    else \
        echo "[ERROR] Unsupported COMPILE_ENV value: ${COMPILE_ENV}" && \
        exit 1; \
    fi && \
    install_flash_attn "${torch_cuda_arch_list}" && \
    update_flash_attn3_path_for_te && \
    install_deepgemm && \
    install_flashmla && \
    install_deepep "${torch_cuda_arch_list}" "${disable_sm90}"

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
        echo "Installing Lerobot ${LEROBOT_VERSION}..." && \
        torch_ver_before=$(python -c "import torch; print(torch.__version__)") && \
        pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@32e97c9096571e82f4ce733f4ac726899e42b368" && \
        cd /workspace/LoongForge/ && \
        uv pip install --no-config --system --break-system-packages --no-cache -e ".[vla]" && \
        cd /workspace && \
        if [ -d "lerobot" ]; then \
            rm -rf lerobot; \
        fi && \
        git clone --depth 1 --branch "${LEROBOT_VERSION}" https://github.com/huggingface/lerobot.git && \
        cd lerobot && \
        if [ -f "requirements-ubuntu.txt" ]; then \
            sed -i '/evdev/d' requirements-ubuntu.txt; \
        fi && \
        # Keep the NGC PyTorch build instead of replacing it with a PyPI wheel. \
        sed -i -E 's/"torch>=([0-9][0-9.]*),<[0-9][0-9.]*"/"torch>=\1"/; s/"torchvision>=([0-9][0-9.]*),<[0-9][0-9.]*"/"torchvision>=\1"/' pyproject.toml && \
        grep -E '"torch(vision)?>=' pyproject.toml && \
        pip install "transformers==5.3.0" && \
        pip install --no-build-isolation . && \
        pip install "numpy==1.26.4" && \
        pip install "scipy==1.15.3" && \
        pip install --no-deps -U "torchcodec==0.11.1" && \
        torch_ver_after=$(python -c "import torch; print(torch.__version__)") && \
        if [ "${torch_ver_before}" != "${torch_ver_after}" ]; then \
            echo "[ERROR] torch was replaced during Lerobot install: ${torch_ver_before} -> ${torch_ver_after}" && \
            exit 1; \
        fi && \
        echo "Lerobot installation completed (torch kept at ${torch_ver_after})"; \
    }; \
    install_transformer_engine && \
    install_lerobot


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
