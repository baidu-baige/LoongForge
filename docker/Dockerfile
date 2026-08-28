ARG BASE_IMAGE="nvcr.io/nvidia/pytorch:26.06-py3"
FROM ${BASE_IMAGE}

# =========================================================
# Build Arguments
# =========================================================
ARG COMPILE_ENV="hopper"
ARG FLASH_ATTN_VERSION="v2.8.3.post1"
ARG FAST_HADAMARD_VERSION="v1.1.0.post2"
ARG DEEPGEMM_VERSION="v2.1.1.post3"
ARG LEROBOT_VERSION="v0.5.0"
ARG FLASH_MLA_COMMIT="15f13e5030374295491c5ce31b02d7e63a7772c6"
ARG DMUON_COMMIT="32e97c9096571e82f4ce733f4ac726899e42b368"

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
RUN rm -rf /workspace/LoongForge
COPY ./LoongForge /workspace/LoongForge

# ======================== Basic Environment Setup ========================
RUN --mount=type=secret,id=apt_sources,required=false \
    set -exuo pipefail && \
    apt_options=() && \
    if [ -s /run/secrets/apt_sources ]; then \
        apt_options=(-o Dir::Etc::sourcelist=/run/secrets/apt_sources -o Dir::Etc::sourceparts=-); \
    fi && \
    apt_log=$(mktemp) && \
    apt_run() { \
        if ! apt-get -qq "${apt_options[@]}" "$@" >"${apt_log}" 2>&1; then \
            echo "apt-get command failed: $1" >&2; \
            rm -f "${apt_log}"; \
            return 1; \
        fi; \
    } && \
    apt_run update && \
    DEBIAN_FRONTEND=noninteractive apt_run install -y \
        tzdata bc tree pwgen nodejs iproute2 ffmpeg=7:6.1.1-3ubuntu5 && \
    rm -f "${apt_log}" && \
    mkdir -p /tmp && chmod -R 777 /tmp && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    rm -rf /var/lib/apt/lists/*

# ======================== Install Base Requirements ========================
RUN --mount=type=secret,id=pip_config,required=false \
    set -exuo pipefail && \
    . /workspace/LoongForge/docker/configure_python_index.sh && \
    cd /workspace/LoongForge/ && \
    pip --disable-pip-version-check --quiet install uv && \
    uv pip install --quiet --system --break-system-packages --no-cache -e ".[gpu]" && \
    pip --disable-pip-version-check --quiet install --no-cache-dir "nvidia-cudnn-cu13==9.24.0.43" && \
    pip --disable-pip-version-check --quiet uninstall -y nvidia-modelopt || true

# =========================================================
# Specific Environment Setup
# =========================================================
RUN --mount=type=secret,id=source_manifest,required=false \
    --mount=type=secret,id=pip_config,required=false \
    set -exuo pipefail && \
    . /workspace/LoongForge/docker/configure_python_index.sh && \
    if [ -d /usr/local/cuda/include/cccl ]; then \
        for cccl_dir in cuda cub thrust; do \
            [ -e "/usr/local/cuda/include/${cccl_dir}" ] || \
                ln -s "/usr/local/cuda/include/cccl/${cccl_dir}" "/usr/local/cuda/include/${cccl_dir}"; \
        done; \
    fi && \
    install_flash_attn() { \
        local torch_cuda_arch_list="$1" && \
        local flash_attn_dir=/workspace/flash-attention && \
        echo "Building Flash Attention 2 & 3 ${FLASH_ATTN_VERSION} from source for CUDA arch ${torch_cuda_arch_list}..." && \
        cd /workspace && \
        /workspace/LoongForge/docker/fetch_source.sh FLASH_ATTN_SOURCE "${flash_attn_dir}" flash-attention \
            git clone --recurse-submodules --shallow-submodules --depth 1 --branch "${FLASH_ATTN_VERSION}" \
            https://github.com/Dao-AILab/flash-attention.git "${flash_attn_dir}" >/dev/null && \
        cd "${flash_attn_dir}" && \
        MAX_JOBS=$(nproc) FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list}" \
            pip --disable-pip-version-check --quiet install --no-build-isolation --no-cache-dir . && \
        # FA3 (hopper variant) only for the hopper target: its setup.py
        # downloads an nvcc 12.6/12.8 toolchain that is incompatible with
        # the CUDA 13 base images used for ampere and blackwell, and the
        # operator-maintained P6K images ship FA2 only.
        if [ "${COMPILE_ENV}" = "hopper" ]; then \
            cd "${flash_attn_dir}/hopper" && \
            MAX_JOBS=$(nproc) FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list}" \
                pip --disable-pip-version-check --quiet install --no-build-isolation --no-cache-dir .; \
        fi && \
        cd /workspace && rm -rf "${flash_attn_dir}"; \
    }; \
    install_fast_hadamard_transform() { \
        local fast_hadamard_dir=/workspace/fast-hadamard-transform && \
        echo "Building fast_hadamard_transform ${FAST_HADAMARD_VERSION} from source..." && \
        cd /workspace && \
        /workspace/LoongForge/docker/fetch_source.sh FAST_HADAMARD_SOURCE "${fast_hadamard_dir}" fast-hadamard-transform \
            git clone --recurse-submodules --shallow-submodules --depth 1 --branch "${FAST_HADAMARD_VERSION}" \
            https://github.com/Dao-AILab/fast-hadamard-transform.git "${fast_hadamard_dir}" >/dev/null && \
        cd "${fast_hadamard_dir}" && \
        FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE FAST_HADAMARD_TRANSFORM_FORCE_CXX11_ABI=TRUE \
            pip --disable-pip-version-check --quiet install --no-build-isolation --no-cache-dir . && \
        cd /workspace && rm -rf "${fast_hadamard_dir}"; \
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
        local deepep_dir=/workspace/DeepEP && \
        echo "Installing DeepEP..." && \
        if [ -d /usr/include/nvshmem_13 ] && [ -d /usr/lib/x86_64-linux-gnu/nvshmem/13 ]; then \
            mkdir -p "${NVSHMEM_DIR}" && \
            ln -sfn /usr/include/nvshmem_13 "${NVSHMEM_DIR}/include" && \
            if [ -f /usr/local/cuda/lib64/libnvshmem_device.a ]; then \
                ln -sfn /usr/local/cuda/lib64 "${NVSHMEM_DIR}/lib"; \
            else \
                ln -sfn /usr/lib/x86_64-linux-gnu/nvshmem/13 "${NVSHMEM_DIR}/lib"; \
            fi; \
        else \
            [ ! -s /run/secrets/source_manifest ] || { \
                echo "[ERROR] internal build base image does not provide NVSHMEM"; exit 1; \
            } && \
            wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
            dpkg -i cuda-keyring_1.1-1_all.deb && \
            apt-get -qq update && apt-get -qq -y install nvshmem && \
            rm cuda-keyring_1.1-1_all.deb; \
        fi && \
        /workspace/LoongForge/docker/fetch_source.sh DEEPEP_SOURCE "${deepep_dir}" DeepEP \
            git clone --depth 1 --branch v1.2.1 https://github.com/deepseek-ai/DeepEP.git "${deepep_dir}" >/dev/null && \
        cd "${deepep_dir}" && \
        # NVSHMEM device linking only works on the hopper target: the CUDA 13
        # base images ship NVSHMEM device bits that do not satisfy DeepEP's
        # `-dlink -lnvshmem_device`, and single-node regressions never touch
        # the internode paths that need it.
        if [ "${COMPILE_ENV}" != "hopper" ]; then \
            sed -i '/disable_nvshmem = False/s/False/True/' setup.py; \
        fi && \
        TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list}" DISABLE_SM90_FEATURES="${disable_sm90}" \
            pip --disable-pip-version-check --quiet install --no-build-isolation . && \
        cd /workspace && rm -rf "${deepep_dir}" && \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    install_deepgemm() { \
        local deepgemm_dir=/workspace/DeepGEMM && \
        # Published wheels are tied to torch 2.8; build against the NGC image's PyTorch ABI. \
        echo "Building DeepGEMM ${DEEPGEMM_VERSION} from source..." && \
        cd /workspace && \
        /workspace/LoongForge/docker/fetch_source.sh DEEPGEMM_SOURCE "${deepgemm_dir}" DeepGEMM \
            git clone --recurse-submodules --shallow-submodules --depth 1 --branch "${DEEPGEMM_VERSION}" \
            https://github.com/deepseek-ai/DeepGEMM.git "${deepgemm_dir}" >/dev/null && \
        cd "${deepgemm_dir}" && \
        DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
            pip --disable-pip-version-check --quiet install --no-build-isolation --no-cache-dir . && \
        cd /workspace && rm -rf "${deepgemm_dir}"; \
    }; \
    install_flashmla() { \
        local flash_mla_dir=/workspace/flash-mla && \
        echo "Building FlashMLA commit ${FLASH_MLA_COMMIT} from source (no upstream release tag exists)..." && \
        source_mode=$(/workspace/LoongForge/docker/fetch_source.sh FLASHMLA_SOURCE "${flash_mla_dir}" FlashMLA \
            git clone --recurse-submodules --shallow-submodules --depth 1 \
            https://github.com/deepseek-ai/FlashMLA.git "${flash_mla_dir}") && \
        cd "${flash_mla_dir}" && \
        if [ "${source_mode}" = public ]; then \
            git fetch --depth 1 origin "${FLASH_MLA_COMMIT}" && \
            git checkout --detach "${FLASH_MLA_COMMIT}" && \
            git submodule update --init --recursive; \
        fi && \
        sed -i 's/version="1.0.0" + rev/version="1.0.0+local"/' setup.py && \
        pip --disable-pip-version-check --quiet install --no-build-isolation --no-cache-dir . && \
        cd /workspace && rm -rf "${flash_mla_dir}"; \
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
    install_fast_hadamard_transform && \
    update_flash_attn3_path_for_te && \
    install_deepgemm && \
    if [ "${COMPILE_ENV}" != "ampere" ]; then install_flashmla; fi && \
    install_deepep "${torch_cuda_arch_list}" "${disable_sm90}"

# =========================================================
# TransformerEngine / Lerobot
# NOTE: Megatron is managed as a git submodule (third_party/Loong-Megatron)
#       and is already included via COPY ./LoongForge. No clone or patch needed.
# =========================================================
RUN --mount=type=secret,id=source_manifest,required=false \
    --mount=type=secret,id=pip_config,required=false \
    set -exuo pipefail && \
    . /workspace/LoongForge/docker/configure_python_index.sh && \
    install_transformer_engine() { \
        local transformerEngine_dir=/workspace/TransformerEngine; \
        local TE_TAGS="v2.9" && \
        echo "Installing TransformerEngine ${TE_TAGS}..." && \
        cd /workspace && \
        source_mode=$(/workspace/LoongForge/docker/fetch_source.sh TRANSFORMER_ENGINE_SOURCE \
            "${transformerEngine_dir}" TransformerEngine \
            git clone --recurse-submodules --shallow-submodules --depth 1 --branch "${TE_TAGS}" \
            https://github.com/NVIDIA/TransformerEngine.git "${transformerEngine_dir}") && \
        cd "${transformerEngine_dir}" && \
        if [ "${source_mode}" = internal ]; then \
            git init && \
            git config user.name loongforge-ci && \
            git config user.email loongforge-ci@localhost && \
            git add . && git commit -m "Internal source baseline"; \
        else \
            git checkout -b "loongforge_${TE_TAGS}"; \
        fi && \
        git restore . && \
        if [ -f "/workspace/LoongForge/patches/apply_patches.sh" ]; then \
            NON_INTERACTIVE=true bash /workspace/LoongForge/patches/apply_patches.sh \
                /workspace/LoongForge/patches/TransformerEngine_${TE_TAGS} \
                ${transformerEngine_dir}; \
        fi && \
        if [ "${source_mode}" = public ]; then git submodule update --init --recursive; fi && \
        pip --disable-pip-version-check --quiet install nvidia-mathdx==25.1.1 && \
        ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))") && \
        export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" && \
        export NVTE_WITH_USERBUFFERS=1 && \
        export MPI_HOME=/usr/local/mpi && \
        export NVTE_FRAMEWORK=pytorch && \
        pip3 --disable-pip-version-check --quiet install --no-build-isolation . && \
        echo "TransformerEngine install completed" && \
        cd /workspace && rm -rf ${transformerEngine_dir}; \
    }; \
    install_lerobot() { \
        echo "Installing Lerobot ${LEROBOT_VERSION}..." && \
        torch_ver_before=$(python -c "import torch; print(torch.__version__)") && \
        dmuon_mode=$(/workspace/LoongForge/docker/fetch_source.sh DMUON_SOURCE /workspace/dmuon dmuon \
            git clone https://github.com/X-Square-Robot/dmuon.git /workspace/dmuon) && \
        if [ "${dmuon_mode}" = public ]; then \
            git -C /workspace/dmuon checkout --detach "${DMUON_COMMIT}"; \
        fi && \
        pip --disable-pip-version-check --quiet install /workspace/dmuon && rm -rf /workspace/dmuon && \
        cd /workspace/LoongForge/ && \
        uv pip install --quiet --no-config --system --break-system-packages --no-cache -e ".[vla]" && \
        cd /workspace && \
        if [ -d "lerobot" ]; then \
            rm -rf lerobot; \
        fi && \
        /workspace/LoongForge/docker/fetch_source.sh LEROBOT_SOURCE /workspace/lerobot lerobot \
            git clone --depth 1 --branch "${LEROBOT_VERSION}" \
            https://github.com/huggingface/lerobot.git /workspace/lerobot >/dev/null && \
        cd lerobot && \
        if [ -f "requirements-ubuntu.txt" ]; then \
            sed -i '/evdev/d' requirements-ubuntu.txt; \
        fi && \
        # Keep the NGC PyTorch build instead of replacing it with a PyPI wheel. \
        sed -i -E 's/"torch>=([0-9][0-9.]*),<[0-9][0-9.]*"/"torch>=\1"/; s/"torchvision>=([0-9][0-9.]*),<[0-9][0-9.]*"/"torchvision>=\1"/' pyproject.toml && \
        grep -E '"torch(vision)?>=' pyproject.toml && \
        pip --disable-pip-version-check --quiet install "transformers==5.3.0" && \
        pip --disable-pip-version-check --quiet install --no-build-isolation . && \
        pip --disable-pip-version-check --quiet install "numpy==1.26.4" && \
        pip --disable-pip-version-check --quiet install "scipy==1.15.3" && \
        pip --disable-pip-version-check --quiet install --no-deps -U "torchcodec==0.11.1" && \
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
# INT4 QAT CUDA Kernels (fake quantization for QAT)
# =========================================================
RUN set -exuo pipefail && \
    echo "Installing INT4 QAT CUDA kernels..." && \
    cd /workspace/LoongForge/ops/cuda_source/int4_qat && \
    pip install --no-build-isolation . && \
    echo "INT4 QAT install completed"

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
