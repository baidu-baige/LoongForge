# 昆仑芯 P800 安装

本文档介绍如何构建可在昆仑芯 P800 上运行的 LoongForge 镜像。

## 1. 使用 Docker 镜像构建与运行（推荐）

我们提供了预装所需底层依赖的纯净基础镜像。

* UV 环境（社区 Docker Hub）：`loongforge/loongforge_kunlun:py310_torch25`
* Conda 环境（您的镜像仓库）：`your-registry.example.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33`

环境版本：
* **操作系统**：Ubuntu 20.04
* **软件**：
    * Python 3.10
    * PyTorch 2.5.1
    * CUDA 11.7
### 1.2 构建 Docker 镜像

**构建前，请先使用子模块方式克隆仓库**，以确保 Loong-Megatron
源码包含在 Docker 构建上下文中：

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
```

然后构建镜像：

```bash
BASE_IMAGE=loongforge/loongforge_kunlun:py310_torch25
DEFAULT_XPYTORCH_URL_ARG=https://baidu-kunlun-public.su.bcebos.com/baidu-kunlun-share/20260409/torch25/xpytorch-cp310-torch251-ubuntu2004-x64.run
DEFAULT_KUNLUN_OPS_URL_ARG=https://baidu-kunlun-public.su.bcebos.com/baidu-kunlun-share/20260428/torch25/kunlun_ops-0.1.122%2Bb4984657-cp310-cp310-linux_x86_64.whl

: "${COCOPOD_URL_ARG:?请将 COCOPOD_URL_ARG 设置为可访问的 cocopod wheel URL}"
: "${XSPEEDGATE_URL_ARG:?请将 XSPEEDGATE_URL_ARG 设置为可访问的 xspeedgate wheel URL}"

docker build  \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg XPYTORCH_URL_ARG="${DEFAULT_XPYTORCH_URL_ARG}" \
    --build-arg COCOPOD_URL_ARG="${COCOPOD_URL_ARG}" \
    --build-arg XSPEEDGATE_URL_ARG="${XSPEEDGATE_URL_ARG}" \
    --build-arg KUNLUN_OPS_URL_ARG="${DEFAULT_KUNLUN_OPS_URL_ARG}" \
    -t LoongForge-kunlun:latest -f LoongForge/docker/Dockerfile.xpu .
```

> **说明**：所有模型系列（LLM / VLM / VLA / Diffusion）现在共用同一个 Docker 镜像。`ENABLE_LEROBOT` 构建参数已被移除 —— 构建镜像时不再区分 lerobot 与非 lerobot。

- `BASE_IMAGE` 是用于构建的基础镜像。可选值包括：
  * `loongforge/loongforge_kunlun:py310_torch25`（默认）[Docker Hub 提供]
  * `your-registry.example.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33`（示例 XMLIR 镜像）
- `XPYTORCH_URL_ARG` 是 xpytorch 安装程序的 URL 参数。
- `COCOPOD_URL_ARG` 和 `XSPEEDGATE_URL_ARG` 在公开检出版本中需要手动填写。
- `KUNLUN_OPS_URL_ARG` 可覆盖默认的公开包地址。

构建完成后，可验证镜像：

```bash
docker images | grep LoongForge
```

---

### 1.3 运行 Docker 容器
以下示例启动一个容器并挂载项目代码、数据等：

```bash
#!/bin/bash

image_addr='LoongForge-kunlun:latest'
DEFAULT_CONTAINER_NAME='loongforge-kunlun'

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 {start|exec|stop|rm} [container_name(default: ${DEFAULT_CONTAINER_NAME})]"
    exit 1
fi

ACTION=$1
CONTAINER_NAME=${2:-$DEFAULT_CONTAINER_NAME}

case $ACTION in
    start)
        echo "Starting container: $CONTAINER_NAME"
        docker run -itd \
        --security-opt=seccomp=unconfined \
        --cap-add=SYS_PTRACE \
        --ulimit=memlock=-1 --ulimit=nofile=120000 --ulimit=stack=67108864 \
        --shm-size=128G \
        --privileged \
        --net=host \
        --name=${CONTAINER_NAME} \
        -v /path/to/data:/workspace/data \
        -w /workspace/ \
        ${image_addr} bash

        docker cp -L  $(which xpu-smi) $CONTAINER_NAME:/bin/xpu-smi || true
        docker exec -it ${CONTAINER_NAME} bash
        ;;
    exec)
        echo "Exec container: $CONTAINER_NAME"
        docker exec -it ${CONTAINER_NAME} bash
        ;;
    stop)
        echo "Stopping container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME
        ;;
    rm)
        echo "Removing container: $CONTAINER_NAME"
        docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME
        ;;
    *)
        echo "Invalid action specified. Use {start|stop|rm}."
        exit 1
        ;;
esac
```

* 启动容器：`./docker_control.sh start`
* 进入容器：`./docker_control.sh exec`
* 删除容器：`./docker_control.sh rm`

进入容器后：
- Conda 环境镜像：通过 `conda activate python310_torch25_cuda` 激活
- UV 环境镜像：通过 `source /opt/loongforge_kunlun/bin/activate` 激活

虚拟环境默认已激活。您可以直接进入 `/workspace/LoongForge/examples_xpu/` 运行相应的训练脚本。

## 2. 从源码安装

如果您已有可用的昆仑 XPU + XPyTorch 环境，可以直接安装 LoongForge：

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[xpu]"
```

注意：XPU **不**需要 TransformerEngine。如需额外的 XPU 特定依赖（如 XPyTorch、DeepSpeed），请参考
[`docker/Dockerfile.xpu`](https://github.com/baidu-baige/LoongForge/blob/master/docker/Dockerfile.xpu)
了解具体版本和构建步骤。
