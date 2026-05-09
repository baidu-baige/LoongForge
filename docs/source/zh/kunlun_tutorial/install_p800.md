# 昆仑芯 P800 安装

本文档介绍如何构建可在昆仑芯 P800 上运行的 LoongForge 镜像。

## 1. 使用 Docker 镜像构建与运行（推荐）

我们提供了预装所需底层依赖的纯净基础镜像。

* UV 环境（社区 Docker Hub）：`loongforge/loongforge_kunlun:py310_torch25`
* Conda 环境（内部 iregistry）：`iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33`

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
ENABLE_LEROBOT=false
DEFAULT_XPYTORCH_URL_ARG=https://baidu-kunlun-public.su.bcebos.com/baidu-kunlun-share/20260409/torch25/xpytorch-cp310-torch251-ubuntu2004-x64.run

docker build  \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg ENABLE_LEROBOT=${ENABLE_LEROBOT} \
    --build-arg XPYTORCH_URL_ARG="${DEFAULT_XPYTORCH_URL_ARG}" \
    -t LoongForge-kunlun:latest -f LoongForge/docker/Dockerfile.xpu .
    # 内部 conda 镜像：
    #-t LoongForge-kunlun:latest -f LoongForge/docker/Dockerfile.xpu.internal .
```
- `BASE_IMAGE` 是用于构建的基础镜像。可选值包括：
  * `loongforge/loongforge_kunlun:py310_torch25`（默认）[Docker Hub 提供]
  * `iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33`[仅限内部使用]
- `XPYTORCH_URL_ARG` 是 xpytorch 安装程序的 URL 参数。
- `ENABLE_LEROBOT`：启用 VLA 模型训练（如 Pi0.5、GR00T）的 LeRobot 依赖。由于与基础环境存在依赖冲突，默认禁用。可选值：`true`、`false`（默认）。

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
        -v /path/to/data:/mnt/cluster/LoongForge/ \
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
