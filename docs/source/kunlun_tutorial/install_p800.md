# Installation on Kunlunxin P800

This document describes how to build the LoongForge image that can run on Kunlunxin P800.

## 1. Build & Run with Docker Image (Recommended)

We provide a clean base image with required underlying dependencies installed.

* UV environment (community Docker Hub): `weiyexu/omni_kunlun:uv_base`
* Conda environment (internal iregistry): `iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33`

Environment versions:
* **OS**: Ubuntu 20.04
* **Software**:
    * Python 3.10
    * PyTorch 2.5.1
    * CUDA 11.7
### 1.2 Build the docker image

**Before building, clone the repository with submodules** so the Loong-Megatron
source is included in the Docker build context:

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
```

Then build the image:

```bash
BASE_IMAGE=weiyexu/omni_kunlun:uv_base
ENABLE_LEROBOT=false
DEFAULT_XPYTORCH_URL_ARG=https://baidu-kunlun-public.su.bcebos.com/baidu-kunlun-share/20260206/xpytorch-cp310-torch251-ubuntu2004-x64.run 
docker build  \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg ENABLE_LEROBOT=${ENABLE_LEROBOT} \
    --build-arg XPYTORCH_URL_ARG="${DEFAULT_XPYTORCH_URL_ARG}" \
    -t LoongForge-kunlun:latest -f LoongForge/docker/Dockerfile.xpu .
    # For internal conda image:
    #-t LoongForge-kunlun:latest -f LoongForge/docker/Dockerfile.xpu.internal .
```
- `BASE_IMAGE` is the base image used for building. Options include:
  * `weiyexu/omni_kunlun:uv_base` (default) [available at Docker Hub]
  * `iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33` [internal use only]
- `XPYTORCH_URL_ARG` is the xpytorch installer url argument.
- `ENABLE_LEROBOT`: enable LeRobot dependencies for VLA model training (e.g., Pi0.5, GR00T). Disabled by default due to dependency conflicts with the base environment. Options: `true`, `false` (default).
After building, you can verify the image:

```bash
docker images | grep LoongForge
```

---

### 1.3 Run the docker container
The example below starts a container and mounts the project code, data, etc.:

```bash
#!/bin/bash

image_addr='LoongForge-kunlun:latest'
DEFAULT_CONTAINER_NAME='omni-kunlun'

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

* Start container: `./docker_control.sh start`
* Enter container: `./docker_control.sh exec`
* Remove container: `./docker_control.sh rm`

After entering the container:
- For conda environment image: activate via `conda activate python310_torch25_cuda`
- For UV image: activate via `source /opt/omni_kunlun/bin/activate`

The virtual environment is activated by default. You can directly navigate to `/workspace/LoongForge/examples_xpu/` to run the corresponding training scripts.

## 2. Install from Source

If you already have a working Kunlun XPU + XPyTorch environment, you can
install LoongForge directly:

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[xpu]"
```

Note: TransformerEngine is **not** required for XPU. For additional XPU-specific
dependencies (e.g. XPyTorch, DeepSpeed), refer to
[`docker/Dockerfile.xpu`](https://github.com/baidu-baige/LoongForge/blob/master/docker/Dockerfile.xpu)
for exact versions and build steps.
