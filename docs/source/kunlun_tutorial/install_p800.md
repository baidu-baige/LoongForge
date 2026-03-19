# Installation on Kunlunxin P800

This document describes how to build the OmniTraining image that can run on Kunlunxin P800.

## Environment Preparation

We provide a clean base image with required underlying dependencies installed.

* UV environment (community Docker Hub): `weiyexu/omni_kunlun:uv_base`
* Conda environment (internal iregistry): `iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33`

Environment versions:

* **OS**: Ubuntu 22.04
* **Software**:
    * Python 3.10
    * PyTorch 2.5.1

## Building the Image

Run the following command from the project root directory:

```bash
BASE_IMAGE=weiyexu/omni_kunlun:uv_base
# For conda image:
# BASE_IMAGE=iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:v0.33
INSTALL_LEROBOT=false
DEFAULT_XPYTORCH_URL_ARG=https://baidu-kunlun-public.su.bcebos.com/baidu-kunlun-share/20260206/xpytorch-cp310-torch251-ubuntu2004-x64.run 
docker build  \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg INSTALL_LEROBOT=${INSTALL_LEROBOT} \
    --build-arg XPYTORCH_URL_ARG="${DEFAULT_XPYTORCH_URL_ARG}" \
    -t OmniTraining-kunlun:latest -f OmniTraining/docker/Dockerfile.xpu .
    # For conda image:
    #-t OmniTraining-kunlun:latest -f OmniTraining/docker/Dockerfile.xpu.internal .
```

After building, you can verify the image:

```bash
docker images | grep OmniTraining-kunlun:latest
```

At this point, all dependencies have been installed.

## Running the Image

The following example demonstrates how to start a container and mount project code, data, and other directories:

```bash
#!/bin/bash

image_addr='OmniTraining-kunlun:latest'
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
        -v /xxx:/data \
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

The virtual environment is activated by default. You can directly navigate to `/workspace/OmniTraining/examples_xpu/` to run the corresponding training scripts.