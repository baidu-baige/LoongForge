# Reference environment for ego2robot (Ubuntu 24.04, Python 3.12, CUDA 12.9).
#   docker build -t ego2robot -f docker/Dockerfile .
#   docker run --gpus all -it -v $PWD:/workspace/ego2robot ego2robot
FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg git libosmesa6 libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Keep the container's torch/torchvision; add the pinned pipeline packages.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps lerobot==0.5.0

# OSMesa is the verified renderer; MUJOCO_GL=egl is faster but some GPUs
# produce corrupted frames, so check its output before switching.
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1

WORKDIR /workspace/ego2robot
