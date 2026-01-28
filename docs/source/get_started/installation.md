# Installation

## 1. Build & Run with Docker Image (Recommended)
### 1.1 Env requirements
* Ubuntu 22.04
* NVIDIA GPU (Ampere / Hopper or newer)
* NVIDIA Driver (version must meet the CUDA requirement)
* Docker ≥ 20.10
* nvidia-container-toolkit

### 1.2 Build the docker image
From the project root run:

```bash
docker build -t AIAK-Training-Omni:latest -f ./docker/Dockerfile .
```

After the build finishes, verify the image:

```bash
docker images | grep AIAK-Training-Omni
```

---

### 1.3 Run the docker container
The example below starts a container and mounts the project code, data, etc.:

```bash
docker run --gpus all -it --rm \
  -v /path/to/your/hf/tokenizer:/workspace/tokenizer \
  -v /path/to/data:/workspace/data \
  -v /path/to/checkpoints:/workspace/checkpoints \
  AIAK-Training-Omni:latest /bin/bash
```

Once inside the container, navigate to `/workspace/AIAK-Training-Omni/examples/` and launch the desired training script.

## 2. Manual Patch Application
### 2.1 When to use
You already have a stable local installation of Megatron-LM / TransformerEngine and want to do secondary development or debugging on top of the upstream code.

### 2.2 Patch files overview
The project contains the following key directories:

* `patches/Megatron-LM/` – patches for the community Megatron-LM
* `patches/TransformerEngine/` – patches for the community TransformerEngine

### 2.3 Apply patches
**1. Clone upstream repositories**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
git clone https://github.com/NVIDIA/TransformerEngine.git
```

**2. Switch to the required tag**

```bash
cd Megatron-LM
git fetch --all --tags
git checkout $tag -b aiak_$tag
git restore .

cd TransformerEngine
git fetch --all --tags
git checkout $tag -b aiak_$tag
git restore .
```

**The tag must match the version declared in the Dockerfile / README; otherwise the patches may fail or behave inconsistently.**

**3. Apply the patches**

```bash
bash {AIAK-Training-Omni-path}/tools/apply_patches/apply_two_repo.sh \
     {AIAK-Training-Omni-path}/patches/Megatron-LM {Megatron-LM-path}

bash {AIAK-Training-Omni-path}/tools/apply_patches/apply_two_repo.sh \
     {AIAK-Training-Omni-path}/patches/TransformerEngine {TransformerEngine-path}
```

The script `apply_two_repo.sh` applies the corresponding patch files to the upstream repositories. After patching, build TransformerEngine:

```bash
cd TransformerEngine
git submodule update --init --recursive
export NVTE_FRAMEWORK=pytorch         # optional
pip3 install --no-build-isolation .   # build & install
```

**4. Install Omni dependencies**

```bash
cd AIAK-Training-Omni
pip install -r requirements.txt
```

All dependencies are now installed; you can run the training scripts under `AIAK-Training-Omni/examples/`.