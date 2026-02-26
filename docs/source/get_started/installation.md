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
docker build -t OmniTraining:latest -f ./docker/Dockerfile .
```

After the build finishes, verify the image:

```bash
docker images | grep OmniTraining
```

---

### 1.3 Run the docker container
The example below starts a container and mounts the project code, data, etc.:

```bash
docker run --gpus all -it --rm \
  -v /path/to/your/hf/tokenizer:/workspace/tokenizer \
  -v /path/to/data:/workspace/data \
  -v /path/to/checkpoints:/workspace/checkpoints \
  OmniTraining:latest /bin/bash
```

Once inside the container, navigate to `/workspace/OmniTraining/examples/` and launch the desired training script.

## 2. Manual Patch Application
### 2.1 When to use
You already have a stable local installation of Megatron-LM / TransformerEngine and want to do secondary development or debugging on top of the upstream code.

### 2.2 Patch files overview
The project contains the following key directories:

* `patches/Megatron-LM/` – patches for the community Megatron-LM
* `patches/TransformerEngine/` – patches for the community TransformerEngine

### 2.3 Automated Environment Setup
We provide a helper script `setup_env.py` to automate the entire process, including cloning repositories, switching tags, applying patches, building TransformerEngine, and installing dependencies.

**Important: Recommended versions:**
- **Megatron-LM**: `0.15.0` (ensure the tag matches the remote repository, e.g., `core_v0.15.0`)
- **TransformerEngine**: `2.9` (e.g., `v2.9`)

**Usage:**

Run the following command from the project root (replace tags with the actual versions you need):

```bash
python setup_env.py --megatron-tag <MEGATRON_TAG> --te-tag <TE_TAG>
```

**Example:**

```bash
# Example for specific versions
python setup_env.py --megatron-tag core_v0.15.0 --te-tag v2.9
```

This script will automatically:
1. Clone `Megatron-LM` and `TransformerEngine` if they don't exist.
2. Checkout the specified tags and create local branches (`aiak_<tag>`).
3. Apply AIAK-Omni patches to both repositories.
4. Compile and install `TransformerEngine`.
5. Install all python dependencies for `OmniTraining`.

All dependencies are now installed; you can run the training scripts under `OmniTraining/examples/`.