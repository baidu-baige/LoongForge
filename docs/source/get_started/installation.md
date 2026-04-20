# Installation

## 1. Build & Run with Docker Image (Recommended)
### 1.1 Env requirements
* Ubuntu 24.04
* NVIDIA GPU (Ampere / Hopper / Blackwell or newer)
* NVIDIA Driver (version must meet the CUDA requirement)
* Docker ≥ 20.10
* nvidia-container-toolkit

### 1.2 Build the docker image

**Before building, initialize the Loong-Megatron submodule** so its contents are included
in the Docker build context:

```bash
cd LoongForge
git submodule update --init third_party/Loong-Megatron
cd ..
```

Then build the image:

```bash
docker build --build-arg COMPILE_ENV=hopper --build-arg INSTALL_LEROBOT=false \
  -t LoongForge:latest -f ./LoongForge/docker/Dockerfile .
```
- `COMPILE_ENV` is used to specify the type of GPU (options: ampere, hopper, blackwell).
- `INSTALL_LEROBOT` is used to determine whether to install lerobot (options: true, false).

After the build finishes, verify the image:

```bash
docker images | grep LoongForge
```

---

### 1.3 Run the docker container
The example below starts a container and mounts the project code, data, etc.:

```bash
docker run --runtime --nvidia --gpus all -itd --rm \
  -v /path/to/your/hf/tokenizer:/mnt/cluster/huggingface.co/ \
  -v /path/to/data:/mnt/cluster/LoongForge/ \
  LoongForge:latest /bin/bash
```

Once inside the container, navigate to `/workspace/LoongForge/examples/` and launch the desired training script.

## 2. Manual Environment Setup
### 2.1 When to use
You already have a stable local environment and want to do secondary development or debugging.

### 2.2 Dependency overview

LoongForge uses two different strategies to manage its key dependencies:

| Dependency | Strategy | Location |
|---|---|---|
| **Megatron-LM** | git submodule → LoongForge fork | `third_party/Loong-Megatron/` |
| **TransformerEngine** | patch against upstream NVIDIA tag | `patches/TransformerEngine_<tag>/` |

**Megatron-LM** is pinned to a specific commit of the LoongForge fork via git submodule.
No patches are applied — all LoongForge-specific changes live directly in the fork branch.

**TransformerEngine** is cloned from the upstream NVIDIA repository, checked out at the
specified community tag, and then patched with LoongForge-specific fixes.
The patch directory suffix matches the upstream tag it targets (e.g. `patches/TransformerEngine_v2.9/`).

### 2.3 Automated Environment Setup
We provide a helper script `setup_env.py` to automate the entire process: initializing the
Megatron-LM submodule, cloning TransformerEngine, applying TE patches, building
TransformerEngine, and installing dependencies.

**Recommended versions:**
- **Megatron-LM**: locked by submodule commit (see `third_party/Loong-Megatron/`)
- **TransformerEngine**: `v2.9`

**Usage:**

Run the following command from the project root:

```bash
python setup_env.py --te-tag <TE_TAG>
```

**Example:**

```bash
python setup_env.py --te-tag v2.9
```

This script will automatically:
1. Initialize the `Loong-Megatron` submodule at `third_party/Loong-Megatron/`.
2. Clone `TransformerEngine` from the upstream NVIDIA repository.
3. Checkout the specified TE tag and create a local branch (`loongforge<tag>`).
4. Apply patches from `patches/TransformerEngine_<tag>/` to TransformerEngine.
5. Compile and install `TransformerEngine`.
6. Install all Python dependencies for `LoongForge`.

All dependencies are now installed; you can run the training scripts under `LoongForge/examples/`.
