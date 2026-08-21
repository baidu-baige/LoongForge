# Installation

## System Requirements

### Hardware

- **Required**: NVIDIA GPU (Ampere / Hopper or newer)
- **NVIDIA Driver**: Version must meet the CUDA Toolkit requirement

### Software

- **Python**: >= 3.10
- **PyTorch**: >= 2.6.0
- **CUDA Toolkit**: >= 12.1
- **OS**: Linux (Ubuntu 22.04 / 24.04 recommended)

Note: For Kunlun XPU installation, see the
[Kunlun Installation Guide](../kunlun_tutorial/install_p800.md).

## Prerequisites

Install [uv](https://docs.astral.sh/uv/), a fast Python package installer and resolver:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Dependency Overview

LoongForge uses two different strategies to manage its key upstream dependencies:

| Dependency | Strategy | Location |
|---|---|---|
| **Megatron-LM** | git submodule (LoongForge fork) | `third_party/Loong-Megatron/` |
| **TransformerEngine** | patch against upstream NVIDIA tag | `patches/TransformerEngine_<tag>/` |

**Megatron-LM** is pinned to a specific commit of the
[Loong-Megatron](https://github.com/baidu-baige/Loong-Megatron) fork via git
submodule. All LoongForge-specific changes live directly in the fork branch —
no patches are applied.

**TransformerEngine** is cloned from the upstream NVIDIA repository, checked out
at the specified community tag, and then patched with LoongForge-specific fixes.
The patch directory suffix matches the upstream tag it targets
(e.g. `patches/TransformerEngine_v2.9/`).

---

## Option A: Docker Image (Recommended)

Use this option if you want a fully reproducible, ready-to-train environment
with zero manual dependency management.

### Prerequisites

- Docker >= 20.10
- nvidia-container-toolkit

### Build the image

Before building, clone the repository with submodules so the Loong-Megatron
source is included in the Docker build context:

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
```

Then build the image:

```bash
docker build --build-arg COMPILE_ENV=hopper \
  -t loongforge:latest -f ./LoongForge/docker/Dockerfile .
```

> **Note:** All model families (LLM / VLM / VLA / Diffusion) now share a single
> Docker image. The `ENABLE_LEROBOT` build argument and the separate `_lerobot`
> image variant have been removed — there is no longer a lerobot / non-lerobot
> distinction when building or pulling images.

| Build Arg | Description | Options |
|---|---|---|
| `COMPILE_ENV` | Target GPU architecture | `ampere`, `hopper`|

After the build finishes, verify:

```bash
docker images | grep loongforge
```

---

## Pre-built Docker Images

LoongForge Docker images are available on Docker Hub:
[https://hub.docker.com/u/loongforge](https://hub.docker.com/u/loongforge).

LoongForge publishes versioned pre-built Docker images. Select the desired tag
from Docker Hub. A single image covers all model families (LLM / VLM / VLA /
Diffusion) — there is no longer a separate `_lerobot` variant.

| Image | Tag Pattern | Description |
|---|---|---|
| `loongforge/loongforge` | `<version>` | Unified image: LLM / VLM / VLA / Diffusion training |

```bash
# Set the version tag you want to use, for example: 0.1.1
LOONGFORGE_VERSION=<version>

# Pull the image
docker pull loongforge/loongforge:${LOONGFORGE_VERSION}
```

### Run the container

```bash
# Set the version tag you want to use, for example: 0.1.1
LOONGFORGE_VERSION=<version>

docker run --runtime=nvidia --gpus all -itd --rm \
  -v /path/to/your/hf/models:/workspace/hf/models \
  -v /path/to/data:/workspace/data \
  loongforge/loongforge:${LOONGFORGE_VERSION} /bin/bash
```

Once inside the container, navigate to `/workspace/LoongForge/examples/` and
launch the desired training script.

---

## Option B: Install from Source

Use this option if you already have a working CUDA + PyTorch environment and
want to set up LoongForge for development or training.

### Clone the repository

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge
```

### Install LoongForge

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[gpu]"
```

### Setup TransformerEngine (GPU only)

The `setup_env.py` script clones, patches, and compiles TransformerEngine:

```bash
python setup_env.py --te-tag v2.9
```

This script will automatically:

1. Clone `TransformerEngine` from the upstream NVIDIA repository.
2. Checkout the specified TE tag and create a local branch (`loongforge_<tag>`).
3. Apply patches from `patches/TransformerEngine_<tag>/` to TransformerEngine.
4. Compile and install `TransformerEngine`.

Tips: Some model architectures (e.g. DeepSeek-series) require additional compiled
dependencies such as DeepEP, DeepGEMM, FlashMLA, and Flash Attention that are
not included in the pip install. These are pre-built in the Docker image.
If you need them for a source install, refer to
[`docker/Dockerfile`](https://github.com/baidu-baige/LoongForge/blob/master/docker/Dockerfile)
for exact versions and build steps.

---

## Next Steps

Head over to the [LLM Pre-training](../llm_tutorial/quick_start_llm_pretrain.md) guide to launch your first training run.
