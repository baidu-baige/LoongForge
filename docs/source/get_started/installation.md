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
docker build --build-arg COMPILE_ENV=hopper --build-arg ENABLE_LEROBOT=false \
  -t loongforge:latest -f ./LoongForge/docker/Dockerfile .
```

| Build Arg | Description | Options |
|---|---|---|
| `COMPILE_ENV` | Target GPU architecture | `ampere`, `hopper`|
| `ENABLE_LEROBOT` | Enable LeRobot dependencies for VLA model training (e.g., Pi0.5, GR00T). Disabled by default due to dependency conflicts with the base environment. | `true`, `false` |

After the build finishes, verify:

```bash
docker images | grep loongforge
```

### Run the container

```bash
docker run --runtime=nvidia --gpus all -itd --rm \
  -v /path/to/your/hf/models:/mnt/cluster/huggingface.co/ \
  -v /path/to/data:/mnt/cluster/LoongForge/ \
  loongforge:latest /bin/bash
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

Head over to the [Quick Start](./README.md) guide to launch your first training run.
