# 安装

## 系统要求

### 硬件

- **必需**：NVIDIA GPU（Ampere / Hopper 或更新架构）
- **NVIDIA 驱动**：版本须满足 CUDA Toolkit 要求

### 软件

- **Python**：>= 3.10
- **PyTorch**：>= 2.6.0
- **CUDA Toolkit**：>= 12.1
- **操作系统**：Linux（推荐 Ubuntu 22.04 / 24.04）

注意：昆仑 XPU 安装请参见[昆仑安装指南](../kunlun_tutorial/install_p800.md)。

## 前置条件

安装 [uv](https://docs.astral.sh/uv/)，一个快速的 Python 包安装和解析工具：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 依赖概览

LoongForge 使用两种不同策略管理其关键上游依赖：

| 依赖 | 策略 | 位置 |
|---|---|---|
| **Megatron-LM** | git 子模块（LoongForge fork） | `third_party/Loong-Megatron/` |
| **TransformerEngine** | 对上游 NVIDIA tag 的补丁 | `patches/TransformerEngine_<tag>/` |

**Megatron-LM** 通过 git 子模块固定到
[Loong-Megatron](https://github.com/baidu-baige/Loong-Megatron) fork 的特定提交。所有 LoongForge 特有的改动直接存在于 fork 分支中，不应用补丁。

**TransformerEngine** 从上游 NVIDIA 仓库克隆，检出指定的社区 tag，然后应用 LoongForge 特有的修复补丁。补丁目录后缀与其目标的上游 tag 匹配（例如 `patches/TransformerEngine_v2.9/`）。

---

## 方式 A：Docker 镜像（推荐）

如果您想要完全可复现、即装即训的环境，无需手动管理依赖，请使用此方式。

### 前置条件

- Docker >= 20.10
- nvidia-container-toolkit

### 构建镜像

构建前，请带子模块克隆仓库，以便 Loong-Megatron 源码包含在 Docker 构建上下文中：

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
```

然后构建镜像：

```bash
docker build --build-arg COMPILE_ENV=hopper --build-arg ENABLE_LEROBOT=false \
  -t loongforge:latest -f ./LoongForge/docker/Dockerfile .
```

| 构建参数 | 描述 | 选项 |
|---|---|---|
| `COMPILE_ENV` | 目标 GPU 架构 | `ampere`, `hopper`|
| `ENABLE_LEROBOT` | 启用 VLA 模型训练（如 Pi0.5, GR00T）的 LeRobot 依赖。由于与基础环境存在依赖冲突，默认禁用。 | `true`, `false` |

构建完成后，验证：

```bash
docker images | grep loongforge
```

### 运行容器

```bash
docker run --runtime=nvidia --gpus all -itd --rm \
  -v /path/to/your/hf/models:/mnt/cluster/huggingface.co/ \
  -v /path/to/data:/mnt/cluster/LoongForge/ \
  loongforge:latest /bin/bash
```

进入容器后，导航到 `/workspace/LoongForge/examples/` 并启动所需的训练脚本。

---

## 方式 B：源码安装

如果您已有可用的 CUDA + PyTorch 环境，并希望为开发或训练搭建 LoongForge，请使用此方式。

### 克隆仓库

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
cd LoongForge
```

### 安装 LoongForge

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[gpu]"
```

### 设置 TransformerEngine（仅 GPU）

`setup_env.py` 脚本会克隆、打补丁并编译 TransformerEngine：

```bash
python setup_env.py --te-tag v2.9
```

此脚本将自动：

1. 从上游 NVIDIA 仓库克隆 `TransformerEngine`。
2. 检出指定的 TE tag 并创建本地分支（`loongforge_<tag>`）。
3. 将 `patches/TransformerEngine_<tag>/` 中的补丁应用到 TransformerEngine。
4. 编译并安装 `TransformerEngine`。

提示：某些模型架构（如 DeepSeek 系列）需要额外的编译依赖，如 DeepEP、DeepGEMM、FlashMLA 和 Flash Attention，这些未包含在 pip 安装中。它们已预构建在 Docker 镜像中。如果源码安装需要这些依赖，请参考 [`docker/Dockerfile`](https://github.com/baidu-baige/LoongForge/blob/master/docker/Dockerfile) 获取确切版本和构建步骤。

---

## 下一步

前往[LLM 预训练](../llm_tutorial/quick_start_llm_pretrain.md)指南启动您的第一次训练。
