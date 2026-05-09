# 快速开始：Wan 模型训练
本节将完整介绍 **Wan 预训练** 流程。

---
## Wan2.2 I2V-A14B 训练流程

### 0. 资源准备

在开始之前，请下载所需的模型权重、分词器和数据集。
所有资源通过 HuggingFace 下载。请先安装 CLI 工具：

```bash
pip install "huggingface_hub[cli]"
```

#### 0.1 下载模型权重

```bash
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan-AI/Wan2.2-I2V-A14B
```

> **注意：** 该模型约需 **126 GB** 磁盘空间（高噪声模型约 57 GB + 低噪声模型约 57 GB + T5 编码器约 11.4 GB + VAE 约 0.5 GB）。下载时间取决于网络状况。

#### 0.2 下载分词器

UMT5 分词器已包含在上方下载的模型权重中（`./Wan-AI/Wan2.2-I2V-A14B/google/umt5-xxl/`）。

#### 0.3 准备数据集

目前没有标准的公开视频数据集可用于快速验证。请按照第 1 节的 `metadata.csv` 格式准备自己的视频数据。以下是最小测试示例：

```bash
mkdir -p ./data/dataset/train
# 将 .mp4 文件放置在 ./data/dataset/train/ 中

cat > ./data/dataset/metadata.csv << 'EOF'
video,prompt
train/sample.mp4,"示例视频描述"
EOF
```

---

### 1. 预处理训练数据
#### 数据集示例
```text
dataset
├── metadata.csv
└── train
    ├── EGO_1.mp4
    ├── EGO_2.mp4
    ├── EGO_3.mp4
```
metadata.csv 示例

```text
video,prompt
train/EGO_1.mp4,"places the bag of clothes on the floor\nPlan:\n pick up the bag of clothes. Put the bag of clothes on the floor.\nactions :\n1. pick up(bag of clothes)\n2. put on(bag of clothes, floor)"
```

#### 步骤

**步骤 1** 安装依赖（模型权重已在第 0.1 节中下载）

```bash
pip install diffsynth==1.1.8
```

**步骤 2** 处理输入

```bash
MODEL_BASE=./Wan-AI/Wan2.2-I2V-A14B  # 应与第 0.1 节中的 --local-dir 一致
MODEL_T5=${MODEL_BASE}/models_t5_umt5-xxl-enc-bf16.pth
MODEL_VAE=${MODEL_BASE}/Wan2.1_VAE.pth
# 脚本位置：LoongForge 仓库中的 examples/wan/wan_preprocess.py
accelerate launch wan_preprocess.py \
  --dataset_base_path <your_dataset> \
  --dataset_metadata_path <your_dataset>/metadata.csv \
  --height 480 --width 832 --num_frames 49 \
  --model_paths "${MODEL_T5},${MODEL_VAE}" \
  --tokenizer_local_path "${MODEL_BASE}/google/umt5-xxl" \
  --output_path ./data/preprocessed \
  --max_timestep_boundary 0.358 --min_timestep_boundary 0

```

#### 输出
每个 `.pth` 文件包含以下三个键：
- `input_latents` – 整个视频的 VAE 隐变量
- `y` – 首帧 VAE 隐变量与可见性掩码拼接
- `context` – 文本编码器嵌入

（高/低噪声张量**未**分离；LoongForge 会在后续在线添加噪声。）

---

### 2. 权重转换（HF → Megatron）

在 **LoongForge** 仓库中：

**步骤 1** 生成具有正确 PP 划分的**随机 Megatron 权重**（用作脚手架）。
- 选择一个空文件夹，例如 `<base>/wan2.2/hg2mcore_pp4/high_noise/Megatron_Random`
- 在 `examples/wan/pretrain_wan2.2_i2v_a14b.sh` 中设置
  - `HIGH_NOISE_CHECKPOINT_PATH` → 上述文件夹
  - `LOW_NOISE_CHECKPOINT_PATH` → 对应的低噪声文件夹
  - `--train-iters 5`
  - `--save-interval 2`
- 运行一次——你将获得 `iter_0000002` 文件夹。

**步骤 2** 将 HF 权重转换为 Megatron 格式
编辑 `examples/wan/convert_wan2.2.sh`（`hg2mcore` 部分）：
- `--load_path` → 步骤 1 中生成的 `iter_0000002`
- `--save_path` → 最终发布文件夹，例如 `<base>/high_noise/Megatron_Release/`
- `--checkpoint_path` → 原始 HF `.safetensors` 目录
- `--pp 4`（或 8）

运行
```bash
bash examples/wan/convert_wan2.2.sh hg2mcore
```
对低噪声模型重复上述操作。

---

### 3. 启动训练

**推荐单节点配置**：PP=4, CP=2
多节点——通过**数据并行**扩展：
```text
dp = (NNODES × GPUS_PER_NODE) / (pp × cp)
```

| 符号 | 含义 |
|---|---|
| `dp` | 数据并行度 |
| `pp` | 流水线并行度 |
| `cp` | 上下文并行度 |

**步骤 1** 调整 `examples/wan/pretrain_wan2.2_i2v_a14b.sh`
- `HIGH_NOISE_CHECKPOINT_PATH` → 高噪声 Megatron 权重路径（来自第 2 节）
- `LOW_NOISE_CHECKPOINT_PATH` → 低噪声 Megatron 权重路径（来自第 2 节）
- `DATASET_PATH` → 第 1 节的输出路径（例如 `./data/preprocessed`）
- `--pipeline-model-parallel-size 4`
- `--context-parallel-size 2`
- `--context-parallel-ulysses-degree 2`

**步骤 2** 启动
- 单节点：
  ```bash
  bash examples/wan/pretrain_wan2.2_i2v_a14b.sh
  ```
- 多节点：在每个节点上执行相同脚本——集群环境变量（`MASTER_ADDR`、`NODE_RANK` 等）会自动读取。

---

### 4. 导出权重（Megatron → HF）

编辑 `examples/wan/convert_wan2.2.sh`（`mcore2hg` 部分）：
- `--load_path` → 训练后的 Megatron 权重
- `--save_path` → 目标 HF 文件夹
- `--checkpoint_path` → 原始 HF 权重目录（仅用于读取模型结构）
- `--pp 4`

运行
```bash
bash examples/wan/convert_wan2.2.sh mcore2hg
```
