# 快速开始：Pi0.5 训练

本文档将引导您完成 LoongForge 框架下 **Pi0.5** SFT（监督微调）训练的快速开始流程。

## 0. 资源准备

在开始之前，请下载所需的模型权重、分词器和数据集。
所有资源通过 HuggingFace 下载。请先安装 CLI 工具：

```bash
pip install "huggingface_hub[cli]"
```

### 0.1 下载模型权重

```bash
hf download lerobot/pi05_base --local-dir ./pi05_base
```

> **注意：** 该模型约需 **14.5 GB** 磁盘空间。

### 0.2 下载分词器

分词器需单独下载。**受限访问**——您需先登录 HuggingFace 并在 https://huggingface.co/google/paligemma-3b-pt-224 接受许可协议后才能下载。

```bash
hf login  # 使用你的 HF token 登录
hf download google/paligemma-3b-pt-224 --local-dir ./paligemma-3b-pt-224
```

### 0.3 下载数据集

我们使用 LeRobot v3.0 格式的 Libero-10 机器人操作数据集进行训练。

```bash
hf download lerobot/libero_10 --repo-type dataset --local-dir ./data/libero_10
```

下载后，在训练脚本中将 `--data-path` 指向 `./data/libero_10`。

---

## 1. 数据准备

### 1.1 数据集格式

VLA 训练使用机器人操作轨迹数据。每个样本通常包含多模态观测（图像、语言指令）和动作序列。LoongForge 要求数据按照 **[LeRobot dataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)** 格式组织，其中通过 `--data-path` 传入的路径指向数据集的根目录。

LeRobot dataset v3.0 将片段（episodes）存储为 Parquet 文件，包含标准化的观测、动作和元数据字段。目录结构通常如下：

### 1.2 数据集参数说明

* `--data-path`：数据集根目录路径。
* `--chat-template empty`：VLA 模型使用 `empty` 聊天模板，因为动作预测不依赖对话提示模板。
* `--split 100,0,0`：训练/验证/测试的比例划分。通常将所有数据用于训练。
* `--num-workers`：数据加载工作进程数（默认 16）。


## 2. 模型权重准备

### 2.1 将 HF 权重转换为 torch 格式

将 HuggingFace 权重转换为 PyTorch 格式以供 LoongForge 训练使用：

```bash
# 设置输入/输出路径
export LOAD=/path/to/pi05_huggingface/
export SAVE=/path/to/pi05_torch/

sh examples/pi05/checkpoint_convert/convert_pi05_hf_to_torch.sh
```

转换后，权重目录结构如下：

```
pi05_torch/
├── latest_checkpointed_iteration.txt
└── release
    └── mp_rank_00
        └── model_optim_rng.pt
```

## 3. 启动 SFT（监督微调）训练

### 3.1 参数配置说明

在支持开源 Megatron 参数的基础上，LoongForge 新增了更便捷的训练启动参数。详细配置可在 `loongforge/train/arguments.py` 文件中找到。Pi0.5 的关键参数如下：

**模型与并行：**

* `--model-name pi05`：选择 Pi0.5 模型系列，映射到 `configs/models/pi05/pi05.yaml`。


* `--training-phase sft`：显式启用 SFT（监督微调）训练阶段。
* `--ckpt-format torch`：与 FSDP DTensor 分片兼容的权重格式。
* `--finetune`：表示本次为微调运行（除非另行覆盖，否则会重置加载权重中的优化器/调度器状态）。
* `--no-load-optim` / `--no-load-rng`：不从权重恢复优化器或 RNG 状态，从头开始。

* `export CUDA_DEVICE_MAX_CONNECTIONS=8`：FSDP 所需；避免因连接数限制导致的死锁。
* `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：启用可扩展的 CUDA 内存段，减少训练期间的 OOM 错误。

### 3.2 SFT（监督微调）训练脚本

完整的 Pi0.5 SFT（监督微调）训练脚本位于 [examples/pi05/finetuning/sft_pi05.sh](https://github.com/baidu-baige/LoongForge/tree/master/examples/pi05/finetuning/sft_pi05.sh)。以下是带注释的版本：

```bash
#!/usr/bin/env bash


# ── 路径配置 ─────────────────────────────────────────────────────
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/pi05/"}

# ── 环境变量 ───────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=8   # FSDP 所需
export USE_BF16_BUFFER=false           # DTensor 不支持 BF16 缓冲区
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 降低 OOM 风险

# ── 分布式启动（默认单节点）────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── 数据与分词器 ────────────────────────────────────────────────────
DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty        # VLA 不使用对话提示模板
  --num-workers 16
)

# ── 核心训练超参数 ───────────────────────────────────────────────────
TRAINING_ARGS=(
    --training-phase sft
    --micro-batch-size 12
    --global-batch-size 96
    --train-iters 30000
    --seq-length 762
    --max-position-embeddings 762
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format torch
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
    --seed 1234
    --lr 2.5e-8
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --lr-decay-iters 30000
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-eps 1e-8
    --adam-beta2 0.95
    --weight-decay 0.01
    --finetune
    --bf16
    --use-precision-aware-optimizer
    --exp-avg-dtype fp32
    --exp-avg-sq-dtype bf16
    --num-distributed-optimizer-instances 1
    --save $CHECKPOINT_PATH
    --save-interval 30000
)

# ── 模型与分布式后端 ─────────────────────────────────────────────────────
MODEL_CONFIG_ARGS=(
    --model-name pi05
    --use-distributed-optimizer
    --distributed-backend nccl
)

# ── 日志 ─────────────────────────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
)

# ── 启动 ───────────────────────────────────────────────────────────────────
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```
