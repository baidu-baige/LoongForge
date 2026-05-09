# 快速开始：昆仑芯 P800 上 VLA 模型 SFT（监督微调）训练

## 快速开始：VLA 模型 SFT（监督微调）训练

本文档引导您完成在 P800 上使用 LoongForge 框架对视觉语言动作模型（VLA）进行 SFT（监督微调）的快速开始流程。

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

## 1. 权重转换

在第 0 节下载资源后，训练前需将 HF 权重转换为 torch 格式。此步骤与 GPU 版本相同：

* **权重转换**：将第 0.1 节下载的 HuggingFace 权重转换为 PyTorch 格式——参见 [快速开始：Pi0.5 训练](https://loongforge.readthedocs.io/en/latest/vla_tutorial/quick_start_pi05_training.html)第 2.1 节。

## 2. SFT（监督微调）训练脚本

LoongForge 目前提供了多种模型的 SFT（监督微调）训练示例脚本。进入容器后，您可以在 `examples_xpu/{model}/finetuning/` 目录下找到相关脚本。以下是 `PI 0.5` 的 SFT（监督微调）训练脚本示例。请参考注释了解各部分脚本的作用：

```bash
#!/usr/bin/env bash
# Pi05 验证性 SFT 启动脚本。此脚本利用轻量级 pi05 训练器
#（虚拟数据，单次前向/反向）验证 Omni
# 框架内部的连接。如果您的仓库布局不同，请调整路径。

set -euo pipefail

# 路径
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}

export XMLIR_ENABLE_FAST_FC=true         # 用于 torch.nn.linear.py（如 LinearWithActFunction）
export XMLIR_MATMUL_FAST_MODE=1          # BF16 精度下 XBLAS 全连接（FC）计算的累加加速
export XMLIR_ENABLE_LINEAR_FC_FUSION=1   # 允许线性层在特定场景下绕过 XBLAS FC 融合（例如使用 addmm），默认值为 1
export XMLIR_PARALLEL_SAVE_MEMORY=false  # 设为 false 时显存占用增加但性能提升；设为 true 时显存占用减少但性能下降
export XDNN_USE_FAST_GELU=true           # GELU 算子的高精度实现
export BKCL_FORCE_SYNC=1                 # 通信前强制 CPU 同步（会降低性能）
export BKCL_TREE_THRESHOLD=0             # 设为 0 以禁用树算法
export BKCL_ENABLE_XDR=1                 # 启用 XDR（XPU Direct RDMA）和直接 RDMA。流量将直接从 XPU 流向 RDMA 网卡，多节点训练时必须启用
export BKCL_RDMA_VERBS=1                 # 与 BKCL_QPS_PER_CONNECTION 配合使用；目前仅海光服务器需要
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4   # 以实际情况为准；多节点训练时请根据服务器环境的网卡连接进行配置
export XTE_USE_MULTI_TENSOR_ADAMW=True   # 使 Adam 优化器与 GPU multi_tensor_adamw 实现对齐

# 分布式启动（默认单节点）
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
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

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty
  --num-workers 16
)

# 核心训练参数 — pi05 训练器仅需最少的 Megatron 参数
TRAINING_ARGS=(
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim
    --training-phase sft
    --micro-batch-size 16
    --global-batch-size 128
    --train-iters 30000
    --seq-length 762
    --max-position-embeddings 762
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format fsdp_dtensor
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
    --no-strict-fsdp-dtensor-load
    --finetune
    --bf16
    --grad-reduce-in-bf16
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --num-distributed-optimizer-instances 1
    --save $CHECKPOINT_PATH
    --save-interval 30000
)

MODEL_CONFIG_ARGS=(
    --model-name pi05
    --use-distributed-optimizer
    --distributed-backend nccl
    #--random-fallback-cpu
)

LOGGING_ARGS=(
    --log-interval 1
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}

```

## 监控日志

默认情况下，脚本将 TensorBoard 日志输出到 `TENSORBOARD_PATH` 指定的目录。您可以通过 TensorBoard 查看训练曲线。

此外，如果安装了 wandb，您可以配置 `WANDB_API_KEY` 将训练指标上传到 wandb 进行在线监控。
