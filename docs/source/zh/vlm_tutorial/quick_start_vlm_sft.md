# 快速入门：VLM SFT

本文档将指导你在 LoongForge 框架下完成视觉语言模型（VLM）微调的快速入门流程。

## 0. 资源准备

在开始之前，请下载所需的模型权重、分词器和数据集。
所有资源通过 HuggingFace 下载。请先安装 CLI 工具：

```bash
pip install "huggingface_hub[cli]"
```

### 0.1 下载模型权重

```bash
hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir ./Qwen3-VL-30B-A3B-Instruct
```

> **注意：** 该模型约需 **62 GB** 磁盘空间（13 个 safetensor 分片）。

### 0.2 下载分词器

分词器已包含在上方下载的模型权重中（`./Qwen3-VL-30B-A3B-Instruct/`）。

### 0.3 下载数据集

我们使用 LLaVA-Instruct-Mix-VSFT-Small 数据集（约 109 MB，2,592 条样本，ShareGPT 格式的多模态图文对）进行 VLM SFT。

```bash
hf download axolotl-ai-co/llava-instruct-mix-vsft-small --repo-type dataset --local-dir ./data/llava-instruct-mix-vsft-small
```

下载后，按照本文档第 1.3 节的说明将数据集转换为 WebDataset 格式。

---

## 1. 数据准备

### 1.1 数据集配置与处理

在 VLM 指令微调场景中，使用**多模态 ShareGPT** 格式（包含 `messages` 和 `images`）。LoongForge 通过 LoongForge/configs/data/sft_dataset_config.yaml 解析此格式。以下是 **ShareGPT 格式示例**：

```yaml
multimodal:
  format: sharegpt
  columns:
    messages: messages
    images: images
  tags:
    role_tag: role
    content_tag: content
    user_tag: user
    assistant_tag: assistant
```

* **role_tag**：在 messages 列表中，表示"角色字段"的键名为 role。
* **content_tag**：在 messages 列表中，表示"内容字段"的键名为 content。
* **user_tag**：当角色字段值为 user 时，表示该消息来自用户。
* **assistant_tag**：当角色字段值为 assistant 时，表示该消息来自助手。

`tags` 告诉解析器消息结构中使用了哪些字段名以及角色值是什么。如果你的数据使用了不同的键名或角色值，应在此处相应地更新。

### 1.2 数据集参数说明

* `--data-path`：数据集路径（可指定多个路径进行混合训练）。
* `--sft-dataset-config`：数据集配置文件路径，默认为 sft_dataset_config.yaml。
* `--packing-sft-data`：启用在线 packing 模式
* `--packing-buffer-size`：Packing 批次大小，影响 packing 效率和内存使用

### 1.3 数据集预处理

转换为 **Energon 加载格式**的流程与预训练部分相同，请参阅 [快速入门：VLM 预训练](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html) 的第 1.2 节。框架提供了两种数据预处理方式：在线 packing 和离线 packing，分别说明如下：

* **在线 Packing**

在训练脚本的 DATA_ARGS 中启用：`--packing-sft-data`、`--packing-buffer-size` 以激活在线 packing 模式。此模式将多个较短的样本拼接为同一序列以提高 token 利用率。Packing 处理批次大小表示每次 packing 操作中处理的样本数量。较大的值通常能获得更好的 packing 效果，但预处理开销和内存使用也会更高。

* **离线 Packing**

提供了"离线序列打包"流水线：将**样本级别**的数据目录（每个样本一个 `json` + 若干媒体文件）按 `max_token_len` 分组重排，生成**打包的 WebDataset**（`pretrain-*.tar` + Energon 元数据）以提升训练吞吐量并减少填充。如需进一步了解离线 packing 详情，请参阅：[offline_data_packing.md](https://loongforge.readthedocs.io/en/latest/features/offline_data_packing.html)

## 2. 模型权重准备

此部分与预训练部分相同，请参阅 [快速入门：VLM 预训练](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html) 的第 2 节

## 3. 启动 SFT 训练

### 3.1 参数配置说明

在开源 Megatron 提供的参数基础上，LoongForge 添加了更便捷的训练启动参数。详细配置可在 loongforge/train/arguments.py 文件中找到。主要参数说明如下：

* `--training-phase sft`：显式启用 SFT 训练阶段。
* `--chat-template qwen2-vl`：指定 SFT 对话模板为 qwen2-vl，用于将多轮对话样本拼接为模型输入
* `+model.image_encoder.freeze=True`：通过 Hydra 配置覆盖，冻结图像编码器模型参数以进行训练

### 3.2 SFT 训练脚本

LoongForge 目前为各种模型提供了 SFT 训练示例脚本。进入容器后，可在 `examples/{model}/finetuning/` 目录下找到相关脚本。以下是使用 Qwen3_vl_30b_a3b SFT 训练脚本的示例：

```bash
#!/bin/bash
# 此脚本需要在至少 2 个节点上运行。

# 代码库根目录，添加到 PYTHONPATH。
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# 数据集根目录或 manifest 路径，由外部 dataloader 使用。
DATA_PATH=${DATA_PATH:-"/path/to/your/dataset"}

# TOKENIZER_PATH：HuggingFace 分词器目录，必须与模型词表匹配。
TOKENIZER_PATH=${TOKENIZER_PATH:-"/path/to/your/hf/tokenizer"}

# 权重加载和保存路径
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/path/to/your/mcore/checkpoint"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/path/to/your/mcore/checkpoint_save"}

# TensorBoard 日志目录，用于记录训练指标。
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/path/to/your/tensorboard"}

# 每个节点的 GPU 数量，由 torchrun 使用。
GPUS_PER_NODE=8

# 多节点配置修改
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

# 指定模型配置文件
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml

# 数据与分词器配置
DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
)

# 核心训练超参数
TRAINING_ARGS=(
    --training-phase sft
    --seq-length 32768
    --max-position-embeddings 262144
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 32
    --lr 6.0e-5
    --min-lr 6.0e-5
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 5000
    --eval-iters 0
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH_SAVE
    --save-interval 10000000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
)

# MoE 路由器与专家行为
MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
    --moe-token-dispatcher-type alltoall
)

# 并行与分布式训练
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
)

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME}
    )
fi

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

### 3.3 监控日志

脚本默认会将 TensorBoard 日志输出到 TENSORBOARD_PATH 指定的目录。你可以通过 TensorBoard 查看训练曲线。
