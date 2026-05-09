# 快速入门：LLM SFT

本指南将带你完成在 LoongForge 框架中启动大语言模型（LLM）**SFT（监督微调）**任务的全过程。

---

## 0. 资源准备

在开始之前，请下载所需的模型权重、分词器和数据集。
所有资源通过 HuggingFace 下载。请先安装 CLI 工具：

```bash
pip install "huggingface_hub[cli]"
```

### 0.1 下载模型权重

```bash
hf download deepseek-ai/DeepSeek-V3.1 --local-dir ./deepseek-v3.1
```

> **注意：** 该模型约需 **1.37 TB** 磁盘空间（163 个 safetensor 分片，FP8 权重）。

### 0.2 下载分词器

分词器已包含在上方下载的模型权重中（`./deepseek-v3.1/`）。

### 0.3 下载数据集

我们使用 Alpaca-Cleaned 数据集（约 30 MB，51,742 条样本）进行 SFT。该数据集已与 LoongForge 内置的 `default` Alpaca 格式匹配。

```bash
hf download yahma/alpaca-cleaned --repo-type dataset --local-dir ./data/alpaca-cleaned
```

将数据集导出为 LoongForge 可读取的 JSON 文件：

```python
from datasets import load_dataset
import json

ds = load_dataset("yahma/alpaca-cleaned", split="train")
records = [{"instruction": item["instruction"], "input": item["input"], "output": item["output"]} for item in ds]
with open("./data/alpaca_cleaned.json", "w") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
```

导出后，在训练脚本中使用 `--data-path ./data/alpaca_cleaned.json --sft-dataset default`。

---

## 1. 准备数据

### 1.1 数据集格式与配置
在指令微调场景中，常见的对话格式有两种：**Alpaca 格式**和 **ShareGPT 格式**。
LoongForge 目前支持 **Alpaca 格式 JSON**，每行一个样本：

```json
[
  {
    "instruction": "User instruction",
    "input": "User question",
    "output": "Model answer"
  }
]
```

不同数据集的字段名可能不同，因此需要提供一个**数据集配置文件**来告诉加载器如何映射字段。
模板文件位于
`/workspace/LoongForge/configs/data/sft_dataset_config.yaml`。

#### (1) 文件格式
我们遵循 [LlamaFactory](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md) 使用的社区惯例。

#### (2) 默认配置
如果你的文件已使用标准 Alpaca 字段名，直接使用内置的 `default` 配置块即可：

```yaml
default:
  format: alpaca
  columns:
    prompt: instruction
    query: input
    response: output
```

#### (3) 添加自定义数据集
假设你的文件名为 `custom_dataset_name.json`，且包含额外字段：

```json
[
  {
    "instruction": "...",
    "input": "...",
    "output": "...",
    "system": "System prompt",
    "history": [
      ["Q1", "A1"],
      ["Q2", "A2"]
    ]
  }
]
```

在 YAML 文件中追加新的配置块：

```yaml
custom_dataset_name:
  format: alpaca
  columns:
    prompt: instruction
    query: input
    response: output
    system: system
    history: history
```

### 1.2 数据集参数
| 参数 | 含义 |
|---------|---------|
| `--data-path` | JSON 文件路径。支持**多个数据集**，使用冒号分隔采样权重：`path1:weight1,path2:weight2`。 |
| `--split` | 训练/验证/测试比例，例如 `--split 90,8,2`。 |
| `--sft-dataset-config` | 上述 YAML 配置文件路径。**默认值：** `configs/data/sft_dataset_config.yaml`。 |
| `--sft-dataset` | YAML 中数据集条目的名称。使用多个数据集时，必须与 `--data-path` 中的顺序**一一对应**。 |

### 1.3 预处理模式
LoongForge 支持**在线**和**离线**两种预处理模式。
当数据集较大时，建议选择**离线**模式，可将分词开销从关键路径中移除。

#### (1) 在线模式（默认）
```bash
--data-path /path/to/custom_dataset_name.json \
--sft-dataset custom_dataset_name \
--sft-data-streaming   # 可选：流式读取大文件
```

#### (2) 离线模式
运行一次辅助脚本：

```bash
#!/bin/bash
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=/path/to/hf/tokenizer
input_data=/path/to/custom_dataset_name.json
output_path=/path/to/save_dir

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  python ${LOONGFORGE_PATH}/tools/data_preprocess/llm/preprocess_sft_data.py \
      --input ${input_data} \
      --output ${output_path} \
      --seq-length 2048 \
      --chat-template ${chat_template} \
      --tokenizer-type HFTokenizer \
      --hf-tokenizer-path $TOKENIZER_PATH \
      --workers 50 \
      --split 100,0,0
      --packing-sft-data
```

关键参数说明
* `--seq-length` -- 最大 token 长度；超长样本将被截断。
* `--chat-template` -- 必须与训练时使用的模板一致。
* `--split` -- 预先划分数据；训练时仍需指定 `--split` 用于校验。
* `--packing-sft-data` -- 将多个样本打包到一个序列中（无填充）。

预处理完成后，将**目录**传递给训练：

```bash
--data-path /path/to/save_dir \
--is-tokenized-data \
--sft-dataset custom_dataset_name
```

---

## 2. 准备权重
与预训练相同 -- 请参阅 [快速入门：LLM 预训练](https://github.com/baidu-baige/LoongForge/tree/master/docs/source/llm_tutorial/quick_start_llm_pretrain.md)。

---

## 3. 启动 SFT 训练

### 3.1 LoongForge 额外参数（除原生 Megatron 参数外）
| 参数 | 用途 |
|---------|---------|
| `--training-phase sft` | 切换到微调阶段。 |
| `--chat-template` | 对话模板选择（`no-template`、`llama3`、`qwen` 等）。 |
| `--sft-dataset` / `--sft-train-dataset` / `--sft-valid-dataset` | YAML 文件中的数据集名称。 |
| `--packing-sft-data` | 启用样本打包以提高吞吐量。 |
| `--sft-data-streaming` | 流式读取大型 JSON 文件，而非全部加载到内存。 |
| `--sft-num-preprocess-workers` | 在线分词的 CPU 工作线程数。 |
| `--reduce-variable-seq-shape-p2p-comm` | 将 p2p 缓冲区填充为固定长度（SFT 推荐启用）。 |
| `--optimizer-cpu-offload` / `--optimizer-offload-fraction` | 将优化器状态卸载到 CPU 以节省 GPU 显存。 |

所有 FP8、流水线并行、重计算、MoE、MTP 相关参数与预训练保持一致。

### 3.2 SFT 脚本示例
开箱即用的脚本位于 `examples/{model}/finetuning/` 目录下。
以下是 DeepSeek-V3.1 的 FP8 SFT 脚本：

```bash
#!/bin/bash
# DeepSeek-V3 FP8 监督微调

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# ------------- 数据 -------------
DATA_PATH=/path/to/your/data
DATA_CACHE_PATH=/path/to/your/data/cache
DATASET_CONFIG_PATH=/path/to/your/dataset_config   # 可选

# ------------- 分词器与权重 -------------
TOKENIZER_PATH=/path/to/hf/tokenizer
CHECKPOINT_PATH=/path/to/mcore/checkpoint
CHECKPOINT_PATH_SAVE=/path/to/save_dir

# ------------- 日志 -------------
TENSORBOARD_PATH=/path/to/tensorboard

# ------------- FP8 量化 -------------
export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

GPUS_PER_NODE=8

# ------------- NCCL 与 NVSHMEM -------------
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
# 根据集群选择 HCA 列表
export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_IB_TRAFFIC_CLASS=130
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3

# ------------- Transformer Engine -------------
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# ------------- CUDA / PyTorch -------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------- 分布式 -------------
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

# ------------- 模型 -------------
MODEL_ARGS=(
  --model-name deepseek-v3
  --multi-latent-attention
  --rotary-base 10000
  --original-max-position-embeddings 4096
  --mscale 1.0
  --mscale-all-dim 1.0
  --norm-epsilon 1e-6
  --rotary-scaling-factor 40
  --enable-fa-within-mla
  --use-fp32-dtype-for-param-pattern expert_bias
)

# ------------- 数据加载 -------------
DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 90,8,2
)

# ------------- SFT 专属参数 -------------
SFT_ARGS=(
  --chat-template no-template
  --sft-num-preprocess-workers 16
  --no-check-for-nan-in-loss-and-grad
  --packing-sft-data
  --sft-dataset sharegpt
)

# ------------- 训练超参数 -------------
TRAINING_ARGS=(
  --training-phase sft
  --seq-length 65536
  --max-position-embeddings 163840
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 128
  --lr 1e-06
  --train-iters 1500
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --min-lr 1.0e-7
  --weight-decay 0.1
  --lr-warmup-fraction 0.002
  --clip-grad 1.0
  --bf16
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_PATH_SAVE
  --save-interval 1000
  --eval-interval 10
  --eval-iters 1
  --no-load-optim
  --no-load-rng
  --recompute-granularity full
  --recompute-method block
  --custom-pipeline-layers 8,7,8,8,8,8,8,6
  --custom-pipeline-recompute-layers 8,7,8,8,8,8,8,6
  --num-virtual-stages-per-pipeline-rank 2
  --reduce-variable-seq-shape-p2p-comm
  --fp8-format e4m3
  --fp8-recipe blockwise
  --fp8-param-gather
  --enable-fp8-comm
  --distributed-timeout-minutes 60
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  --enable-experimental
)

# ------------- MoE -------------
MOE_ARGS=(
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk 8
  --moe-aux-loss-coeff 1e-3
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0.001
  --moe-router-num-groups 8
  --moe-router-group-topk 4
  --moe-router-score-function sigmoid
  --moe-router-topk-scaling-factor 2.5
  --moe-router-dtype fp32
  --empty-unused-memory-level 2
)

# ------------- 并行与优化器 -------------
MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 8
  --expert-model-parallel-size 32
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --moe-token-dispatcher-type flex
  --moe-enable-deepep
  --use-precision-aware-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --use-distributed-optimizer
  --moe-permute-fusion
  --cross-entropy-loss-fusion
  --overlap-grad-reduce
  --overlap-param-gather
)

# ------------- MTP -------------
MTP_ARGS=(
  --mtp-loss-scaling-factor 0.1
)

# ------------- 日志 -------------
LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --check-weight-hash-across-dp-replicas-interval 30
)

# ------------- 启动 -------------
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $LOONGFORGE_PATH/loongforge/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${SFT_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  ${MTP_ARGS[@]}
```

---

## 4. 监控
TensorBoard 日志将写入 `TENSORBOARD_PATH` 指定的目录。
打开 TensorBoard 即可查看损失、困惑度、内存、吞吐量等指标。
