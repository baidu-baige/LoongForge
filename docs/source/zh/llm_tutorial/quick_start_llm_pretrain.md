# 快速入门：LLM 预训练

本指南将带你完成在 LoongForge 框架中启动大语言模型（LLM）预训练任务的全过程。

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

> **注意：** 该模型约需 **1.37 TB** 磁盘空间（163 个 safetensor 分片，FP8 权重）。下载时间取决于网络状况。

### 0.2 下载分词器

分词器已包含在上方下载的模型权重中（`./deepseek-v3.1/`）。

### 0.3 下载数据集

快速验证可使用 WikiText-103-raw 数据集（约 180 MB）。正式预训练请准备自己的大规模语料。

```bash
hf download wikitext --repo-type dataset --include "wikitext-103-raw-v1*" --local-dir ./data/wikitext
```

原始 wikitext 数据需转换为换行分隔的 JSON 格式后才能进行预处理。运行：

```python
from datasets import load_dataset
import json

ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
with open("./data/wikitext_train.jsonl", "w") as f:
    for item in ds:
        text = item["text"].strip()
        if text:
            f.write(json.dumps({"text": text}) + "\n")
```

转换完成后，将 `./data/wikitext_train.jsonl` 传入第 1.1 节的预处理工具。

---

## 1. 准备数据

### 1.1 数据预处理
在训练之前，通常需要将大量原始语料转换为能够最大化训练速度的格式。

1. 将语料组织为**换行分隔的 JSON** 格式，每行一个文档：

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

2. 启动容器并运行内置工具：

```bash
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=/path/to/your/tokenizer
input_data=/path/to/your/json
output_prefix=/path/to/your/output_prefix

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  python ${LOONGFORGE_PATH}/tools/data_preprocess/llm/preprocess_pretrain_data.py \
      --input ${input_data} \
      --output-prefix ${output_prefix} \
      --tokenizer-type HFTokenizer \
      --hf-tokenizer-path $TOKENIZER_PATH \
      --json-keys text \
      --workers 50 \
      --append-eod
```

各模型系列的示例脚本可在 `examples/{model}/pretrain/` 目录下找到。

---

## 2. 准备权重

训练通常从开源的 HuggingFace 权重开始。
请先下载权重，然后将其转换为框架所需的 Megatron-Core 格式。

### 2.1 下载 HuggingFace 权重
以 DeepSeek-V3.1 为例：
https://huggingface.co/deepseek-ai/DeepSeek-V3.1

### 2.2 转换权重格式
LoongForge 提供了统一的转换工具 `tools/convert_checkpoint`。
以下将原始 FP8 HuggingFace 权重转换为 MCore FP8 格式：

```bash
#!/bin/bash
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/path/to/hf_checkpoint          # FP8 HuggingFace 权重
SAVE=/path/to/your/save              # 转换后的 MCore FP8 权重

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/deepseek3/deepseek_v3.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/deepseek3/ckpt_convert/deepseek_v3_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --expert_parallel_size=32 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --custom_pipeline_layers 8,7,8,8,8,8,8,6 \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    --amax_epsilon=1e-12
```

关键参数说明：
* `--amax_epsilon` -- FP8 量化缩放因子；需与训练中使用的 FP8_EPS 环境变量保持一致。
* `--custom_pipeline_layers` -- 每个流水线并行阶段的层数。
  如果启用了 vPP（`--num-virtual-stages-per-pipeline-rank`），则按顺序列出每个 vPP chunk 的层数，例如
  `--custom_pipeline_layers 4,3,4,4,4,4,4,3,4,4,4,4,4,4,4,3` 表示 2 个虚拟阶段。

完整详情请参阅 [llm_ckpt_convert.md](https://github.com/baidu-baige/LoongForge/tree/master/docs/source/llm_tutorial/llm_ckpt_convert.md)。

---

## 3. 启动预训练

### 3.1 LoongForge 提供的额外参数
除原生 Megatron 参数外，框架还添加了便捷选项（定义在 `loongforge/train/arguments.py` 中）：

* `--config-file` -- 包含所有模型超参数的 YAML 文件路径，例如 `configs/models/deepseek3/deepseek_v3.yaml`。
* `--model-name` -- 模型简短名称，如 `deepseek-v3`；系统会自动查找对应的 YAML 文件。
* `--training-phase` -- 训练阶段，如 `pretrain`、`sft` 等。
* `--tokenizer-type` -- 推荐使用 `HFTokenizer` 并配合 `--hf-tokenizer-path`。
* `--no-create-attention-mask-in-dataloader` -- 跳过注意力掩码创建以加速数据加载。
* `--custom-pipeline-layers` -- 每阶段的层分配，例如 `19,20,20,21`。
* `--custom-pipeline-recompute-layers` -- 每阶段的重计算层数，例如 `10,11,12,13`。
* `--reduce-variable-seq-shape-p2p-comm` -- 将 p2p 缓冲区填充为固定长度（适用于 SFT）。
* `--use-fp32-dtype-for-param-pattern` -- 将特定参数保持在 FP32 精度。

### 3.2 预训练脚本示例
开箱即用的脚本位于 `examples/{model}/pretrain/` 目录下。
以下是 DeepSeek-V3.1 的 FP8 预训练脚本（已添加注释说明）：

```bash
#!/bin/bash
# DeepSeek-V3 FP8 混合精度预训练

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# ------------- 数据 -------------
DATA_PATH=/path/to/your/dataset

# ------------- 分词器与权重 -------------
TOKENIZER_PATH=/path/to/your/hf/tokenizer
CHECKPOINT_PATH=/path/to/your/mcore/checkpoint
CHECKPOINT_PATH_SAVE=/path/to/your/save_dir

# ------------- 日志 -------------
TENSORBOARD_PATH=/path/to/your/tensorboard

# ------------- FP8 量化 -------------
export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

GPUS_PER_NODE=8

# ------------- NCCL 与 NVSHMEM -------------
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
# 根据集群选择 HCA 列表
export NVSHMEM_HCA_LIST=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
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
  --split 99990,8,2
  --no-create-attention-mask-in-dataloader
)

# ------------- 训练超参数 -------------
TRAINING_ARGS=(
  --training-phase pretrain
  --seq-length 32768
  --max-position-embeddings 163840
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 1024
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
  --save-interval 100
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
  --distributed-timeout-minutes 60
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
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  ${MTP_ARGS[@]}
```

---

## 4. 监控

脚本会将 TensorBoard 日志写入 `TENSORBOARD_PATH` 指定的目录。
启动 TensorBoard 并在浏览器中打开，即可查看损失曲线、吞吐量、内存使用等指标。
