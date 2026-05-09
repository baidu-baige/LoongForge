# 快速开始：昆仑芯 P800 上 LLM 模型 SFT（监督微调）训练

## 快速开始：LLM 模型 SFT（监督微调）训练

本文档引导您完成在 P800 上使用 LoongForge 框架对大语言模型（LLM）进行 SFT（监督微调）的快速开始流程。

## 0. 资源准备

在开始之前，请下载所需的模型权重、分词器和数据集。
所有资源通过 HuggingFace 下载。请先安装 CLI 工具：

```bash
pip install "huggingface_hub[cli]"
```

### 0.1 下载模型权重

```bash
hf download Qwen/Qwen3-8B --local-dir ./Qwen3-8B
```

> **注意：** 该模型约需 **16 GB** 磁盘空间。下载时间取决于网络状况。

### 0.2 下载分词器

分词器已包含在上方下载的模型权重中（`./Qwen3-8B/`）。

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

## 1. 数据准备与权重转换

在第 0 节下载资源后，训练前需进行数据准备和权重转换。这些步骤与 GPU 版本相同：

* **数据准备**：格式化 Alpaca 数据集并配置 SFT 数据加载器——参见 [快速开始：LLM SFT](https://loongforge.readthedocs.io/en/latest/llm_tutorial/quick_start_llm_sft.html)第 1.1–1.3 节。
* **权重转换**：将第 0.1 节下载的 HF 权重转换为 Megatron-Core 格式——参见 [快速开始：LLM 预训练](https://loongforge.readthedocs.io/en/latest/llm_tutorial/quick_start_llm_pretrain.html)第 2 节。

## 2. SFT（监督微调）训练脚本

LoongForge 目前提供了多种模型的 SFT（监督微调）训练示例脚本。进入容器后，您可以在 `examples_xpu/{model}/finetuning/` 目录下找到相关脚本。以下是 `Qwen3-8B` 的 SFT（监督微调）训练脚本示例。请参考注释了解各部分脚本的作用：

```bash
# ! /bin/bash
# 此脚本需要在至少 2 个节点上运行。

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/dataset/sft_aplaca_zh_data.json"}

DATA_PATH=${DATA_PATH:-"/mnt/cluster/LoongForge/qwen3/sft_aplaca_zh_tokenized"}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"/mnt/cluster/LoongForge/qwen3/sft_aplaca_zh_data_cache"}

DATASET_CONFIG_PATH=${DATASET_CONFIG_PATH:-"/workspace/LoongForge/configs/data/sft_dataset_config.yaml"}

TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/cluster/huggingface.co/Qwen/Qwen3-8B"}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/qwen3/Qwen3_8B_mcore_tp1pp1"}

TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/qwen3-8b-sft"}

GPUS_PER_NODE=8

###################### 昆仑芯 P800 ######################
# bf16 专用（Megatron 相关变量请参考 <Loong Megatron 专用>）
export XMLIR_ENABLE_FAST_FC=true         # 用于 torch.nn.linear.py（LinearWithActFunction 等）
#export XMLIR_ENABLE_FAST_FC_FWD_OUT=true # 前向
#export XMLIR_ENABLE_FAST_FC_BWD_DW=true  # 反向 dw
#export XMLIR_ENABLE_FAST_FC_BWD_DX=true  # 反向 dx
export FORCE_DISABLE_INPLACE_BF16_CAST=false    # 默认为 false，特殊情况下需启用（异步权重）

export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4" # 多节点时使用，根据实际网络连接调整
export BKCL_SOCKET_IFNAME=eth0                  # 根据实际环境调整，默认禁用，找不到网卡时需指定
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0                     # 设置为 1 可能导致空间不足时 OOM
export BKCL_ENABLE_XDR=1
export BKCL_ALL_TO_ALL_OPT=1                    # 多节点 alltoall 开关
export BKCL_RING_HOSTID_USE_RANK=1              # 从 1.2.11 版本开始支持，未来将成为默认值

export XMLIR_PARALLEL_SAVE_MEMORY=false         # false：内存占用更多但性能更好；true：内存占用减少但性能下降
export XMLIR_BATCH_PARALLEL=false               # 启用通信融合算子，bf16 下 USE_CAST_FC_FUSION 自动禁用
export XMLIR_ENABLE_FAST_FC=true
export SAVE_LOG_FILE_WITH_RANK_ID=false         # 设为 true 时，训练日志将按 rank_id 分别存储
export XMLIR_LOG_PATH="log-path"                # 指定训练日志存储目录
export XMLIR_LOG_PREFIX="log-file-prefix"       # 指定训练日志文件名前缀
export P800_DEBUG=false                         # 设为 true 时，梯度范数变为 nan 将保存权重并退出
export P800_DUMP_DIR="ckpt-dump-dir-path"       # 指定梯度范数变为 nan 时权重和信息的转储目录
export XMLIR_DIST_ASYNC_ISEND_IRECV=true        # true：send/recv 使用异步逻辑，默认为同步
export XMLIR_CUDNN_ENABLED=1                    # true：使用 cuDNN，支持 conv3d 等；false：禁用 cuDNN

# LINEAR 开关
export XMLIR_ENABLE_LINEAR_FC_FUSION=1          # 允许 linear 在特定场景下绕过 xblas fcfusion，例如使用 addmm，默认为 1
export XDNN_FC_GEMM_DTYPE=int32_with_ll         # GEMM_DTYPE 使用 int32_with_ll，可选
export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1         # 启用 xpu_plugin 以获得更好性能（推荐）

XFLAGS --disable transformer_engine_1_7
XFLAGS --disable transformer_engine_1_13
######################################################

# 多节点配置
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

MODEL_ARGS=(
    --model-name qwen3-8b
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 100,0,0
)

SFT_ARGS=(
    --chat-template qwen
    --sft-num-preprocess-workers 16
    --no-check-for-nan-in-loss-and-grad
    --is-tokenized-data
    #--packing-sft-data
    #--sft-data-streaming

    #--train-on-prompt
    #--eod-mask-loss

    #--sft-dataset-config $DATASET_CONFIG_PATH
    #--sft-dataset sft_aplaca_zh_data # 定义在 --sft-dataset-config 中，默认值：default
    #--data-cache-path $DATA_CACHE_PATH
)

TRAINING_ARGS=(
    --training-phase sft # 可选值：pretrain, sft
    --seq-length 4096
    --max-position-embeddings 32768
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 5000
    --lr-decay-iters 5000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --save-interval 500
    --eval-interval 100
    --eval-iters 10
    #--ckpt-step 0
    --no-load-optim
    --no-load-rng
    #--num-workers 8
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    #--overlap-grad-reduce
    #--overlap-param-gather
    --distributed-backend nccl
    --sequence-parallel
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
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

## 监控日志

默认情况下，脚本将 TensorBoard 日志输出到 `TENSORBOARD_PATH` 指定的目录。您可以通过 TensorBoard 查看训练曲线。

此外，如果安装了 wandb，您可以配置 `WANDB_API_KEY` 将训练指标上传到 wandb 进行在线监控。
