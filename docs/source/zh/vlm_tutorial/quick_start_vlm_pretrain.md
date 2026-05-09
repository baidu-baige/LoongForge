# 快速入门：VLM 预训练

本文档将指导你在 LoongForge 框架下完成视觉语言模型（VLM）预训练的快速入门流程。

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

> **注意：** 该模型约需 **62 GB** 磁盘空间（13 个 safetensor 分片）。HuggingFace 上没有非 Instruct 的 base 变体，预训练和 SFT 均使用 Instruct 变体。

### 0.2 下载分词器

分词器已包含在上方下载的模型权重中（`./Qwen3-VL-30B-A3B-Instruct/`）。

### 0.3 下载数据集

快速验证可使用 LLaVA-Instruct-Mix-VSFT-Small 数据集（约 109 MB，2,592 条样本，ShareGPT 格式的多模态图文对）。正式大规模预训练请准备自己的多模态语料。

```bash
hf download axolotl-ai-co/llava-instruct-mix-vsft-small --repo-type dataset --local-dir ./data/llava-instruct-mix-vsft-small
```

下载后，按照本文档第 1.2 节的说明将数据集转换为 WebDataset 格式。

---

## 1. 数据准备

在模型训练之前，你需要对大规模预训练数据进行处理和转换，以最大化训练速度。具体流程如下：

### 1.1 原始数据

原始数据集为 JSON/JSONL 格式，每条数据包含图像路径和对应的多轮对话内容。

**示例数据（data.json）：**

```json
[
  {
    "messages": [
      {
        "content": "<image>Who are they?",
        "role": "user"
      },
      {
        "content": "They're Kane and Gretzka from Bayern Munich.",
        "role": "assistant"
      },
      {
        "content": "What are they doing?",
        "role": "user"
      },
      {
        "content": "They are celebrating on the soccer field.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg"
    ]
  }
]
```

### 1.2 转换为 **Energon 加载格式**

考虑到多模态数据集的多样性，框架采用 **Energon** 加载器来提升数据处理性能。**Energon** 要求数据集以标准 **WebDataset** 格式存储。WebDataset 以原生文件格式（jpg、mp4 等）存储数据，这使得各种原生多模态数据集可以简单地压缩并转换为 WebDataset 格式，然后由 Energon 读取。

将数据转换为 **WebDataset 并适配 Energon 加载格式**的脚本如下：

```bash
python /workspace/LoongForge/tools/data_preprocess/vlm/convert_to_webdataset.py \
    --output_dir /tmp/mllm/wds \
    --json_file /tmp/mllm/mllm_demo.json \
    --image_dir /tmp/mllm/ \
    --video_dir /tmp/vlm/ \
    --media mix \
    --columns_messages messages \
    --maxcount 10000 \
    --maxsize 3000000000 \
    --sample_type multi_mix_qa
```

转换后的数据集目录结构：

```
.
├── .nv-meta
│   ├── .info.yaml
│   ├── dataset.yaml
│   └── split.yaml
├── pretrain-0.tar
└── pretrain-0.tar.idx
```

功能说明：

* `convert_to_webdataset.py` 会从 **json_file** 中提取每个样本，将其存储为独立的 json 文件，连同 **image_dir** 中对应的图像一起，压缩到 **$output_dir** 中。每个 tar 包最多包含 **maxcount** 个样本；
* 后续启动训练时，通过 --data-path 参数指定 WebDataset 路径 /tmp/mllm/wds 来读取训练数据；
* Energon 格式相比 WebDataset 额外添加了 yaml 文件来记录数据集信息，用于后续 dataloader 解析
    * `.info.yaml`：记录每个压缩包中的样本数量
    * `dataset.yaml`：记录样本信息
    * `split.yaml`：记录数据集划分

如需进一步了解数据集转换的各种参数和详细功能，请参阅 [dataset_conversion.md](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/dataset_conversion.html)

## 2. 模型权重准备

训练通常从开源的 HuggingFace 权重开始。我们需要先下载权重，然后将其转换为本框架支持的格式（Megatron-Core 格式）。

### 2.1 下载 HuggingFace 模型

以 Qwen3-VL-30B-A3B 为例，请从 HuggingFace 下载模型权重（[https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)）。

### 2.2 转换权重格式

LoongForge 为支持的模型提供了统一的权重转换工具 `tools/convert_checkpoint`，可以方便地在 HuggingFace 和 MCore 格式之间进行转换。以 Qwen3-VL-30B-A3B 为例，如需将 HuggingFace 权重转换为 LoongForge 支持的 MegatronCore 格式，可参考以下示例：

```bash
#!/bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

LOAD=/path/to/hf_checkpoint  # 原始 Qwen3-VL-30B-A3B 权重路径
SAVE=/path/to/your/save  # 转换后的权重保存路径

SAVE_LANGUAGE_MODEL=${SAVE}/tmp/language-mcore
SAVE_VISION_MODEL=${SAVE}/tmp/vision-model-mcore
SAVE_ADAPTER=${SAVE}/tmp/adapter-mcore
SAVE_PATCH=${SAVE}/tmp/patch-mcore

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert_qwen3vl.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen3_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_3_mlp_adapter_convert.yaml

ETP=1
DTP=1
PP=2
EP=8

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --moe-grouped-gemm

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# 合并
if [ $EP -gt 1 ]; then
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron_expert.py\
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --expert_parallel_size $EP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE
else
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py\
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE
fi

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
```

部分参数说明：

* ETP / DTP：编码器和解码器的张量并行参数（支持 ETP != DTP 的异构并行策略）

如需进一步了解权重转换的各种参数和详细功能，请参阅：[vlm_ckpt_convert.md](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/vlm_ckpt_convert.html)

## 3. 启动预训练

### 3.1 参数配置说明

在开源 Megatron 提供的参数基础上，LoongForge 添加了更便捷的训练启动参数。详细配置可在 loongforge/train/arguments.py 文件中找到。主要参数说明如下：

* `--training-phase`：指定训练阶段为 pretrain
* `--add-question-in-pretrain`：启用后，问题将被拼接并添加到训练输入中；禁用时，仅使用答案或其他默认文本字段进行训练
* `--enable-discard-sample`：启用后，超过 --seq-length 的样本将被直接丢弃，不进行截断或其他处理
* `--dataloader-save`：启用后，训练过程中 dataloader 状态将写入此路径，便于权重重启时恢复一致的数据读取顺序
* `--packing-sft-data`：启用后，将激活在线 packing 策略，将多个较短的样本拼接为一个长样本

### 3.2 预训练脚本

LoongForge 目前为各种模型提供了预训练示例脚本。进入容器后，可在 examples/{model}/pretrain/ 目录下找到相关脚本。以下是使用 Qwen3-VL-30B-A3B 预训练脚本的示例：

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

# 多节点配置修改，分布式训练连接设置。
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
    --add-question-in-pretrain
    --enable-discard-sample
    --num-workers 16
)

# 核心训练超参数
TRAINING_ARGS=(
    --training-phase pretrain # 选项：pretrain, sft
    --seq-length 32768
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 32
    --lr 0.0002
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
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
)

# 并行与分布式训练
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
)

# 模型架构/配置文件
MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

# 日志与监控
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

### 监控日志

脚本默认会将 TensorBoard 日志输出到 TENSORBOARD_PATH 指定的目录。你可以通过 TensorBoard 查看训练曲线。
