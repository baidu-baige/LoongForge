# 快速开始：昆仑芯 P800 上 VLM 模型 SFT（监督微调）训练

## 快速开始：VLM 模型 SFT（监督微调）训练

本文档引导您完成在 P800 上使用 LoongForge 框架对视觉语言模型（VLM）进行 SFT（监督微调）的快速开始流程。

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

我们使用 LLaVA-Instruct-Mix-VSFT-Small 数据集（约 109 MB，2,592 条样本，ShareGPT 格式的多模态图文对）进行 VLM SFT。

```bash
hf download axolotl-ai-co/llava-instruct-mix-vsft-small --repo-type dataset --local-dir ./data/llava-instruct-mix-vsft-small
```

## 1. 数据准备与权重转换

在第 0 节下载资源后，训练前需将数据集转换为 WebDataset 格式并转换权重。这些步骤与 GPU 版本相同：

* **数据集转换**：将下载的数据集转换为 Energon/WebDataset 格式——参见 [快速开始：VLM SFT](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_sft.html)第 1.3 节。
* **权重转换**：将 HF VLM 权重（语言模型、视觉编码器、适配器）转换为 Megatron-Core 格式——参见 [快速开始：VLM 预训练](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html)第 2 节。

## 2. SFT（监督微调）训练脚本

LoongForge 目前提供了多种模型的 SFT（监督微调）训练示例脚本。进入容器后，您可以在 `examples_xpu/{model}/finetuning/` 目录下找到相关脚本。以下是 `Qwen3-VL-30B-A3B` 的 SFT（监督微调）训练脚本示例。请参考注释了解各部分脚本的作用：

```bash
#! /bin/bash
# 此脚本需要在至少 2 个节点上运行。
source /root/.bashrc
source activate && conda activate python310_torch25_cuda

pkill -9 python || true

function check_for_infer() {
    /usr/local/xpu/tools/rw $1 0x300010B8 0
    /usr/local/xpu/tools/rw $1 0x300410B8 0
    /usr/local/xpu/tools/rw $1 0x300810B8 0
    /usr/local/xpu/tools/rw $1 0x310010B8 0
    /usr/local/xpu/tools/rw $1 0x310410B8 0
    /usr/local/xpu/tools/rw $1 0x310810B8 0
    /usr/local/xpu/tools/rw $1 0x320010B8 0
    /usr/local/xpu/tools/rw $1 0x320410B8 0
    /usr/local/xpu/tools/rw $1 0x320810B8 0
    /usr/local/xpu/tools/rw $1 0x330010B8 0
    /usr/local/xpu/tools/rw $1 0x330410B8 0
    /usr/local/xpu/tools/rw $1 0x330810B8 0
}
for ((i=0; i<8; i++))
do
    check_for_infer $i
done

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

DATA_PATH=${DATA_PATH:-"/mnt/rapidfs/loongforge-test/sft_qwen3_vl_30b_a3b_temp/data-path/LLaVA-Pretrain_202511180001/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/rapidfs/loongforge-test/sft_qwen3_vl_30b_a3b_temp/hf-tokenizer-path/Qwen3-VL-30B-A3B-Instruct_202512180001/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/rapidfs/loongforge-test/sft_qwen3_vl_30b_a3b_temp/load/qwen3-vl-30b-tp4pp1ep8etp1-groupedgemm_202512180001/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/rapidfs/users/baige/checkpoints/qwen3-vl/qwen3-vl-30b-tp4pp1ep8etp1-groupedgemm-save/tensorboard-log/"}

GPUS_PER_NODE=8
###################### 昆仑芯 P800 ######################
# bf16 专用（Megatron 相关变量请参考 <Loong Megatron 专用>）
export XMLIR_ENABLE_FAST_FC=true                # 用于 torch.nn.linear.py（LinearWithActFunction 等）
# export XMLIR_ENABLE_FAST_FC_FWD_OUT=true      # 用于前向输出
# export XMLIR_ENABLE_FAST_FC_BWD_DW=true       # 用于反向 DW
# export XMLIR_ENABLE_FAST_FC_BWD_DX=true       # 用于反向 DX
export FORCE_DISABLE_INPLACE_BF16_CAST=false    # 默认为 false，特殊情况下需启用（异步权重）

export CUDA_DEVICE_MAX_CONNECTIONS=1            # Megatron 框架设置，防止 tp>1 时乱序

export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4" # 多节点时使用，根据实际网络连接调整
export BKCL_SOCKET_IFNAME=eth0                  # 根据实际环境调整，默认禁用，找不到网卡时需指定
export BKCL_TREE_THRESHOLD=0
export BKCL_FORCE_L3_RDMA=0                     # 设置为 1 可能导致空间不足时 OOM
export BKCL_ENABLE_XDR=1
export BKCL_ALL_TO_ALL_OPT=1                    # 多节点 alltoall 开关
export BKCL_RING_HOSTID_USE_RANK=1              # 从 1.2.11 版本开始支持，未来将成为默认值
export BKCL_RDMA_VERBS=1                        # 与 BKCL_QPS_PER_CONNECTION 配合使用，目前仅海光机器需要
export XMLIR_PARALLEL_SAVE_MEMORY=false         # false：内存占用更多但性能更好；true：内存占用减少但性能下降
export XMLIR_BATCH_PARALLEL=false               # 启用通信融合算子，bf16 下 USE_CAST_FC_FUSION 自动禁用
export SAVE_LOG_FILE_WITH_RANK_ID=false          # 设为 true 时，训练日志将按 rank_id 分别存储
export XMLIR_LOG_PATH="/mnt/rapidfs/loongforge-test/sft_qwen3_vl_30b_a3b_temp/logs"  # 指定训练日志存储目录
export XMLIR_LOG_PREFIX="qwen3_vl_30b_sft"      # 指定训练日志文件名前缀
export P800_DEBUG=false                         # 设为 true 时，梯度范数变为 nan 将保存权重并退出
export P800_DUMP_DIR="ckpt-dump-dir-path"       # 指定梯度范数变为 nan 时权重和信息的转储目录
export XMLIR_DIST_ASYNC_ISEND_IRECV=true        # true：send/recv 使用异步逻辑，默认为同步
export XMLIR_CUDNN_ENABLED=1                    # true：使用 cuDNN，支持 conv3d 等；false：禁用 cuDNN

# LINEAR 开关
export XMLIR_ENABLE_LINEAR_FC_FUSION=1          # 允许 linear 在特定场景下绕过 xblas fcfusion，例如使用 addmm，默认为 1
export XDNN_FC_GEMM_DTYPE=int32_with_ll         # GEMM_DTYPE 使用 int32_with_ll，可选
export XMLIR_MEGATRON_CORE_XPU_PLUGIN=1        # xpu_plugin，针对 P800 特性的模拟实现，推荐启用以提升性能

XFLAGS --disable transformer_engine_1_7         # 遗留
XFLAGS --disable transformer_engine_1_13        # 遗留
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
    --model-name qwen3_vl_30b_a3b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 8
    --chat-template qwen2-vl
    --packing-sft-data
    --packing-batch-size 1000
    --max-packed-tokens 4096
    --enable-discard-sample
)

TRAINING_ARGS=(
    --seed 42
    --norm-epsilon 1e-6
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length 4096
    --max-position-embeddings 262144
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-5
    --min-lr 0.
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-08
    --train-iters 100
    --lr-decay-style cosine
    --lr-warmup-fraction 0.03
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    #--save $CHECKPOINT_PATH
    --save-interval 10000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
    --no-rope-fusion
    --no-bias-dropout-fusion
    --no-bias-gelu-fusion
    --no-gradient-accumulation-fusion
    --exit-interval 500
)

MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    # --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 1e-3
    --moe-router-topk 8
    #--empty-unused-memory-level 2
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    #--overlap-grad-reduce
    #--overlap-param-gather
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    model.image_encoder.apply_rope_fusion=False \
```

## 监控日志

默认情况下，脚本将 TensorBoard 日志输出到 `TENSORBOARD_PATH` 指定的目录。您可以通过 TensorBoard 查看训练曲线。

此外，如果安装了 wandb，您可以配置 `WANDB_API_KEY` 将训练指标上传到 wandb 进行在线监控。
