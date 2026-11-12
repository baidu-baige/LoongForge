# 快速入门：GigaBrain-0.7模型训练

本文档介绍如何在 LoongForge 框架下快速启动 **GigaBrain-0.7** SFT（监督微调）训练的快速开始流程。

## 0. 资源准备
### 0.1 模型权重
GigaBrain-0.7 主权重（PaliGemma2 视觉-语言 backbone + action expert）由 `GB0_PRETRAINED` 指定，并通过 `model.pretrained` 加载。以官方 `GigaBrain-0.7-3.5B-Base` 权重为例，将其下载到本地目录：

```bash
hf download open-gigaai/GigaBrain-0.7-3.5B-Base --local-dir /workspace/ckpt/open-gigaai/GigaBrain-0.7-3.5B-Base
```
### 0.2 Tokenizer
GigaBrain-0.7 使用两个 tokenizer：PaliGemma 的文本 tokenizer（`GB0_TOKENIZER`，通过 `--tokenizer-path` 与 `data.tokenizer_model_path` 加载）和 FAST 动作 tokenizer（`GB0_FAST_TOKENIZER`，通过 `data.fast_tokenizer_path` 加载）：

```bash
hf download google/paligemma-3b-pt-224 --local-dir /workspace/ckpt/paligemma-3b-pt-224
hf download physical-intelligence/fast --local-dir /workspace/ckpt/physical-intelligence/fast
```
### 0.3 数据集
GigaBrain-0.7 使用 LeRobot v3.0 格式数据集，示例为 AgiBot World 的 `task_327`。另需提供与具身对应的归一化统计文件（`norm_stats.json`）：

```bash
export GB0_DATA_PATH_V30=/workspace/data/agibot-world/AgiBotWorldLerobot_v30/agibotworld/task_327
export GB0_NORM_STATS=/workspace/data/agibot-world/AgiBotWorldLerobot/agibot_a2d_norm_stats.json
```
## 1. 数据配置
GigaBrain-0.7 通过 `--dataset-strategy giga_brain07` 复用 LoongForge 共享的 LeRobot v3 reader，**无需额外离线预处理**。训练时在线执行 GigaBrain-0.7 的 per-sample transform（多视角图像 resize、state/action 归一化、语言与 FAST 动作 tokenize）与 batch collator。

数据相关默认值在 `configs/models/giga_brain07/giga_brain07.yaml` 的 `data` 段：

```yaml
data:
  camera_keys:
  - observation.images.top_head
  - observation.images.hand_left
  - observation.images.hand_right
  action_chunk: 50
  video_backend: torchcodec
  robot_type: a2d
  embodiment_override: AGIBOT_G1
  embodiment_id: '1'
  resize_hw: [224, 224]
  max_length: 120
  use_quantiles: true
```
字段说明：

- `camera_keys`：多视角图像在 LeRobot 数据集中的 key，顺序即视角顺序（第一路为主视角）。
- `action_chunk`：每个样本的动作块长度，需与 `model.action_horizon` 一致。
- `video_backend`：视频解码后端。
- `robot_type` / `embodiment_override` / `embodiment_id`：具身类型与具身编号。`embodiment_id` 选择模型中对应具身的参数，需与训练该 checkpoint 时使用的取值一致。
- `resize_hw`：输入图像 resize 尺寸。
- `max_length`：语言 token 最大长度。
- `use_quantiles`：使用 `norm_stats.json` 中的分位数做归一化。

## 2. 启动训练
先统一设置路径（参考 `examples/giga_brain07/run_giga_brain07_torchrun.sh`）：

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export GB0_PRETRAINED=/workspace/ckpt/open-gigaai/GigaBrain-0.7-3.5B-Base
export GB0_TOKENIZER=/workspace/ckpt/paligemma-3b-pt-224
export GB0_FAST_TOKENIZER=/workspace/ckpt/physical-intelligence/fast
export GB0_DATA_PATH_V30=/workspace/data/agibot-world/AgiBotWorldLerobot_v30/agibotworld/task_327
export GB0_NORM_STATS=/workspace/data/agibot-world/AgiBotWorldLerobot/agibot_a2d_norm_stats.json
export OUTPUT_DIR=/workspace/outputs/giga_brain07_fsdp
```
### 2.1 启动脚本
单机 8 卡 FSDP2 微调：

```bash
bash examples/giga_brain07/run_giga_brain07_torchrun.sh
```
脚本默认已开启以下优化，可通过环境变量关闭或调整：

```bash
GB0_COMPILE=decoder,vision           # 对 decoder 与 vision encoder 层做 regional torch.compile；置空则关闭
GB0_KV_SLICE=1                       # 每个 decoder 层只接收自身的 KV cache；置 0 则关闭
GB0_AC_PATTERNS=...                  # 重计算的模块，默认覆盖 26 个 decoder 层与 vision encoder 各层
```
其他常用环境变量：`GB0_BATCH_SIZE`（每卡 batch size，默认 80）、`GB0_MAX_STEPS`（训练步数，默认 100）、`GB0_NUM_WORKERS`（DataLoader worker 数，默认 8）、`GB0_DTYPE`（训练精度，默认 `float32`）。
### 2.2 关键参数说明
脚本内主要参数分组如下。

**模型与分布式：**

```bash
--model-name giga_brain07            # 通过 catalog 映射到 GigaBrain-0.7 配置
--training-phase sft                 # 监督微调
--distributed-strategy fsdp          # FSDP2 分布式策略
--dtype float32                      # 训练精度（GB0_DTYPE）
--activation-checkpoint-module-patterns "$GB0_AC_PATTERNS"  # 激活重计算的模块
# GPUS_PER_NODE=8                    # 单机 GPU 数（脚本变量）
```
**数据：**

```bash
--dataset-format lerobot_datasets    # 使用 LeRobot 数据集
--dataset-strategy giga_brain07      # GigaBrain-0.7 的数据构建策略
--dataset-path $GB0_DATA_PATH_V30    # LeRobot v3.0 数据集目录
--tokenizer-path $GB0_TOKENIZER      # PaliGemma tokenizer 目录
--num-workers 8                      # DataLoader worker 数
data.norm_stats_path=$GB0_NORM_STATS # 归一化统计文件
data.fast_tokenizer_path=$GB0_FAST_TOKENIZER  # FAST 动作 tokenizer 目录
```
**训练：**

```bash
--train-iters 100
--per-device-batch-size 80
--gradient-accumulation-steps 1
--optimizer AdamW
--lr-base 2.5e-5                     # 基础学习率
--lr-decay-style giga_warmup_cosine  # 与 GigaBrain 原版一致的 warmup + cosine 调度
--lr-warmup-iters 10
--lr-decay-iters 300
--min-lr 2.5e-6
--adam-beta1 0.9
--adam-beta2 0.95
--adam-eps 1e-8
--weight-decay 1e-10
--clip-grad 0                        # 不做梯度裁剪
model.pretrained=$GB0_PRETRAINED     # 预训练权重
--save-interval 0                    # checkpoint 保存间隔，0 表示不保存
--seed 6666
```

> **说明：** `data.embodiment_id` 决定训练/推理使用的具身参数，请确保其与所加载 checkpoint 的训练具身一致；`data.norm_stats_path` 也需与该具身的数据统计对应。
