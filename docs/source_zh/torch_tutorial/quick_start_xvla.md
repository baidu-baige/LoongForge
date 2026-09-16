# 快速入门：X-VLA模型训练

本文档介绍如何在 LoongForge 框架下快速启动 **X-VLA** SFT（监督微调）训练的快速开始流程。

## 0. 资源准备
### 0.1 模型权重
X-VLA 主权重（Florence2 视觉-语言 backbone + SoftPromptedTransformer 动作头）由 `CHECKPOINT_PATH` 指定，并通过 `--pretrained-checkpoint` 加载，支持原始 HuggingFace safetensors 权重形态。以官方 `X-VLA-WidowX` 微调权重为例，将其下载到本地目录：

```bash
# 将官方发布的 X-VLA-WidowX checkpoint 放到该目录
hf download 2toINF/X-VLA-WidowX --local-dir /workspace/ckpt/X-VLA-WidowX
```
### 0.2 Tokenizer / Processor
X-VLA 复用 Florence2 的 tokenizer / image processor，与主权重存放在同一 checkpoint 目录，由 `TOKENIZER_PATH` 指定并通过 `--tokenizer-path` 加载：

```bash
export TOKENIZER_PATH=/workspace/ckpt/X-VLA-WidowX
```
### 0.3 数据集
X-VLA 使用 episode 级 HDF5 数据集（`HDF5VLADataset`）。数据集通过一个 `metadata.json` 描述文件组织，`DATA_PATH` 指向包含该文件及 `episode_*.hdf5` 的目录：

```bash
hf download 2toINF/X-VLA-SoftFold --repo-type dataset --local-dir /workspace/data/X-VLA-SoftFold
```
## 1. 数据配置
X-VLA 采用与参考实现对齐的原生 HDF5 数据管道，**无需额外离线预处理**，训练时在线执行 X-VLA 的 per-sample transform（多视角图像编码、语言 tokenize、由 `robot_type` 解析 `domain_id`）与 batch collator。

数据目录通过 `metadata.json` 描述，示例（`examples/vla/xvla/xvla_soft_fold/metadata.json`）：

```json
{
  "dataset_name": "AIR-AGILEX",
  "robot_type": "AIR-AGILEX",
  "datalist": [
    "/mnt/cfs/pxy/data/XVLA-Soft-Fold/0928_10am_new/episode_0.hdf5",
    "/mnt/cfs/pxy/data/XVLA-Soft-Fold/0928_10am_new/episode_1.hdf5"
  ],
  "observation_key": [
    "observations/images/cam_high",
    "observations/images/cam_left_wrist",
    "observations/images/cam_right_wrist"
  ],
  "language_instruction_key": "language_instruction"
}
```
字段说明：

- `dataset_name` / `robot_type`：数据集名与具身类型。`robot_type` 决定 `domain_id`（选择 SoftPromptedTransformer 的 per-domain soft prompt 与 DomainAwareLinear 权重），需与训练该 checkpoint 时使用的取值一致。
- `datalist`：episode HDF5 文件的绝对路径列表；缺省时会在 `DATA_PATH` 下自动发现 `episode_*.hdf5`。
- `observation_key`：多视角图像在 HDF5 中的 key，顺序即视角顺序（第一路为主视角）。
- `language_instruction_key`：语言指令字段名。

## 2. 启动训练
先统一设置路径（参考 `examples/vla/xvla/run_xvla_ddp_finetune.sh`）：

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export TOKENIZER_PATH=/workspace/ckpt/X-VLA-WidowX
export CHECKPOINT_PATH=/workspace/ckpt/X-VLA-WidowX
export DATA_PATH=/workspace/data/X-VLA-SoftFold/0928_10am_new/
export OUTPUT_DIR=/workspace/outputs/xvla_ddp
```
### 2.1 启动脚本
单机 8 卡 DDP 微调：

```bash
bash examples/vla/xvla/run_xvla_ddp_finetune.sh
```
默认已开启了优化，参数在 `configs/models/vla/xvla.yaml`：

```yaml
enable_torch_compile: true
torch_compile_mode: default
```
### 2.2 关键参数说明
脚本内主要参数分组如下。

**模型与分布式：**

```bash
--model-name xvla                    # 通过 models/catalog.py 映射到 XVLA 配置
--distributed-strategy ddp           # DDP 分布式策略
--dtype bfloat16                     # 训练精度
# GPUS_PER_NODE=8                    # 单机 GPU 数（脚本变量）
```
**数据：**

```bash
--dataset-format hdf5_datasets       # 使用原生 HDF5 数据集
--dataset-path $DATA_PATH            # metadata.json 所在目录
--tokenizer-path $TOKENIZER_PATH     # Florence2 tokenizer / processor 目录
--robot-type libero_franka           # 具身类型标识
--num-workers 2                      # DataLoader worker 数
```
**训练：**

```bash
--trainer-type FinetuneTrainer
--train-iters 40
--gradient-accumulation-steps 1
--lr-base 1e-4                       # 基础学习率
--lr-group "model.vlm=1e-5,model.transformer.soft_prompt_hub=1e-5"  # 分组学习率：VLM backbone 与 soft prompt 使用更小 lr
--lr-warmup-iters 5
--optimizer AdamW
--clip-grad 1.0
--weight-decay 0.0
--adam-beta1 0.9
--adam-beta2 0.95
--adam-eps 1e-8
--pretrained-checkpoint $CHECKPOINT_PATH   # 预训练权重
--save-interval 100                  # checkpoint 保存间隔
--seed 42
```

> **说明：** `metadata.json` 中的 `robot_type` 决定训练/推理使用的 `domain_id`，请确保其与所加载 checkpoint 的训练具身一致，否则会选错 per-domain 的 soft prompt / 线性权重。

### 2.3 正确性验证
为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的X-VLA 与官方实现进行了逐 step 的 loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：
![alt text](../../assets/images/precision/xvla.png)
