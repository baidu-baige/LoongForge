# 快速入门：Pi05模型训练

本文档介绍如何在 LoongForge 框架下快速启动 **Pi0.5（π₀.₅）** 的微调训练与离线评测

## 0. 资源准备
### 0.1 模型权重
pi05 主权重由 `CHECKPOINT_PATH` 指定，并通过`--pretrained-checkpoint` 加载，支持原始huggingface safetensors权重形态：

```bash
hf download lerobot/pi05_base --local-dir /workspace/huggingface.co/lerobot/pi05_base
```
### 0.2 Tokenizer / Processor
pi05 需要tokenzier 参数，由 TOKENIZER_PATH 制指定，通过 --tokenizer-path 加载

```bash
huggingface-cli download google/paligemma-3b-pt-224 --local-dir /workspace/huggingface.co/google/paligemma-3b-pt-224
```


### 0.3 数据集
脚本默认使用 LeRobot v3.0 格式的 LIBERO 数据， 如果数据集来自 Hugging Face Dataset，可下载到同一目录：

```bash
hf download lerobot/libero_10 --repo-type dataset --local-dir /workspace/libero_10
```
## 1. 数据配置
针对标准 LeRobot v3.0 格式，不需要额外离线预处理，训练时会在线执行GR00T 的 sample transform 和 batch collator。

默认数据相关配置：

```yaml
data:
  image_size: 224
  image_normalize_mode: identity
```
## 2. 启动训练
先统一设置路径：

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export TOKENIZER_PATH=/workspace/huggingface.co/google/paligemma-3b-pt-224

export CHECKPOINT_PATH=/workspace/workspace/huggingface.co/lerobot/pi05_base

export DATA_PATH=/workspace/libero_10
export OUTPUT_DIR=/workspace/outputs/pi05
export TENSORBOARD_PATH=/workspace/tensorboard-log/pi05
```
### 2.1 脚本


DDP 脚本已经打开主要性能优化：

**DDP：**

```bash
bash examples/vla/pi05/run_pi05_ddp_finetune.sh
```
**DDP + ZeRO-1：**

```bash
bash examples/vla/pi05/run_pi05_ddp_zero1_finetune.sh
```
**FSDP：**

```bash
bash examples/vla/pi05/run_pi05_fsdp_finetune.sh
```
### 2.2 正确性验证
为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 Pi05 与官方实现进行了逐 step 的 action loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：
![alt text](../../assets/images/precision/pi05.png)




## 3. 评测
 提供独立的 benchmark 评测模块（LIBERO / CALVIN / SimplerEnv / RoboTwin / ManiSkill），benchmark client 与 policy server 通过 WebSocket 解耦。

### 3.1 快速运行 LIBERO
1. 编辑评测 YAML

（示例：`examples/vla/pi05/eval/configs/libero/smoke_steps10.yaml`），填写：

    * `server.ckpt_path`：训练得到的 checkpoint（目录含 `model.safetensors` 或权重文件）
    * `server.dataset_statistics_path`：数据集统计（action 反归一化，Pi0.5 默认 q99）
    * `server.tokenizer_path`：PaliGemma tokenizer

2. 启动：

```bash
cd $LOONGFORGE_PATH
# 默认使用 smoke_steps10.yaml；可用 CONFIG 覆盖
CONFIG=examples/vla/pi05/eval/configs/libero/smoke_steps10.yaml \
  bash examples/vla/pi05/eval/run_libero_eval.sh
```
