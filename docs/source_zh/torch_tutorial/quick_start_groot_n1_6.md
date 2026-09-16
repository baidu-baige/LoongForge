# 快速入门：GR00TN1.6模型训练

本文档介绍如何在 LoongForge 框架下快速启动 框架下 **GROOTN1.6** SFT（监督微调）训练

## 0. 资源准备
### 0.1 模型权重
GR00T-N1.6 主权重由 `CHECKPOINT_PATH` 指定，并通过`--pretrained-checkpoint` 加载，支持原始huggingface safetensors权重形态：

```bash
hf download nvidia/GR00T-N1.6-3B --local-dir /workspace/huggingface.co/nvidia/GR00T-N1.6-3B
```
### 0.2 Tokenizer / Processor
Eagle processor/tokenizer 由 `EAGLE_LOCAL_PATH` 指定：

```bash
mkdir -p /workspace/huggingface.co/aravindhs-NV
huggingface-cli download aravindhs-NV/eagle3-processor-groot-n1d6 --local-dir /workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6
```
注意：`EAGLE_LOCAL_PATH` 不是主模型权重路径，它只负责 Eagle/VLM processor、tokenizer 和相关 config 的本地加载。

### 0.3 数据集
脚本默认使用标准 LeRobot v3.0 格式的 LIBERO 数据， 如果数据集来自 Hugging Face Dataset，可下载到同一目录：

```bash
hf download lerobot/libero_10 --repo-type dataset --local-dir /workspace/libero_10
```
## 1. 数据配置
针对标准的 LeRobot v3.0 格式，不需要额外离线预处理，训练时会在线执行GR00T 的 sample transform 和 batch collator。

默认数据相关配置：

```yaml
data:
  embodiment_tag: libero_panda
  groot_preprocess_mode: sample
  preprocess_action_horizon: 16
  preprocess_max_action_dim: 29
  preprocess_max_state_dim: 29
  image_crop_size: [224, 224]
  image_target_size: [224, 224]
  formalize_language: true
  use_relative_action: true
```
## 2. 启动训练
先统一设置路径：

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export CHECKPOINT_PATH=/workspace/huggingface.co/nvidia/GR00T-N1.6-3B
export EAGLE_LOCAL_PATH=/workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6
export DATA_PATH=/workspace/libero_10
export OUTPUT_DIR=/workspace/outputs/groot_n1_6
export TENSORBOARD_PATH=/workspace/tensorboard-log/groot_n1_6
```
### 2.1 无性能优化项的脚本
用于先验证链路，关闭 CUDA Graph、DDP static graph，并使用普通 `AdamW`：

```bash
bash /workspace/LoongForge/examples/vla/groot_n1_6/run_groot_n1_6_ddp_finetune.sh \
    --optimizer AdamW \
    --cuda-graph-impl none \
    --cuda-graph-pad-length 0 \
    --no-ddp-static-graph
```
### 2.2 性能优化项及对应的开关方式
默认脚本已经打开主要性能优化：

```bash
bash /workspace/LoongForge/examples/vla/groot_n1_6/run_groot_n1_6_ddp_finetune.sh
```
关键性能优化项开关：

* 启用TEFusedAdamW优化器

```python
--optimizer TEFusedAdamW
```
* 启用CUDA Graph

```python
--cuda-graph-impl local                                 # 启用CUDA Graph
--cuda-graph-scope per_microbatch                       # 捕获的粒度，full_iteration和per_microbatch可选，Per-Microbatch 模式在保持接近 Full-Iteration 性能的同时，具备更好的 Loss 对齐效果
--cuda-graph-warmup-steps 3                             # eager模式warmup步数
--cuda-graph-pad-length 220                             # 序列长度固定为220，确保输入尺寸恒定，使CUDA Graph可复用。
--no-cuda-graph-ddp-sync-in-graph                       # graph外部进行DDP 梯度all-reduce
--cuda-graph-grad-sync-bucket-mb 400                    # 设置梯度同步时每个bucket的大小
--cuda-graph-grad-sync-impl coalesced                   # 按bucket分别同步梯度
--cuda-graph-grad-sync-dtype bf16                       # 梯度all-reduce通信时使用的数据类型
```
### 2.3 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 GR00T-N1.6 与Lerobot官方实现进行了逐 step 的 action loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：
![alt text](../../assets/images/precision/GR00TN1.6.png)
