# 快速入门：LingBot-VA模型训练

本文档介绍如何在 LoongForge 框架下快速启动 **LingBot-VA** 后训练，示例提供 RoboTwin 和 LIBERO 两个 8 卡训练入口

## 0. 资源准备
LingBot-VA 的后训练权重和 LeRobot 数据集发布在以下地址：

|资源|地址|
|-|-|
|RoboTwin 后训练权重|[https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin)|
|LIBERO-Long 后训练权重|[https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long)|
|RoboTwin LeRobot 数据集|[https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot)|
|LIBERO-Long LeRobot 数据集|[https://huggingface.co/datasets/robbyant/libero-long-lerobot](https://huggingface.co/datasets/robbyant/libero-long-lerobot)|

本项目采用与 LingBot-VA 官方社区相同的离线预处理方案。请提前将权重和数据准备到本地 `/workspace` 目录，数据目录需要包含预生成 latent 和对应的 `empty_emb.pt`。使用自有数据集时，先将原始机器人数据转换为 LeRobot 格式，再在 `meta/episodes.jsonl` 的每个 episode 中补充 `action_config`（动作片段起止帧和文本描述），将动作映射到标准 30 维格式（不足维度补 0），并使用 Wan2.2 VAE 提取视频 latent，放入与 `videos/` 对应的 `latents/` 目录。视频建议按官方数据规格预处理。详细字段、目录结构和预处理命令请参考官方 [README.md](https://github.com/Robbyant/lingbot-va/blob/main/README.md) 的 **Custom Dataset Preparation** 章节。

## 1. 启动训练
```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export NCCL_DEBUG=WARN
export GPUS_PER_NODE=8
```
RoboTwin：

```bash
export CHECKPOINT_PATH=/workspace/models/lingbot-va-posttrain-robotwin
export DATA_PATH=/workspace/datasets/robotwin-clean-and-aug-lerobot
export OUTPUT_DIR=/workspace/outputs/lingbot_va_robotwin

bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh
```
LIBERO-Long：

```bash
export CHECKPOINT_PATH=/workspace/models/lingbot-va-posttrain-libero-long
export DATA_PATH=/workspace/datasets/libero-long-lerobot
export OUTPUT_DIR=/workspace/outputs/lingbot_va_libero

bash examples/embodied/lingbot_va/run_lingbot_va_libero_fsdp_finetune.sh
```
## 2. 性能优化项
### 2.1 固化项
以下优化已经固化在 LingBot 实现中，没有关闭开关：

|优化项|作用|解决的性能瓶颈|
|-|-|-|
|Self-Flex mask cache|复用已生成的 attention mask。|避免重复构造 mask 带来的 CPU/GPU 开销。|
|LeRobot repo discovery cache|缓存数据目录中的 repo 列表。|避免每轮反复扫描大量 metadata，降低启动和 dataloader 初始化开销。|
|Transformer Engine Q/K RMSNorm|使用融合的 Q/K RMSNorm 实现。|减少归一化的 kernel launch、访存和中间张量开销。|
|Self-Flex block64|为 Flex Attention 的 Triton 融合前向/反向 kernel 固定 block64 配置（FWD=`64,64,4,3`，BWD=`32,32,4,1`）。|避免退化到 dense attention，降低 attention 的计算和显存压力。|
|Cross attention 优化实现|使用高效的 cross-attention 路径。|减少 cross-attention 的 kernel 和显存开销。|

### 2.2 可选项
以下性能优化可由用户控制，示例脚本默认开启：

|开关|默认值|作用|
|-|-|-|
|`LINGBOT_BALANCED_SAMPLER`|`1`|按样本计算量在 DP ranks 间做确定性负载均衡，降低慢 rank 等待；同时改变样本加载顺序，如需与官方社区逐 step 对齐 loss，请关闭。|
|`LINGBOT_LAYERWISE_COMPILE`|`1`|编译 layerwise norm、modulation 和 residual gate 路径。|

例如关闭可选优化：

以下命令以 RoboTwin 为例；LIBERO 使用 `run_lingbot_va_libero_fsdp_finetune.sh`。

```bash
LINGBOT_BALANCED_SAMPLER=0 \
LINGBOT_LAYERWISE_COMPILE=0 \
bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh
```
以下开关用于日志、诊断或产物管理，不改变模型训练语义：

|开关|默认值|作用|
|-|-|-|
|`LINGBOT_BASELINE_LOSS_LOG`|`1`|输出与官方 baseline 兼容的 loss 日志，便于比较训练曲线。|
|`LINGBOT_SAMPLE_META_EXPORT`|`0`|导出样本、帧和 CFG 元数据，便于诊断样本顺序和复现实验。|
|`LINGBOT_SKIP_FINAL_CHECKPOINT`|`0`|跳过最终 checkpoint 保存，减少磁盘占用和收尾时间。|

### 2.3 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 LingBot-VA 与官方实现进行了逐 step 的 loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：
![alt text](../../assets/images/precision/Lingbot-va.png)
