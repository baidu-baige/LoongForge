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
|Self-Flex BlockMask 缓存|按 latent/action 形状、chunk、window、patch 等构成的键缓存 BlockMask，最多保留 256 条。|避免同形状重复构造 mask 的 CPU 元数据与 H2D 拷贝开销。|
|Self-Flex block64 kernel 配置|为 FlexAttention 的 Triton 融合 kernel 固定 block64 配置（FWD=`64,64,4,1`，BWD=`32,32,4,1`），并按 `IS_DIVISIBLE` 走整除快路径（要求 q/k/v 序列长度为 128 的倍数）。|避免退化到 dense attention，降低 attention 的计算量与显存压力。|
|编译版 FlexAttention 与 BlockMask 构造|self-attention 与 `create_block_mask` 均走 `torch.compile(dynamic=True)`；编译不可用时回退 eager 并只告警一次。|减少 attention 路径的 kernel launch 与 Python 调度开销。|
|Layerwise 融合路径编译|modulation prologue、self residual + cross norm、cross residual + FF norm、residual gate、输出 modulation norm 等 6 段 BF16 逐层逻辑统一编译。|把逐层小算子融合，削减 kernel 数与中间张量。|
|Triton RoPE 成对应用|用 Triton kernel 一次完成 q/k 的旋转位置编码（`apply_triton_rope_pair`）。|替掉多次 elementwise 与转置，降低访存与 launch 开销。|
|RoPE 频率与 timestep 频率缓存|旋转频率按 grid 键缓存（上限 16 条），timestep 频率按设备与维度缓存，time/text embed 走编译版实现。|避免每步重算三角函数与投影，缩短前向的 host 侧准备时间。|
|Transformer Engine Q/K RMSNorm|Q/K 归一化使用 `te.RMSNorm` 融合实现。|减少归一化的 kernel launch、访存与中间张量开销。|
|Cross attention 走 SDPA|cross-attention 直接调用 `F.scaled_dot_product_attention`，不进入 flex 路径。|cross-attention 无需块稀疏 mask，避免 flex 路径的额外开销。|
|LeRobot repo discovery 缓存|repo 列表与样本代价按数据签名落盘缓存（默认 `<dataset_path>/.lingbot_cost_cache`），多 rank 通过文件锁协同，最长等待 1800 s、轮询 2 s。|避免每个 rank 反复扫描海量 metadata，压缩启动与 dataloader 初始化时间。|
|FSDP2 嵌套包装与 post-step reshard|按 block + root 顺序嵌套 `fully_shard`，并在 optimizer step 之后再 reshard（`LINGBOT_FSDP_RESHARD=0` 时生效）。|去掉前反向之间的重复 all-gather，减少通信量。|
|编译版 device-side loss guard|NaN/inf 与 loss 缩放检查用 `fullgraph` 编译的 device 端算子完成。|避免每个 micro-batch 因检查而产生 device→host 同步。|

> 注：若开启框架的 `--manual-gc`，LingBot 会保留新生代回收并抑制 gen2 回收（阈值 `GC_GENERATION2_THRESHOLD`），消除周期性 GC 抖动；该行为随框架开关生效，示例脚本默认不开启。

### 2.2 可选项
以下性能优化可由用户控制，示例脚本已按推荐值导出：

|开关|默认值|作用|
|-|-|-|
|`LINGBOT_BALANCED_SAMPLER`|`1`|按样本计算量在 DP ranks 间做确定性负载均衡，并对齐每个 rank 的 microbatch 代价，降低慢 rank 等待；同时会改变样本加载顺序，如需与官方社区逐 step 对齐 loss 请关闭（关闭后回退到公开 `DistributedSampler` 的切分方式）。|
|`LINGBOT_FSDP_RESHARD`|`0`|`0` 表示前反向之间保持参数不 reshard（已验收配置）；置 `1` 恢复框架默认的 reshard 行为。|
|`LINGBOT_FSDP_BF16_REDUCE`|`1`|`1` 用参数 dtype（BF16）做梯度 reduce，省掉每步一次 FP32 reduce-scatter；`0` 用 FP32 reduce。该开关会覆盖 `--fsdp-reduce-dtype`，属于数值选择而不只是速度选项。|

例如关闭可选优化：

以下命令以 RoboTwin 为例；LIBERO 使用 `run_lingbot_va_libero_fsdp_finetune.sh`。

```bash
LINGBOT_BALANCED_SAMPLER=0 \
LINGBOT_FSDP_RESHARD=1 \
LINGBOT_FSDP_BF16_REDUCE=0 \
bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh
```

### 2.3 功能性开关
以下环境变量用于诊断、复现与产物管理，不改变模型训练语义：

|开关|默认值|作用|
|-|-|-|
|`LINGBOT_SAMPLE_ORDER_EXPORT_DIR`|空（不导出）|把每个 rank、每个 epoch 的采样顺序导出为 JSON（含 seed、balance 组大小、索引序列），用于复现与逐 step 对齐核对。|
|`LINGBOT_REPO_DISCOVERY_CACHE_DIR`|`<dataset_path>/.lingbot_cost_cache`|指定 repo discovery 缓存目录；数据集目录只读或多任务共享数据时改到可写路径。|

### 2.4 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 LingBot-VA 与官方实现进行了逐 step 的 loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：
![alt text](../../assets/images/precision/Lingbot-va.png)
