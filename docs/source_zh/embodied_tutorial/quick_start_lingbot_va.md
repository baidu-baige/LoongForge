# 快速入门：LingBot-VA 模型训练与评测

本文档以 RoboTwin-2.0 和 LIBERO-Long 为例，介绍如何在 LoongForge 框架下从 `lingbot-va-base` 后训练 **LingBot-VA**，并按社区评测协议评测得到的 checkpoint。同硬件、同数据、同 global batch 下，LoongForge 训练比社区代码快 **2.85×（RoboTwin）/ 2.17×（LIBERO）**，训练曲线与社区代码重合，评测成功率不低于社区代码训练的权重。

- [1. 训练准备](#1-训练准备)
- [2. 后训练](#2-后训练)
- [3. 评测准备](#3-评测准备)
- [4. 评测](#4-评测)

---

## 1. 训练准备

### 1.1 权重

| 权重 | 用途 |
|---|---|
| [lingbot-va-base](https://huggingface.co/robbyant/lingbot-va-base) | 后训练的初始化权重（`transformer/`） |
| [lingbot-va-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) | 官方 RoboTwin 后训练权重；评测时提供 `vae/`、`text_encoder/`、`tokenizer/` |
| [lingbot-va-posttrain-libero-long](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long) | 官方 LIBERO-Long 后训练权重；评测时提供 `vae/`、`text_encoder/`、`tokenizer/` |

### 1.2 训练数据

| 数据集 | 地址 |
|---|---|
| RoboTwin LeRobot 数据集 | [robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot) |
| LIBERO-Long LeRobot 数据集 | [robbyant/libero-long-lerobot](https://huggingface.co/datasets/robbyant/libero-long-lerobot) |

本项目采用与 LingBot-VA 官方社区相同的离线预处理方案，数据目录需要包含预生成的 latent 和对应的 `empty_emb.pt`。使用自有数据集时，先将原始机器人数据转换为 LeRobot 格式，再在 `meta/episodes.jsonl` 的每个 episode 中补充 `action_config`（动作片段起止帧和文本描述），将动作映射到标准 30 维格式（不足维度补 0），并使用 Wan2.2 VAE 提取视频 latent，放入与 `videos/` 对应的 `latents/` 目录。详细字段、目录结构和预处理命令请参考官方 [README.md](https://github.com/Robbyant/lingbot-va/blob/main/README.md) 的 **Custom Dataset Preparation** 章节。

---

## 2. 后训练

### 2.1 启动训练

训练脚本：`examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh`、`examples/embodied/lingbot_va/run_lingbot_va_libero_fsdp_finetune.sh`。两个脚本的默认值就是本文结果所用的配方（从 `lingbot-va-base` 起训、FP32 梯度 reduce），只需传入路径：

```bash
cd /path/to/LoongForge

# RoboTwin-2.0：50000 步，8 卡 x GA 8 = global batch 64
CHECKPOINT_PATH=/path/to/lingbot-va-base \
DATA_PATH=/path/to/robotwin-clean-and-aug-lerobot \
OUTPUT_DIR=/path/to/outputs/lingbot_va_robotwin \
SAVE_INTERVAL=2000 \
bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh

# LIBERO-Long：5000 步，8 卡 x GA 10 = global batch 80
CHECKPOINT_PATH=/path/to/lingbot-va-base \
DATA_PATH=/path/to/libero-long-lerobot \
OUTPUT_DIR=/path/to/outputs/lingbot_va_libero \
SAVE_INTERVAL=1000 \
bash examples/embodied/lingbot_va/run_lingbot_va_libero_fsdp_finetune.sh

# 冒烟测试：只跑 20 步验证训练链路
TRAIN_ITERS=20 bash examples/embodied/lingbot_va/run_lingbot_va_robotwin_fsdp_finetune.sh
```

checkpoint 以 DCP 格式保存在 `$OUTPUT_DIR/checkpoints/steps_<N>`；中断后用同一个 `OUTPUT_DIR` 追加 `--resume` 即可续训。脚本末尾的 `"$@"` 会原样透传给 `train.py`，可用于覆盖任意训练参数。

### 2.2 关键参数

**脚本参数**（均可通过环境变量覆盖）：

| 参数 | RoboTwin 默认值 | LIBERO 默认值 | 说明 |
|---|:---:|:---:|---|
| `LOONGFORGE_PATH` | 脚本所在仓库 | 脚本所在仓库 | LoongForge 仓库根目录 |
| `LOCAL_VLA_ARTIFACTS_ROOT` | `/path/to/vla_artifacts` | `/path/to/vla_artifacts` | 下面三个路径的默认根目录 |
| `CHECKPOINT_PATH` | `<root>/lingbot_va/models/lingbot-va-base` | 同左 | 初始化权重目录（含 `transformer/`） |
| `DATA_PATH` | `<root>/lingbot_va/datasets/robotwin-clean-and-aug-lerobot` | `<root>/lingbot_va/datasets/libero-long-lerobot` | 数据集目录（含 `latents/` 与 `empty_emb.pt`） |
| `OUTPUT_DIR` | `$LOONGFORGE_PATH/outputs/lingbot_va_robotwin` | `$LOONGFORGE_PATH/outputs/lingbot_va_libero` | 输出目录 |
| `GPUS_PER_NODE` / `NNODES` | 8 / 1 | 8 / 1 | 每节点 GPU 数 / 节点数 |
| `TRAIN_ITERS` | 50000 | 5000 | 总训练步数 |
| `GRADIENT_ACCUMULATION_STEPS` | 8 | 10 | 梯度累积步数；换卡数时按 global batch 不变调整（如 16 卡时分别为 4 和 5） |
| `SAVE_INTERVAL` | 0 | 0 | checkpoint 保存间隔，0 表示只在结束时保存 |
| `SEED` | 42 | 42 | 全局随机种子，配合 `--set-seed-by-rank` 每个 rank 使用不同种子 |

**影响训练数值的开关**（示例脚本已按推荐值导出）：

| 开关 | 默认值 | 说明 |
|---|:---:|---|
| `LINGBOT_FSDP_BF16_REDUCE` | `0` | `0` 用 FP32 做梯度 reduce；`1` 用参数 dtype（BF16）reduce，省掉每步一次 FP32 reduce-scatter。该开关会覆盖 `--fsdp-reduce-dtype`：action 分支梯度量级远小于 video 分支，RoboTwin 上 BF16 reduce 会使 action loss 升高约 6%、成功率下降，因此默认用 FP32 |
| `LINGBOT_VA_FIXED_MICROBATCH_RNG` | `0` | `0` 与社区训练一样使用进程全局随机流，配合 `--set-seed-by-rank` 每个 rank 各自采样 timestep / 噪声 / chunk / window；`1` 在每个 microbatch 前重设种子，所有 rank 共用一组随机数（用于与打过补丁的社区训练逐 step 对比） |
| `LINGBOT_BALANCED_SAMPLER` | `1` | 按样本计算量在 DP ranks 间做确定性负载均衡，降低慢 rank 等待；会改变样本加载顺序，需要与社区训练逐 step 对齐 loss 时置 `0`（回退到公开 `DistributedSampler` 的切分方式） |
| `LINGBOT_FSDP_RESHARD` | `0` | `0` 表示前反向之间保持参数不 reshard（已验收配置）；`1` 恢复框架默认的 reshard 行为 |

示例脚本还开启了 `--manual-gc --manual-gc-interval 50`、`--detail-log-interval 20`、`--loss-log-rank=-1`，它们只影响 GC 节奏与日志，不改变数值。开启 `--manual-gc` 后，LingBot 会保留新生代回收并抑制 gen2 回收（阈值 `GC_GENERATION2_THRESHOLD`），消除周期性 GC 抖动。

**功能性开关**（用于诊断、复现与产物管理，不改变训练语义）：

| 开关 | 默认值 | 说明 |
|---|:---:|---|
| `LINGBOT_SAMPLE_ORDER_EXPORT_DIR` | 空（不导出） | 把每个 rank、每个 epoch 的采样顺序导出为 JSON（含 seed、balance 组大小、索引序列），用于复现与逐 step 对齐核对 |
| `LINGBOT_REPO_DISCOVERY_CACHE_DIR` | `<dataset_path>/.lingbot_cost_cache` | repo discovery 缓存目录；数据集目录只读或多任务共享数据时改到可写路径 |

### 2.3 训练加速

单机 8×A800、同一环境、同数据、同 global batch 下各训练 50 步，取第 15–45 步的平均步时（LoongForge 使用示例脚本默认的 FP32 reduce 配置）：

| benchmark | 配置 | LoongForge | 社区代码 | 加速比 |
|---|:---:|:---:|:---:|:---:|
| RoboTwin-2.0 | 8 卡 × GA 8（global batch 64） | 8.98 s/step | 25.55 s/step | **2.85×** |
| LIBERO-Long | 8 卡 × GA 10（global batch 80） | 12.49 s/step | 27.13 s/step | **2.17×** |

按此步时推算，RoboTwin 50000 步训练从约 14.8 天缩短到约 5.2 天，LIBERO-Long 5000 步从约 37.7 小时缩短到约 17.3 小时。

加速来自以下固化在 LingBot 实现中的优化（没有关闭开关）：

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

### 2.4 Loss 曲线

从同一 `lingbot-va-base` 出发、同数据、同 global batch 下，LoongForge 与社区代码的 video / action / total loss 与 grad norm 全程重合，loss 分段均值相差在 1% 以内、grad norm 在 2% 以内（社区代码 RoboTwin 前 6000 步的 grad norm 日志口径不同，未画出）：

| RoboTwin-2.0（50000 步，200 步滑动平均） |
|:---:|
| ![RoboTwin loss](../../assets/images/lingbot_va/loss_robotwin.png) |
| **LIBERO-Long（5000 步，50 步滑动平均）** |
| ![LIBERO loss](../../assets/images/lingbot_va/loss_libero_10.png) |

### 2.5 正确性验证

为保证训练精度不受优化手段影响，我们在相同数据、权重和训练配置下，对 LoongForge 适配的 LingBot-VA 与官方实现进行了逐 step 的 loss 对比验证。结果表明，LoongForge 的各项性能优化对训练精度无损：

![LingBot-VA precision](../../assets/images/precision/Lingbot-va.png)

---

## 3. 评测准备

### 3.1 评测环境

评测采用 server / client 架构：LoongForge policy server 加载 checkpoint 进行推理，仿真器作为 client 与其交互，两者可以使用不同的 Python 环境。server 使用训练所用的 LoongForge 环境；client 环境按各 benchmark 的说明安装：

- RoboTwin-2.0：见 [RoboTwin 评测](../../source/embodied_tutorial/eval_docs/benchmarks/robotwin.md)（SAPIEN / Vulkan 环境）。
- LIBERO：见 [LIBERO 评测](../../source/embodied_tutorial/eval_docs/benchmarks/libero.md)；无显示环境可使用 `osmesa` 渲染。

### 3.2 导出评测 checkpoint

训练得到的是 DCP 分片，policy server 需要一个同时包含 `transformer/`、`vae/`、`text_encoder/`、`tokenizer/` 的 checkpoint 根目录。用 `export_eval_checkpoint.py` 一步导出：

```bash
cd /path/to/LoongForge

# RoboTwin：vae / text_encoder / tokenizer 取自官方 RoboTwin 后训练权重
python examples/embodied/lingbot_va/export_eval_checkpoint.py \
    --ckpt  /path/to/outputs/lingbot_va_robotwin/checkpoints/steps_50000 \
    --base  /path/to/lingbot-va-base \
    --codec /path/to/lingbot-va-posttrain-robotwin \
    --out   /path/to/eval_ckpt/lingbot_va_robotwin_steps_50000

# LIBERO-Long：vae / text_encoder / tokenizer 取自官方 LIBERO-Long 后训练权重
python examples/embodied/lingbot_va/export_eval_checkpoint.py \
    --ckpt  /path/to/outputs/lingbot_va_libero/checkpoints/steps_5000 \
    --base  /path/to/lingbot-va-base \
    --codec /path/to/lingbot-va-posttrain-libero-long \
    --out   /path/to/eval_ckpt/lingbot_va_libero_steps_5000
```

脚本会合并 DCP 中的模型张量（保持 FP32）、按 4 GiB 分片写出 safetensors 与索引，沿用初始化权重的 `config.json` 并把 `attn_mode` 设为推理用的 `torch`，并以软链接引入三个冻结组件。评测官方后训练权重时，直接把官方权重目录作为 checkpoint 根即可。

---

## 4. 评测

### 4.1 启动评测

先把评测 YAML 中的 `/path/to/...` 填成实际路径（`server.ckpt_path` 指向上一步导出的目录），再启动。脚本会自动启动和关闭 policy server：

```bash
cd /path/to/LoongForge

# RoboTwin-2.0：默认 configs/robotwin/robotwin_50tasks.yaml，50 个 task 依次评测
REPO_ROOT=$PWD \
BENCHMARK_PYTHON=/path/to/robotwin/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash examples/embodied/lingbot_va/eval/run_robotwin_eval.sh

# LIBERO-Long：默认 configs/libero/libero_10_full.yaml，10 个 task × 50 个 episode
REPO_ROOT=$PWD \
BENCHMARK_PYTHON=/path/to/libero/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash examples/embodied/lingbot_va/eval/run_libero_eval.sh

# 冒烟测试：只评测 1 个 episode
CONFIG=examples/embodied/lingbot_va/eval/configs/robotwin/adjust_bottle.yaml bash examples/embodied/lingbot_va/eval/run_robotwin_eval.sh
CONFIG=examples/embodied/lingbot_va/eval/configs/libero/libero_10.yaml bash examples/embodied/lingbot_va/eval/run_libero_eval.sh
```

RoboTwin 的 Hard 难度只需把 `benchmark.task_config` 从 `demo_clean` 改为 `demo_randomized`。每个 task 的结果写在 `run.output_dir/<task_name>/` 下（`results.jsonl`、`summary.csv`、`suite_summary.csv`）。完整评测耗时较长，可按 task 拆成多个 YAML 分到多张 GPU 并行运行。

### 4.2 评测协议

附带的完整评测 YAML 已按社区 LingBot-VA 评测 client 配置好协议，填路径即可，无需修改以下字段：

| 配置项 | RoboTwin（`robotwin_50tasks.yaml`） | LIBERO（`libero_10_full.yaml`） |
|---|---|---|
| 指令 | `benchmark.instruction_type: seen`（社区 client 固定使用 seen；不写时 runner 默认 unseen） | 任务自带指令 |
| 种子 | `benchmark.start_seed: 10000`，开启 expert check，只评测专家可解的种子 | `run.seed: 7`，episode 种子 = 7 + 1000 × task + idx，每个 episode 重设 |
| 步数上限 | 逐 task 取官方 `task_config/_eval_step_limit.yml`（400–1700），写在 `sweep` 中 | `benchmark.max_steps: 800`，`num_steps_wait: 5` |
| 规模 | 50 task × 50 episode | 10 task × 50 episode |
| 动作接口 | `action_bridge: lingbot_va_ee_quat_16d`（16 维双臂末端位姿） | 7 维 delta 末端动作，连续夹爪 |
| 推理参数 | 官方 `va_robotwin_cfg.py`：25 步 video / 50 步 action，三相机 256×320 | 官方 `va_libero_cfg.py`：20 步 video / 50 步 action，双相机 128×128 |

`adjust_bottle.yaml` 与 `libero_10.yaml` 是单 episode 的冒烟测试配置，不是完整协议。自己编写 RoboTwin YAML 时须显式设置 `benchmark.instruction_type: seen`。

### 4.3 评测结果

RoboTwin-2.0（50 task × 50 episode，50000 步权重）：

| 权重 | Easy（demo_clean） | Hard（demo_randomized） |
|---|:---:|:---:|
| LoongForge 训练 | **87.6%**（2189 / 2500） | **85.7%**（2143 / 2500） |
| 社区代码训练 | 86.1% | 85.1% |

LIBERO-Long（10 task × 50 episode，5000 步权重）：

| 权重 | 成功率 |
|---|:---:|
| LoongForge 训练 | **84.6%**（423 / 500） |
| 社区代码训练 | 77.8%（389 / 500） |

> **注意**：
> - 2500 episode 时 87.6% 的 95% 置信区间约为 86.2%–88.8%，500 episode 时 84.6% 约为 81.2%–87.5%；单个 task 只有 50 episode，逐 task 差异不宜单独解读。
> - 评测存在随机性，每次运行的结果会略有差异。

### 4.4 Rollout 示例

以下为 LoongForge 训练权重在最难任务上的成功 rollout。RoboTwin-2.0（Easy 中成功率最低的 5 个 task）：

| hanging_mug | open_microwave | turn_switch |
|:---:|:---:|:---:|
| ![hanging_mug](../../assets/images/lingbot_va/replay_robotwin_hanging_mug.gif) | ![open_microwave](../../assets/images/lingbot_va/replay_robotwin_open_microwave.gif) | ![turn_switch](../../assets/images/lingbot_va/replay_robotwin_turn_switch.gif) |
| **move_stapler_pad** | **put_bottles_dustbin** | |
| ![move_stapler_pad](../../assets/images/lingbot_va/replay_robotwin_move_stapler_pad.gif) | ![put_bottles_dustbin](../../assets/images/lingbot_va/replay_robotwin_put_bottles_dustbin.gif) | |

LIBERO-Long（成功率最低的 2 个 task）：

| libero_10 task 4 | libero_10 task 6 |
|:---:|:---:|
| ![libero_10 task 4](../../assets/images/lingbot_va/replay_libero_10_task4.gif) | ![libero_10 task 6](../../assets/images/lingbot_va/replay_libero_10_task6.gif) |
| *put the white mug on the left plate and put the yellow and white mug on the right plate* | *put the white mug on the plate and put the chocolate pudding to the right of the plate* |
