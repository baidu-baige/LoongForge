# 总览

本文档面向使用 LoongForge框架进行具身模型训练的用户，聚焦框架级通用能力，说明框架各模块支持的功能、配置入口及使用方式。具体模型的专用训练方案、数据预处理脚本、性能优化项目请参阅对应的 **快速入门** 文档。

## 1. 使用入口与配置约定

本章说明 embodied 的目录结构与启动脚本约定，并介绍 `TrainingArgs` / `ModelConfig` / `DataConfig` 三类配置对象及其覆盖优先级。

### 1.1 目录说明

| 路径 | 说明 |
| --- | --- |
| `examples/embodied/` | 模型级启动脚本目录，可通过脚本末尾的透传参数覆盖默认配置 |
| `configs/models/embodied/` | 模型默认 YAML 配置目录，包含 `model:` / `data:` 两个顶层配置段 |
| `loongforge/embodied/train.py` | 训练入口，负责解析配置、构建 Trainer 并启动训练 |
| `loongforge/embodied/train/training_args.py` | 通用训练参数定义文件，负责生成 shell CLI |
| `loongforge/embodied/train/config_map.py` | 模型配置路由表，将 `--model-name` 绑定到 YAML、`ModelConfig` 与 `DataConfig` |
| `loongforge/embodied/model/` | 模型组网、模型注册 |
| `loongforge/embodied/data/datasets/` | 数据处理相关功能 |

训练链路如下：

```text
examples/embodied/<model>/run_*.sh
    ↓
loongforge/embodied/train.py
    ↓
parse_train_args()
    ↓
build_model_trainer()
    ↓
trainer.train()
```

### 1.2 启动脚本约定

启动脚本通常负责设置环境变量、路径、分布式参数与模型默认训练参数，并在命令末尾保留 `"$@"`，用于透传用户追加的 shell flag 或 YAML dotlist 覆盖项：

```bash
PYTHONPATH=$LOONGFORGE_PATH:${PYTHONPATH:-} \
torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$LOONGFORGE_PATH/loongforge/embodied/train.py" \
    "${MODEL_CONFIG_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${DISTRIBUTED_TRAINING_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "$@"
```

示例：

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --train-iters 10000 \
    --per-device-batch-size 8 \
    model.action_horizon=64 \
    data.image_size=256
```

### 1.3 配置分层

将配置分为三类对象：

| 配置对象 | 配置入口 | 作用域 | 示例 |
| --- | --- | --- | --- |
| `TrainingArgs` | Shell flag | 训练流程参数 | `--train-iters`, `--lr-base`|
| `ModelConfig` | YAML 或 `model.xxx=...` | 模型结构与训练策略参数 | `model.action_horizon=64` |
| `DataConfig` | YAML 或 `data.xxx=...` | 数据加载与预处理参数 | `data.image_size=256` |

配置优先级为：

```text
dataclass 默认值  <  YAML 配置  <  shell flag / dotlist 覆盖
```

## 2. 数据处理

数据处理模块负责：将数据集样本转换为模型可直接消费的 `PreparedBatch`，用户通过 `TrainingArgs` 指定数据集格式与 DataLoader 行为，通过 `DataConfig` 配置模型相关的数据处理逻辑。

数据处理链路：

```text
Dataset
    ↓
sample-level transform
    ↓
preprocessor / collate_fn
    ↓
PreparedBatch
    ↓
batch.to(device)
    ↓
model.forward(batch)
```

### 2.1 数据集格式

通过 `--dataset-format` 选择数据读取格式，支持 LeRobot、HDF5 和 dummy 三种格式。`dummy_datasets` 可在无真实数据时生成随机样本，用于调试验证。

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 数据格式 | `--dataset-format` | `lerobot_datasets` | `lerobot_datasets`, `hdf5_datasets`, `dummy_datasets` | 选择数据集格式 |
| 数据路径 | `--dataset-path` | `None` | 本地路径或数据集 id | 训练数据路径 |
| 数据 split | `--split` | `train` | 字符串 | 数据集切分 |
| dummy 样本数 | `--num-samples` | `100` | 正整数 | `dummy_datasets` 下生成的样本数 |
### 2.2 LeRobot 数据策略

使用 LeRobot 格式时，可通过以下参数进一步控制数据加载行为：指定数据集格式版本（v2.0 / v2.1 / v3.0）、选择针对不同机器人或任务的构建策略、配置视频解码后端，以及指定 Robot型号以匹配 embodiment 的 action-state 布局。

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| LeRobotdataset 版本 | `--lerobotdataset-version` | `v3.0` | `v2.0`, `v2.1`, `v3.0` | 解析不同 LeRobot 磁盘格式 |
| 数据策略 | `--dataset-strategy` | `default` | `default`, `fastwam`, `groot_n1_7`, `cosmos3_droid`, `dreamzero` | 选择 LeRobot 构建策略 |
| 视频后端 | `--video-backend` | `torchcodec` | `torchcodec`, `decord`, `opencv`, `pyav`, `torchvision_av` | 视频解码实现 |
| robot 类型 | `--robot-type` | `None` | 字符串 | 选择 embodiment / action-state layout |

> **注意：** `--video-backend` 用于指定 `--lerobotdataset-version` v2.x 系列及其变体的数据读取后端；v3.0 系列默认支持 `torchcodec`、`pyav` 两种后端。

### 2.3 DataLoader 行为

本节配置项控制 DataLoader 的 worker 并行度、多进程启动方式、分布式 index 切分与流式读取模式，覆盖从数据预取到超大规模数据集加载的常见需求。

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| worker 数 | `--num-workers` | `4` | 非负整数 | 每个 rank 的 DataLoader worker 数 |
| 预取倍数 | `--dataloader-prefetch-factor` | `2` | 正整数 | 每个 worker 预取的 batch 数；仅在 `--num-workers` 大于 0 时生效 |
| worker seed | `--dataloader-seed-workers` | `False` | 布尔开关 | 是否基于 `--seed` 设置 worker seed |
| 多进程上下文 | `--dataloader-multiprocessing-context` | `None` | `fork`, `spawn`, `forkserver` | DataLoader worker 启动方式 |
| 分布式采样方式 | `--distributed-sampler-mode` | `cyclic` | `cyclic`, `block` | 分布式 sampler 的 index 切分方式（可扩展） |
| 流式读取 | `--streaming` | `False` | 布尔开关 | 使用 streaming / iterable dataset |

## 3. 训练配置

训练配置模块负责解析 shell 参数和 YAML 文件，并生成 `TrainingArgs`、`ModelConfig`、`DataConfig` 三个类型化配置对象。通用训练能力均通过 `TrainingArgs` 暴露为 shell flag。

### 3.1 模型配置选择

支持以下能力：

- 通过 `--model-name` 选择预注册模型，自动绑定 YAML、`ModelConfig` 与 `DataConfig`
- 通过 `--config-file` 指定默认 YAML，使用自定义配置文件
- 通过 `--tokenizer-path` 指定 tokenizer 路径（本地路径或 HF repo id）

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 选择模型 | `--model-name` | `None` | `config_map.py` 中注册的模型名 | 选择模型 schema、默认 YAML、`ModelConfig` 与 `DataConfig` |
| 指定 YAML | `--config-file` | `None` | YAML 文件路径 | 覆盖 `--model-name` 对应的默认 YAML |
| 指定 tokenizer | `--tokenizer-path` | `None` | 本地路径或 HF repo id | 设置 tokenizer 路径，并同步到 `TOKENIZER_PATH` 环境变量 |

> **注意：** 即使使用 `--config-file`，仍需提供 `--model-name`，用于选择结构化配置类。

### 3.2 训练基础参数

本节配置项控制训练规模（迭代步数与 batch 大小）、可复现性（随机种子）与产物输出目录，通过梯度累积在不增加单卡显存的前提下弹性扩大全局 batch。

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 训练总步数 | `--train-iters` | `150000` | 正整数 | optimizer update 步数 |
| 每设备 batch | `--per-device-batch-size` | `4` | 正整数 | 单个 rank forward时的micro-batch |
| 梯度累积 | `--gradient-accumulation-steps` | `1` | 正整数 | 每次 optimizer step 前累积的 micro-batch 数 |
| 随机种子 | `--seed` | `3047` | 整数 | 用于训练初始化与数据随机性控制 |
| 输出目录 | `--output-dir` | `outputs/default` | 路径 | 保存日志、checkpoint 与运行产物 |

全局 batch 计算方式：

```text
global_batch_size = per_device_batch_size * world_size * gradient_accumulation_steps
```

### 3.3 学习率与优化器

学习率与优化器相关能力较多，本节按「基础调度 → 优化器实现 → 分组学习率」的顺序展开。

支持以下能力：

- 基础学习率与分模块独立学习率
- 10 种学习率调度策略，涵盖线性、cosine、polynomial、恒定及带 warmup / min_lr 变体
- 6 种优化器实现，包含标准 AdamW 及多种 CUDA 融合加速变体
- 梯度裁剪与权重衰减，支持对 bias / norm 参数单独分组

#### 3.3.1 基础学习率与调度

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 基础学习率 | `--lr-base` | `2.5e-5` | float | 默认参数组学习率 |
| 学习率策略 | `--lr-decay-style` | `cosine_with_min_lr` | 见下表 | scheduler 类型 |
| warmup 步数 | `--lr-warmup-iters` | `2000` | 非负整数 | 线性 warmup 步数 |
| decay 步数 | `--lr-decay-iters` | `None` | 正整数或不传 | 不传时使用 `--train-iters` |
| 最小学习率 | `--min-lr` | `1e-6` | float | decay 下限 |
| 梯度裁剪 | `--clip-grad` | `1.0` | float，`<=0` 表示关闭 | 最大梯度范数 |
| 权重衰减 | `--weight-decay` | `0.01` | float | decoupled weight decay 系数 |
| 权重衰减分组 | `--weight-decay-grouping` | `all` | `all`, `bias_norm` | 是否对 bias / norm 参数禁用 weight decay |

`--lr-decay-style` 支持 10 种调度策略，涵盖linear、cosine、polynomial等。

| 取值 | 说明 |
| --- | --- |
| `linear` | warmup 后线性衰减 |
| `cosine` | warmup 后 cosine 衰减 |
| `cosine_with_restarts` | cosine 衰减并带 hard restart |
| `polynomial` | polynomial decay |
| `constant` | 恒定学习率 |
| `constant_with_warmup` | warmup 后保持恒定 |
| `inverse_sqrt` | inverse sqrt decay |
| `cosine_with_min_lr` | cosine 衰减到 `--min-lr` |
| `cosine_warmup_with_min_lr` | 带最小 LR 的 cosine warmup |
| `lambda_linear` | 框架自定义 cycle linear scheduler |

#### 3.3.2 优化器实现

通过 `--optimizer` 指定优化器实现，支持标准 AdamW、多种融合加速实现（PyTorch / TransformerEngine / Apex）及 Adam、SGD：

| 取值 | 说明 |
| --- | --- |
| `AdamW`（默认） | 标准 AdamW |
| `TorchFusedAdamW` | PyTorch fused AdamW |
| `TEFusedAdamW` | TransformerEngine FusedAdam |
| `ApexFusedAdamW` | Apex FusedAdam |
| `Adam` | torch Adam |
| `SGD` | torch SGD |

Fused Adam 加速实现说明：

- **TEFusedAdamW**：将参数更新融合为单次 CUDA kernel，显著降低显存带宽压力，依赖 `Transformer-engine`
- **ApexFusedAdamW**：多参数组融合更新，在大模型场景下优化器步骤耗时更低，依赖 `apex`

#### 3.3.3 模块级分组学习率

通过 `--lr-group` 为不同模块配置独立学习率，常用于微调时对 backbone 使用较小学习率、对动作头使用较大学习率，未匹配的参数使用 `--lr-base`。

- **配置项**：`--lr-group`
- **默认值**：`None`（不启用，所有参数统一使用 `--lr-base`）
- **格式**：`module.path=lr,module.path=lr`

示例：

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --lr-base 1.0e-4 \
    --lr-group "model.backbone=1.0e-5,model.action_head=1.0e-4"
```

配置规则：

- 路径匹配顺序敏感，子模块路径应位于父模块路径之前
- 未匹配参数使用 `--lr-base`
- 模块路径以模型实现中的实际属性路径为准

### 3.4 Checkpoint

Checkpoint 模块提供以下能力：

- **权重保存**：支持 safetensors、pt、dcp 三种格式，其中 DCP 可启用异步保存
- **训练状态保存**：持久化 optimizer、scheduler、RNG 与 DataLoader 状态，用于断点续训
- **预训练加载**：可指定外部 checkpoint 初始化模型参数

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 加载预训练权重 | `--pretrained-checkpoint` | `None` | checkpoint 路径 | 用于初始化模型参数 |
| 续训 | `--resume` | `False` | 布尔开关 | 从 `output_dir/checkpoints` 查找最新 checkpoint 并恢复 |
| 保存间隔 | `--save-interval` | `10000` | 非负整数，`0` 表示关闭 | 每 N 个 update step 保存一次 checkpoint |
| 保存格式 | `--save-format` | `safetensors` | `safetensors`, `pt`, `dcp` | checkpoint 文件格式 |
| 保存训练状态 | `--save-training-state` | `True` | 布尔开关 | 保存 optimizer、scheduler、RNG 与 DataLoader 状态 |
| 异步保存 | `--async-save` | `False` | 布尔开关 | DCP 格式下可启用异步保存 |

续训示例：

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --output-dir /path/to/previous_run \
    --resume
```

### 3.5 日志与监控

支持以下监控方式：

- 控制台 metrics 日志，可配置记录间隔与阶段计时详细程度
- W&B 集成，支持 online / offline / disabled 三种模式
- TensorBoard 集成，指定目录即可启用
- 可按 rank 粒度控制 loss 聚合与输出来源

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 日志间隔 | `--log-interval` | `1` | 正整数 | 每 N 步记录 metrics |
| 详细计时间隔 | `--detail-log-interval` | `20` | 非负整数 | 每 N 步记录阶段耗时 |
| 计时日志级别 | `--timing-log-level` | `0` | `0`, `1` | 阶段耗时日志详细程度 |
| W&B 项目 | `--wandb-project` | `loongforge` | 字符串 | W&B project 名称 |
| W&B 模式 | `--wandb-mode` | `disabled` | `online`, `offline`, `disabled` | W&B 启用模式 |
| TensorBoard 目录 | `--tensorboard-dir` | `None` | 路径 | 不传表示关闭 TensorBoard |
| loss 日志 rank | `--loss-log-rank` | `[-1]` | rank 列表，`-1` 表示全局平均 | 控制 loss 聚合与输出来源 |

### 3.6 冻结训练

通过 `--freeze-modules` 冻结指定模块参数，常用于微调时固定视觉编码器或语言模型主干，仅更新动作头等目标模块。模块路径以模型实现中的 `named_modules()` 为准，具体模型的常用冻结路径在对应 Quick Start 中说明。

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 冻结模块 | `--freeze-modules` | 空字符串（不冻结任何模块） | 逗号分隔模块路径 | 将匹配模块参数设置为 `requires_grad=False` |

## 4. 分布式 Trainer

Trainer 模块负责训练生命周期编排，包括分布式上下文初始化、模型构建、权重加载、模型包装、优化器与 scheduler 构建、DataLoader 构建、训练循环、日志、checkpoint 与资源清理。

### 4.1 Trainer 选择

通过 `--trainer-type` 选择训练器，可选值为 `trainer_builder.py` 中注册的 Trainer 类名。默认的 `FinetuneTrainer` 适用于标准单数据流的监督微调；若涉及多数据流、特殊 loss 组合或非标准 step 调度，可在 `trainer_builder.py` 中注册自定义 Trainer 类。

### 4.2 分布式策略

支持两种分布式并行策略：

- **DDP**：标准数据并行，适用于模型与 optimizer state 均可放入单卡的场景；可叠加 ZeRO-1 分片 optimizer state 以节省显存
- **FSDP**：全参数分片，适用于模型、梯度或 optimizer state 超出单卡显存的场景；支持 HSDP（二维 mesh 分片）

训练精度支持 `bfloat16`（默认）、`float16`、`float32`。

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 分布式策略 | `--distributed-strategy` | `fsdp` | `ddp`, `fsdp` | 选择 DDP 或 FSDP |
| 训练精度 | `--dtype` | `bfloat16` | `bfloat16`, `float16`, `float32` | 模型训练 dtype |
| DDP ZeRO-1 | `--zero-optimizer` | `False` | 布尔开关 | DDP 下分片 optimizer state |
| HSDP shard size | `--hsdp-shard-size` | `None` | 正整数 | FSDP 下启用 HSDP |

DDP 示例：

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --distributed-strategy ddp \
    --dtype bfloat16
```

FSDP 示例：

```bash
bash examples/embodied/pi05/run_pi05_fsdp_finetune.sh \
    --distributed-strategy fsdp \
    --dtype bfloat16
```

DDP + ZeRO-1 示例：

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --distributed-strategy ddp \
    --zero-optimizer
```

策略选择建议：

| 场景 | 建议 |
| --- | --- |
| 模型参数与 optimizer state 可完整放入单卡 | DDP |
| 模型可放入单卡，但 optimizer state 显存占用较高 | DDP + ZeRO-1 |
| 模型、梯度或 optimizer state 难以放入单卡 | FSDP |
| 多节点训练，希望参数分片限制在 shard group 内、减少跨节点 FSDP 通信 | FSDP + HSDP |

#### 4.2.1 DDP / ZeRO 通用参数

以下参数提供对 DDP 策略的通信行为和 ZeRO-1 的精细控制，仅在 `--distributed-strategy ddp` 时生效：

- **DDP 行为**：可调节未使用参数检测、静态图优化、bucket 大小与 bucket view，用于减少通信开销或节省内存
- **ZeRO-1**：开启后分片 optimizer state，可进一步配置 bucket view 和 fp32 master 参数维护

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| 未使用参数检测 | `--ddp-find-unused-parameters` | `True` | 布尔开关 | 模型存在条件分支时通常需要保持开启 |
| 静态图优化 | `--ddp-static-graph` | `False` | 布尔开关 | 计算图每步稳定时可开启 |
| bucket view 梯度 | `--ddp-gradient-as-bucket-view` | `False` | 布尔开关 | 复用 DDP bucket 内存 |
| DDP bucket 大小 | `--ddp-bucket-cap-mb` | `None` | 整数 MB | 控制 DDP all-reduce bucket 大小 |
| ZeRO-1 | `--zero-optimizer` | `False` | 布尔开关 | 分片 optimizer state |
| ZeRO bucket view | `--zero-parameters-as-bucket-view` | `False` | 布尔开关 | ZeRO 下复用 bucket 内存 |
| ZeRO master 参数 | `--zero-master-param-dtype` | `none` | `none`, `fp32` | 是否维护 fp32 master 参数 |

示例：

```bash
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh \
    --distributed-strategy ddp \
    --no-ddp-find-unused-parameters \
    --ddp-static-graph
```

#### 4.2.2 FSDP 通用参数

以下参数提供对 FSDP 分片、wrap 策略、dtype、预取和通信的精细控制，仅在 `--distributed-strategy fsdp` 时生效：

- **分片与 reshard**：控制 forward 后是否立即 reshard，并可按模块类覆盖；FSDP2 root group 在 forward 后保持 unsharded
- **wrap 策略**：可手动指定或排除 FSDP unit 类，也可按参数量阈值自动包装
- **dtype 控制**：分片前参数 dtype、all-gather 后 dtype 与梯度 reduce dtype 均可独立配置
- **执行与通信**：可预取相邻 FSDP unit，并可选用 Delta-FP8 压缩 BF16 AllGather 通信

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| HSDP | `--hsdp-shard-size` | `None` | 正整数 | 启用二维 mesh 的 shard 维度 |
| 默认 reshard 策略 | `--fsdp-reshard-default` | `None` | `true`, `false`, `none`, 大于 1 的整数 | 控制 forward 后参数 reshard |
| 按类覆盖 reshard 策略 | `--fsdp-reshard-module-overrides` | `None` | 逗号分隔的 `ClassName=value` | 为指定模块类覆盖 `reshard_after_forward` |
| 指定 wrap 类 | `--fsdp-wrap-modules` | `None` | 逗号分隔模块类名 | 指定 FSDP unit |
| 排除 wrap 类 | `--fsdp-no-wrap-modules` | `None` | 逗号分隔模块类名 | 排除指定模块类 |
| 排除分片的参数 | `--fsdp-ignored-param-names` | `[]` | 空格分隔的参数名子串 | 命中任一子串的冻结参数在每张卡各留一份完整副本 |
| 复制冻结模块类 | `--fsdp-ignore-frozen-module-classes` | `None` | 逗号分隔模块类名 | 将完全冻结的命中模块保留在 FSDP 分片外，避免无效 AllGather；会增加参数副本显存 |
| 复制冻结参数 dtype | `--fsdp-ignored-frozen-param-dtype` | `None` | `fp32`, `bf16`, `fp16` | 可选指定复制冻结参数的 dtype；设置时必须与 `--dtype` 一致 |
| 自动 wrap 阈值 | `--fsdp-min-param-num` | `1000000` | 非负整数 | 自动包装重复层的参数阈值 |
| 原始参数 dtype | `--fsdp-original-param-dtype` | `None` | `fp32`, `bf16`, `fp16` | FSDP 分片前参数 dtype |
| unsharded 参数 dtype | `--fsdp-unshard-param-dtype` | `None` | `fp32`, `bf16`, `fp16` | AllGather 后前向/反向 dtype |
| reduce dtype | `--fsdp-reduce-dtype` | `fp32` | `fp32`, `bf16`, `fp16` | 梯度 reduce dtype |
| 输出 dtype | `--fsdp-output-dtype` | `None` | `fp32`, `bf16`, `fp16` | 可选指定每个 FSDP unit 浮点输出的 dtype |
| cast forward inputs | `--fsdp-cast-forward-inputs` | `True` | 布尔开关 | 是否将输入 cast 到参数 dtype |
| forward 预取距离 | `--fsdp-forward-prefetch-distance` | `0` | 非负整数 | forward 时预取后续已配置 FSDP unit 的数量 |
| backward 预取距离 | `--fsdp-backward-prefetch-distance` | `0` | 非负整数 | backward 时预取前序已配置 FSDP unit 的数量 |
| Delta-FP8 AllGather | `--fsdp-delta-fp8-allgather` | `False` | 布尔开关 | 将 BF16 FSDP2 AllGather 的参数差值按 block 压缩为 FP8；详见[使用文档](../features/delta_fp8_allgather.md) |

复制冻结模块类时，所有命中参数都必须满足 `requires_grad=False`。该模式与 `--init-on-meta` 不兼容；若设置 `--fsdp-ignored-frozen-param-dtype`，其值必须与 `--dtype` 选择的训练计算 dtype 一致。

Delta-FP8 只改变 FSDP 参数通信精度，模型计算仍使用 FSDP 混合精度策略选择的 dtype。使用条件、启动方式、参数说明和验证建议详见 [Delta-FP8 AllGather](../features/delta_fp8_allgather.md)。

### 4.3 稳定性与运行时控制

训练过程中如遇 loss 异常波动、数值不稳定或显存 OOM，可通过以下手段介入：

- loss spike 防护与 NaN/Inf 检查，保障训练健壮性
- 确定性模式与 TF32 禁用，用于结果复现对齐
- 手动 GC 模式，减少 GC 引入的步间抖动

| 功能 | 配置项 | 默认值 | 取值 / 类型 | 说明 |
| --- | --- | --- | --- | --- |
| loss spike 防护 | `--loss-spike-threshold` | `100.0` | float | loss 超过阈值或为 NaN/Inf 时，该次 loss 贡献置零 |
| NaN / Inf 检查 | `--check-for-nan-in-loss-and-grad` | `True` | 布尔开关 | 检查 loss 与 gradient 中的异常值 |
| 确定性模式 | `--deterministic-mode` | `False` | 布尔开关 | 启用确定性算法 |
| 禁用 TF32 | `--disable-tf32` | `False` | 布尔开关 | 禁用 CUDA TF32 |
| 手动 GC | `--manual-gc` | `False` | 布尔开关 | 关闭自动 GC 并改为显式触发 |
| 手动 GC 间隔 | `--manual-gc-interval` | `0` | 非负整数 | `--manual-gc` 启用后每 N 步执行 GC |
