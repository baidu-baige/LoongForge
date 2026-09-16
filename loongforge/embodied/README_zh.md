# LoongForge-Embodied —— 具身模型训练子系统

`loongforge/embodied/` 是一个 **torch 原生的具身模型训练子系统**——涵盖视觉-语言-动作（VLA）策略模型与世界-动作模型（WAM）——在广泛的开源模型支持之上，兼顾生产级的训练性能。

- **Torch 原生架构**：完全基于原生 PyTorch，构建了统一的 `data`、`model`、`trainer` 抽象，可灵活复用或扩展。
- **广泛的开源模型支持**：π0.5、GR00T-N1.6 / N1.7、xVLA、Lingbot-VA、FastWAM、DreamZero、Cosmos3 等均支持微调，且精度与官方 baseline 对齐。
- **高训练吞吐**：典型模型最高可达 2x+ 吞吐提升，集成 `torch.compile`、CUDA Graph、自定义 kernel 与 I/O 优化，并支持 DDP、ZeRO-1、FSDP、HSDP 等多种分布式策略。

---

## 为什么单独拆成一个子系统？

与典型大模型相比，具身模型参数量小得多（通常在 10B 以内），典型的 VLA 多为一个 VLM 叠加一个动作头，其瓶颈不在于模型参数规模。因此 Megatron 的 TP/PP/EP 模型并行在这个规模下收益有限。

基于这一判断，该子系统构建在**原生 PyTorch 的 DDP/FSDP** 引擎之上，拥有独立的配置、训练器、数据、分布式与评测层。它共享 LoongForge 的代码仓库、发布流程与工具链，但与 Megatron 核心刻意保持解耦（不共享 args / parser / core），从而两套栈各自独立演进。下文的核心抽象，都是这个选择的自然结果。

---

## 快速开始

完整的框架使用说明请参阅 [用户手册](../../docs/source_zh/embodied_tutorial/overview.md)，各模型的快速入门：

- [π0.5 (pi05)](../../docs/source_zh/embodied_tutorial/quick_start_pi05.md)
- [GR00T-N1.6](../../docs/source_zh/embodied_tutorial/quick_start_groot_n1_6.md)
- [GR00T-N1.7](../../docs/source_zh/embodied_tutorial/quick_start_groot_n1_7.md)
- [FastWAM](../../docs/source_zh/embodied_tutorial/quick_start_fastwam.md)
- [DreamZero](../../docs/source_zh/embodied_tutorial/quick_start_dreamzero.md)
- [Cosmos3](../../docs/source_zh/embodied_tutorial/quick_start_cosmos3.md)
- [xVLA](../../docs/source_zh/embodied_tutorial/quick_start_xvla.md)
- [Lingbot-VA](../../docs/source_zh/embodied_tutorial/quick_start_lingbot_va.md)

---

## 性能

相较主流开源 baseline 的训练加速比（性能仍在积极优化中，这些数字后续还会持续提升）：

| 模型 | 类型 | Baseline | 加速比 |
|---|---|---|---|
| DreamZero (DROID Wan2.2-5B Full) | WAM | DreamZero | **2.67×** |
| GR00T-N1.6 | VLA | LeRobot | **2.31×** |
| π0.5 | VLA | OpenPI | **2.23×** |
| Lingbot-VA | WAM | LingBot-VA | **1.80×** |
| xVLA | VLA | X-VLA | **1.69×** |

数据反映测量时刻的 baseline 与 LoongForge 版本，可能随实现演进而变化。跨所有模型族的完整基准图表见 [根 README](../../README_zh.md#-性能表现)。

---

## 目录结构

```
loongforge/embodied/
├── train.py                                # 入口：解析参数 → 构建训练器 → 训练
├── train/                                  # 配置系统 + 训练器
│   ├── parser.py                           # 三层配置解析（CLI → YAML → 冻结实例）
│   ├── training_args.py                    # 通用训练参数（单一来源）
│   ├── config_map.py                       # model-name → (YAML, ModelConfig, DataConfig)
│   ├── global_vars.py                      # 冻结的全局配置单例
│   └── trainers/                           # BaseTrainer（模板方法）+ FinetuneTrainer / 各模型专用训练器
├── model/                                  # 模型架构（以 pi05 为例展开）
│   ├── registry.py                         # @register_model + 自动模块导入
│   └── <model>/                            # 每个模型一个目录，如 pi05，一般至少需要包括如下类型文件
│       ├── modeling_<model>.py             # 模型定义（前向 / 损失计算）
│       └── model_configuration_<model>.py  # 模型配置数据类（架构超参）
├── data/                                   # 数据流水线
│   ├── dataloader.py                       # 顶层 dataloader 组装
│   └── datasets/
│       ├── dataset_builder.py              # 数据集构建 + 注册入口
│       ├── sampler_builder.py              # （有状态）分布式采样器
│       ├── lerobot_dataset.py              # 数据集后端（另有 hdf5 / dummy + video_backends）
│       ├── transforms/                     # 共享 transform 框架：base / pipeline / registry / collator
│       └── <model>/                        # 各模型的数据配置 + 自定义数据格式 + 数据处理流程等
├── distributed/                            # DDP/FSDP 封装、分布式上下文、检查点
│   ├── context.py                          # DistributedContext
│   ├── parallel.py                         # wrap_model()：DDP / FSDP
│   └── checkpoint.py                       # safetensors / pt / dcp 保存与加载
├── optimizer/                              # AdamW、LR 调度、梯度裁剪 / NaN 清理
├── eval/                                   # 离线 benchmark 评测（见 eval/README.md）
└── tools/                                  # 辅助工具，例如 dcp_to_safetensors.py
```

---

## 核心抽象

入口 `train.py` 本身一目了然（解析配置 → 构建训练器 → 训练），此处略过，真正的核心是下面四个抽象。

### 1. 模型定义（`model/`）

每个模型一个目录，通过 `@register_model` 注册到统一入口：

- `modeling_<name>.py` —— 架构、前向、损失计算；
- `model_configuration_<name>.py` —— 模型配置数据类（架构超参）；
- 对上层（Trainer / 评测）暴露统一接口，新增模型无需改训练循环。

### 2. 数据集处理（`data/`）

公共能力共享，模型差异下沉到各自目录：

- **公共能力**：数据集读取后端（`lerobot / hdf5 / dummy`）、（有状态）分布式采样、可组合的 transform 框架；
- **模型差异**（`datasets/<name>/`）：该模型特有的数据怎么读（如 `fastwam` 多帧几何）、动作/图像怎么变换、batch 怎么拼；
- **数据配置**：每个模型定义 `DataConfig`（如 `data_configuration_pi05.py`），列出图像尺寸、动作维度、归一化统计等参数，通过 YAML `data:` 段调整或命令行 dotlist 覆盖。

### 3. 训练配置（`train/parser.py`）

配置分三部分，分别对应三个对象：

- **YAML `model:` 段** → `ModelConfig`（定义在 `model_configuration_<name>.py`）→ 解析为 `model_cfg`，承载模型架构参数（层数、维度、动作头等）；
- **YAML `data:` 段** → `DataConfig`（定义在 `data_configuration_<name>.py`）→ 解析为 `data_cfg`，承载数据参数（图像尺寸、动作维度、归一化统计等）；
- **命令行参数** → `TrainingArgs` → 解析为 `training_args`，承载通用训练参数（`--train-iters`、`--lr-base`、`--distributed-strategy` 等）。

解析流程：`--model-name` 经 `config_map.py` 找到对应的 YAML 及 `ModelConfig` / `DataConfig` 类型，YAML 的 `model:` / `data:` 段分别合并进这两个类型，命令行参数填入 `TrainingArgs`；命令行还可用 dotlist 覆盖 YAML 字段（如 `model.action_horizon=64`）。三者最终 `to_object()` 冻结为不可变对象，存入全局单例。

### 4. 分布式 Trainer（`train/trainers/`、`distributed/`）

`BaseTrainer` 用模板方法固化训练全生命周期（`setup → 训练循环 → 单步 → 前向/反向 → 收尾`）：

- **通用能力**：优化器 / LR 调度、梯度裁剪与 NaN 清理、checkpoint 保存与续训、分布式日志、确定性控制；
- **训练器选择**：标准 SFT 用 `FinetuneTrainer`；特殊范式（多流、CUDA Graph、ZeRO-1）自定义子类，并在 `config_map.py` 绑定 `trainer_cls`。`--trainer-type` 仍作为通用 trainer 的回退入口；
- **分布式**：内置多种策略，按需选择——`ddp`（数据并行）、`ddp` + `--zero-optimizer`（ZeRO Stage-1，分片优化器状态）、`fsdp`（全分片）、`hsdp`（混合分片，设 `--hsdp-shard-size`）。

### 新增一个模型

1. 添加 `model/<name>/modeling_<name>.py` 与 `model_configuration_<name>.py`，用 `@register_model` 注册。
2. 添加 `data/datasets/<name>/`，包含 `data_configuration_<name>.py`（DataConfig）、transform 与 collator。
3. 在 `configs/models/embodied/` 下添加 YAML（含 `model:` / `data:` 段），并在 `config_map.py` 中登记（绑定 YAML + ModelConfig + DataConfig）。
4. 若训练范式不同，继承 `BaseTrainer` 并在 `config_map.py` 绑定 `trainer_cls`；否则复用 `FinetuneTrainer`。
5. 在 `examples/` 下添加启动脚本。

---

## 评测

离线 benchmark 评测（LIBERO / CALVIN / SimplerEnv / RoboTwin / ManiSkill）是一个独立模块，详见 [评测用户指南](../../docs/source_zh/embodied_tutorial/eval_user_guide.md)。
