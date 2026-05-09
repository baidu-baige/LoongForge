# MCore Bridge：在线 HF 权重加载与保存

本模块提供 HuggingFace (HF) 格式权重的在线加载和保存。它允许你在训练启动时直接读取 HF 模型权重，无需任何离线转换，并可在训练后可选地将权重导出回 HF 格式。

> **致谢**：本模块灵感来自 [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)。

## Bridge 是什么？

在标准工作流中，训练前需要单独的离线转换步骤（`HF → Megatron`）。Bridge 消除了这一步骤——它检测你的权重目录中的 HF safetensors，并在启动时**即时**将其转换为 Megatron 格式。这为你节省了：

- **时间** — 无需等待离线转换任务完成
- **存储** — 无需同时保留同一权重的 HF 和 Megatron 两份副本
- **复杂度** — 更少的脚本运行和更少的路径管理

## 支持的模型

理论上，离线转换支持的模型也均支持在线加载。以下模型已经过测试验证：

| 类型 | 模型 |
|------|------|
| **LLM** | LLaMA 3/3.1, Qwen 2.5/3, DeepSeek V2 Lite |
| **VLM** | Qwen2.5-VL, InternVL 2.5/3.5, LLaVA-OV 1.5 |

## 支持的并行策略

| 策略 | 参数 | 说明 |
|----------|------|------|
| TP | `--tensor-model-parallel-size` | 张量并行 |
| PP | `--pipeline-model-parallel-size` | 流水线并行 |
| 自定义流水线 | `--custom-pipeline-layers` | 自定义流水线层分布 |
| EP | `--expert-model-parallel-size` | 专家并行（MoE） |
| ETP | `--expert-tensor-parallel-size` | 专家张量并行 |
| VPP | `--num-virtual-stages-per-pipeline-rank` | 虚拟流水线并行 |
| 异构 TP | `--encoder-tensor-model-parallel-size` | 编码器和解码器使用不同 TP（仅 VLM） |

## 快速开始

### 前提条件

确保以下环境变量已设置并**导出**（而非仅本地赋值）：

```bash
export MEGATRON_PATH=/path/to/Loong-Megatron
export LOONGFORGE_PATH=/path/to/LoongForge  # 本仓库
export CHECKPOINT_PATH=/path/to/Qwen2.5-7B-Instruct  # 包含 config.json, model*.safetensors, tokenizer 文件的 HF 模型目录
```

> **重要**：`LOONGFORGE_PATH` **必须**被导出。模型 YAML 配置依赖此变量在运行时通过 Hydra 的 `oc.env` 解析器解析路径，例如：
>
> ```yaml
> hydra:
>   searchpath:
>     - file://${oc.env:LOONGFORGE_PATH}/configs/models/
>
> convert_file: ${oc.env:LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/internvl_vit_0.3b_convert.yaml
> ```
>
> 如果 `LOONGFORGE_PATH` 未导出，配置解析将在启动时失败并抛出 `KeyError` 或类似错误。

### 步骤 1：验证 HF 模型目录

确保你的权重目录包含标准 HF 权重文件：

```
$CHECKPOINT_PATH/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-xxxxx.safetensors
└── tokenizer files...
```

> **提示**：如果你想保持原始 HF 目录整洁，可以将 `--load` 指向 HF 目录，将 `--save` 指向单独的输出目录。Bridge 将从 `--load` 加载 HF 权重并将 MCore 权重写入 `--save`。但是，恢复训练时需要将 `--load` 重新指向 MCore 权重目录。

### 步骤 2：编写训练脚本

将 `--load` 指向 HF 模型目录，将 `--save` 指向 MCore 权重存储位置（可以是同一路径）。完整示例见 `examples/qwen2.5/pretrain/pretrain_qwen2.5_7b_bridge.sh`。

关键参数：

```bash
TRAINING_ARGS=(
    --load $CHECKPOINT_PATH     # HF 模型目录路径
    --save $CHECKPOINT_PATH     # MCore 权重输出路径（可与 --load 不同）
    --save-interval 40          # 每 40 步保存 MCore 权重
    --save-hf true              # 可选：训练后导出 HF 权重
    --save-hf-path /path/to/output  # 可选：默认为 <save>/release_hf_weights/
    ...
)
```

### 步骤 3：启动训练

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun --nproc_per_node 8 --nnodes 1 \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} ${DATA_ARGS[@]} ${TRAINING_ARGS[@]} ...
```

- **首次运行**：目录中没有 `latest_checkpointed_iteration.txt`。Bridge 自动加载 HF 权重并即时转换为 MCore 格式。
- **后续运行**：存在 `latest_checkpointed_iteration.txt`。系统自动加载最新的 MCore 权重（恢复训练）。

## 加载机制

系统根据 `latest_checkpointed_iteration.txt` 是否存在来判断加载 HF 权重还是 MCore 分片：

| 目录状态 | 行为 |
|----------------|----------|
| 无 `latest_checkpointed_iteration.txt`，有 HF 权重 | **Bridge 在线加载** — 将 HF safetensors 转换为 MCore 格式 |
| 有 `latest_checkpointed_iteration.txt`，有 MCore 分片 | 加载 MCore 分片（标准行为） |
| 两者都存在 | **MCore 分片优先** — 不重新加载 HF 权重 |

### 训练过程中的目录结构

```
# 首次运行前（纯 HF 目录）
$CHECKPOINT_PATH/
├── config.json
├── model.safetensors.index.json
└── model-00001-of-xxxxx.safetensors

# 训练保存权重后
$CHECKPOINT_PATH/
├── latest_checkpointed_iteration.txt    # 自动生成
├── config.json                          # 原始 HF 文件保留
├── model.safetensors.index.json         # 原始 HF 文件保留
├── model-00001-of-xxxxx.safetensors     # 原始 HF 文件保留
└── iter_0000040/                        # 新的 MCore 权重

# 恢复训练：再次运行相同脚本
# → 系统检测到 latest_checkpointed_iteration.txt，从 iter_0000040/ 恢复
```

## 保存机制

### MCore 权重保存（默认）

MCore 权重每 `--save-interval` 步保存一次。此行为与标准训练工作流一致。

### HF 权重导出（可选）

由 `--save-hf` 参数控制。训练完成后将权重导出为 HF 格式。

```bash
TRAINING_ARGS=(
    --save $CHECKPOINT_PATH         # MCore 权重路径
    --save-hf true                  # 启用 HF 导出
    --save-hf-path /path/to/output  # 可选：默认为 <save>/release_hf_weights/
)
```

结果目录结构：

```
$CHECKPOINT_PATH/
├── latest_checkpointed_iteration.txt
├── iter_xxxx/                    # MCore 权重
├── model-00001-of-xxxxx.safetensors  # 原始 HF 权重（如有）
└── release_hf_weights/           # 训练后导出的 HF 权重
    ├── model.safetensors.index.json
    └── model-00001-of-xxxxx.safetensors
```

> 注意：HF 导出仅保存模型权重，不包含优化器状态。

## 权重恢复

权重恢复完全自动——只需再次运行相同的训练脚本，无需额外参数。

```
检测 latest_checkpointed_iteration.txt
  → 读取最新迭代（如 40）
  → 从 iter_0000040/ 加载模型权重、优化器状态、RNG 状态
  → 从迭代 41 继续训练
```

### 重要说明

- 并行配置（TP、PP、EP 等）必须与中断的运行保持一致
- `--train-iters` 应设置为目标总迭代数；系统自动从上次中断处继续
- 可选参数：
  - `--ckpt-step 80` — 从特定迭代恢复（而非最新）
  - `--no-load-optim` — 跳过优化器状态加载
  - `--no-load-rng` — 跳过 RNG 状态加载

## VLM 异构 TP 配置

对于视觉语言模型（VLM），编码器和解码器可以使用不同的 TP 大小。

**步骤 1**：在模型 YAML 中配置编码器 TP：

```yaml
# configs/models/<model>/<model>.yaml
model:
  image_encoder:
    tensor_model_parallel_size: 2    # encoder TP = 2
```

**步骤 2**：在训练脚本中配置解码器 TP：

```bash
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4            # decoder TP = 4
    --encoder-tensor-model-parallel-size 2    # encoder TP = 2
    --pipeline-model-parallel-size 2
)
```

## 端到端流程概览

```
HF 模型目录 (safetensors)
  │
  │  首次运行 → bridge 在线加载，自动转换为 MCore 格式
  ▼
训练循环
  │
  │  每 --save-interval 步 → 保存 MCore 权重
  │  自动生成 latest_checkpointed_iteration.txt
  ▼
$CHECKPOINT_PATH/iter_XXXXXXX/ (MCore 格式)
  │
  ├── 中断后重启 → 自动加载最新 MCore 分片（恢复）
  │
  ├── 训练完成 → [--save-hf true] 导出 HF 权重到 release_hf_weights/
  ▼
HF 模型 (safetensors)
```

## 常见问题

**Q：能否保持原始 HF 模型目录不变？**

可以。将 `--load` 指向原始 HF 目录，将 `--save` 指向新目录。Bridge 将从 `--load` 加载 HF 权重并将 MCore 权重写入 `--save`。但是，恢复训练时需要将 `--load` 重新指向 MCore 权重目录。

**Q：Bridge 是否支持 SFT（监督微调）？**

支持。在训练参数中设置 `--training-phase sft`。加载和保存行为与预训练相同。

**Q：恢复训练时能否使用不同的 TP/PP 配置？**

不能。并行配置（TP、PP、EP 等）必须与中断的运行保持一致。更改并行配置需要从 HF 权重新加载。

## 往返测试

### 概述

往返测试验证 HF 权重在完整的 `HF → MCore → HF` 往返转换后是否保持数值一致。它复用与实际训练完全相同的模型构建、加载和保存流水线，但执行**零训练步骤**。结果直接反映 Bridge 转换的正确性。

### 测试流程

```
1. initialize_loongforge_megatron    # 初始化 Megatron 环境（与训练相同）
       ↓
2. get_model()                  # 构建模型（与训练相同）
       ↓
3. load_hf_checkpoint_online()  # 在线加载原始 HF 权重
       ↓
4. save_hf_checkpoint_online()  # 保存回 HF 格式（往返输出）
       ↓
5. compare_weights()            # 比较原始权重与往返权重（仅 rank 0）
```

### 比较标准

| 级别 | 容差 | 含义 |
|-------|-----------|---------|
| **Exact** | `rtol=1e-4, atol=1e-6` | 完全匹配 |
| **Close** | `rtol=1e-3, atol=1e-4` | 近似匹配（典型的 bfloat16 精度差异） |
| **Diff** | 超出上述容差 | 不匹配 — 需要排查 |

**通过条件**：`num_diff == 0`，且无缺失键、多余键或形状不匹配。

### 运行单个模型测试

以 Qwen2.5-0.5B 为例：

```bash
bash tools/dist_checkpoint/test/qwen2.5/0.5b_bridge_roundtrip.sh
```

测试脚本中的关键参数：

```bash
TRAINING_ARGS=(
    --training-phase pretrain
    --train-iters 0              # 不训练 — 仅加载 + 保存
    --no-load-optim              # 跳过优化器状态
    --no-load-rng                # 跳过 RNG 状态
    --load $TOKENIZER_PATH       # 原始 HF 权重目录
    --save-hf-path $SAVE_HF_PATH # 往返输出目录
    --save-hf true                # 启用 HF 导出
    --bf16                       # 使用 bf16 精度
)
```

> 注意：入口点是 `hf_roundtrip_test.py`，**不是** `loongforge/train.py`：
>
> ```bash
> PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
>     torchrun --nproc_per_node 4 \
>     $LOONGFORGE_PATH/tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py \
>     ${MODEL_ARGS[@]} ${TOKENIZER_ARGS[@]} ${TRAINING_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]}
> ```

### 运行某个模型系列的所有测试

```bash
# Qwen2.5（所有尺寸）
bash tools/dist_checkpoint/test/qwen2.5/all.sh

# Qwen3（所有尺寸）
bash tools/dist_checkpoint/test/qwen3/all.sh

# InternVL 2.5（所有尺寸）
bash tools/dist_checkpoint/test/internvl2.5/all.sh
```

### 输出报告

测试完成后，在 `--save-hf-path` 目录下生成 `roundtrip_comparison.json`：

```json
{
  "passed": true,
  "num_baseline": 361,
  "num_roundtrip": 361,
  "num_common": 361,
  "missing_keys": [],
  "extra_keys": [],
  "shape_mismatches": [],
  "num_exact_matches": 361,
  "num_close_matches": 0,
  "num_different": 0,
  "mismatched_keys": [],
  "max_abs_diff": 0.0,
  "mean_abs_diff": 0.0
}
```

### 可用的测试脚本

测试按模型系列组织在 `tools/dist_checkpoint/test/` 下：

| 模型系列 | 路径 |
|-------------|------|
| Qwen 2.5 | `test/qwen2.5/` |
| Qwen 3 | `test/qwen3/` |
| Qwen 2.5-VL | `test/qwen2.5vl/` |
| DeepSeek V2 | `test/deepseek2/` |
| DeepSeek V3 | `test/deepseek3/` |
| LLaMA 3 | `test/llama3/` |
| LLaMA 3.1 | `test/llama3.1/` |
| InternVL 2.5 | `test/internvl2.5/` |
| InternVL 3.5 | `test/internvl3.5/` |
| LLaVA-OV 1.5 | `test/llavaov1.5/` |

### 模型提供函数选择

测试代码默认使用 `omni_model_provider`（兼容所有模型）。如需切换：

- **纯 LLM**（LLaMA、Qwen2.5、DeepSeek V2 等）：使用 `llm_model_provider`
- **多模态**（Qwen2.5-VL、InternVL 等）：使用 `omni_model_provider`

编辑 `tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py` 中的 `get_model()` 调用：

```python
# 纯 LLM：
model = get_model(llm_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

# 多模态（默认）：
model = get_model(omni_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
```

### 重要说明

- 往返测试**不**需要训练数据 — 只需要 HF 权重目录和 tokenizer 路径
- GPU 数量必须满足并行配置（TP x PP <= 可用 GPU 数）
- 会生成 cProfile 性能报告（`profile_stats.prof`）用于诊断加载/保存瓶颈
- 如果发现权重不匹配，查看日志中的 `[DIFF]` 条目获取张量名称和差值以定位转换问题

## 示例脚本

| 模型 | 路径 |
|-------|------|
| Qwen2.5 7B | `examples/qwen2.5/pretrain/pretrain_qwen2.5_7b_bridge.sh` |
| DeepSeek V2 Lite | `examples/deepseek_v2/pretrain/pretrain_deepseek_v2_lite_group_bridge.sh` |
