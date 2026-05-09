# 自适应 FP8 训练（选择性 FP8）

自适应 FP8（也称为**选择性 FP8**）是 LoongForge 中一种**基于基准测试的动态精度选择**机制。在模型初始化时，它查询预生成的性能策略文件（FP8 Dynamic Policy），以逐模块的方式决定使用 FP8 还是 BF16。这样可以在有效的地方保留 FP8 的加速效果，同时避免在不利的场景下（例如小型 MoE 专家、高 TP 并行度、短序列）出现性能回退。

---

## 0. 背景与动机

全量 FP8 训练（[FP8 训练](fp8_training.md)）对于具有大隐藏尺寸和长序列的 Dense 模型可带来显著的加速。然而，并非每个层或配置都能从 FP8 中受益：

- **MoE Grouped GEMM**：当专家规模较小时，FP8 量化开销可能超过计算收益。
- **高 TP 并行度**：经过张量并行切分后，每个 tile 更小，削弱了 FP8 的优势。
- **短序列 / 小批次**：token 数量不足，FP8 内核无法充分饱和硬件。

自适应 FP8 通过**仅在基准数据确认有加速的层上启用 FP8**来解决此问题，其余层保持 BF16。目标是"永远不比 BF16 慢，尽可能接近全量 FP8"。

---

## 1. 前置条件

| 条目 | 要求 |
|------|------|
| **硬件** | 目标 FP8 硬件平台支持原生 FP8 |
| **软件** | 框架中已启用 Transformer Engine |
| **基线** | 全量 FP8 训练已验证可正常工作（参见 [FP8 训练](fp8_training.md)） |

---

## 2. 工作流程

自适应 FP8 的使用包含两个阶段：**生成策略**和**在训练中启用**。

### 2.1 阶段 1 — 基准测试以生成 FP8 策略

使用 `tools/benchmark_te_parallel_layers.py` 对目标模型在不同 TP/EP 配置下的 TE 并行层进行基准测试，并生成策略文件。

#### 2.1.1 Dense 模型

```bash
# 步骤 1：在每个目标 TP 大小下运行基准测试
for tp in 1 2 4; do
    TE_LAYER_PERF_OMNI_CONFIG_PATH="configs/models/qwen2.5/qwen2_5_72b.yaml" \
    TE_LAYER_PERF_TP_SIZE=$tp \
    TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
    TE_LAYER_PERF_FP8_RECIPE="blockwise" \
    TE_LAYER_PERF_REPORT_PATH="outputs/report_tp${tp}.json" \
    TE_LAYER_PERF_WARMUP=5 \
    TE_LAYER_PERF_ITERS=5 \
        torchrun --nproc_per_node $tp tools/benchmark_te_parallel_layers.py
done

# 步骤 2：将多个 TP 的报告合并为统一策略
python tools/benchmark_te_parallel_layers.py merge-policy \
    --reports outputs/report_tp1.json outputs/report_tp2.json outputs/report_tp4.json \
    --output configs/models/qwen2.5/fp8_policy_qwen2_5_72b.json \
    --speedup-threshold 1.0
```

#### 2.1.2 MoE 模型

MoE 模型需要额外的 EP 覆盖：

```bash
# TP=1, EP=4（需要 4 块 GPU）
TE_LAYER_PERF_OMNI_CONFIG_PATH="configs/models/deepseek3/deepseek_v3.yaml" \
TE_LAYER_PERF_TP_SIZE=1 \
TE_LAYER_PERF_EP_SIZE=4 \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
TE_LAYER_PERF_FP8_RECIPE="blockwise" \
TE_LAYER_PERF_REPORT_PATH="outputs/report_tp1_ep4.json" \
TE_LAYER_PERF_WARMUP=5 \
TE_LAYER_PERF_ITERS=5 \
    torchrun --nproc_per_node 4 tools/benchmark_te_parallel_layers.py

# TP=2, EP=4（需要 8 块 GPU，world_size = tp * ep）
TE_LAYER_PERF_TP_SIZE=2 TE_LAYER_PERF_EP_SIZE=4 \
TE_LAYER_PERF_REPORT_PATH="outputs/report_tp2_ep4.json" \
    torchrun --nproc_per_node 8 tools/benchmark_te_parallel_layers.py

# 合并
python tools/benchmark_te_parallel_layers.py merge-policy \
    --reports outputs/report_tp1_ep4.json outputs/report_tp2_ep4.json \
    --output configs/models/deepseek3/fp8_policy_deepseek_v3.json \
    --speedup-threshold 1.0
```

#### 2.1.3 VLM（视觉语言模型）

基准测试工具会自动从 VLM 配置中提取 ViT 和 LLM 组件：

```bash
TE_LAYER_PERF_OMNI_CONFIG_PATH="configs/models/qwen3_vl/qwen3_vl_235b_a22b.yaml" \
TE_LAYER_PERF_TP_SIZE=1 \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
TE_LAYER_PERF_FP8_RECIPE="blockwise" \
TE_LAYER_PERF_REPORT_PATH="outputs/report_qwen3_vl_tp1.json" \
    torchrun --nproc_per_node 1 tools/benchmark_te_parallel_layers.py
```

#### 2.1.4 基准测试环境变量参考

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `TE_LAYER_PERF_OMNI_CONFIG_PATH` | 模型 YAML 配置路径 | — |
| `TE_LAYER_PERF_TP_SIZE` | 张量并行大小 | 模型默认值 |
| `TE_LAYER_PERF_EP_SIZE` | 专家并行大小 | 模型默认值 |
| `TE_LAYER_PERF_ETP_SIZE` | 专家-张量并行大小 | 模型默认值 |
| `TE_LAYER_PERF_PRECISIONS` | 要基准测试的精度 | `"bf16,fp8"` |
| `TE_LAYER_PERF_FP8_RECIPE` | FP8 recipe | `blockwise` |
| `TE_LAYER_PERF_WARMUP` | 预热迭代次数 | `10` |
| `TE_LAYER_PERF_ITERS` | 计时迭代次数 | `10` |
| `TE_LAYER_PERF_REPORT_PATH` | 报告输出路径 | — |
| `TE_LAYER_PERF_FP8_POLICY_PATH` | 直接策略导出路径 | — |
| `TE_LAYER_PERF_SPEEDUP_THRESHOLD` | 认为 FP8 有益的最低加速比 | `1.0` |
| `TE_LAYER_PERF_SHAPE_SWEEP` | 自定义形状扫描（如 `"1024x1,4096x2"`） | 自动生成 |

---

### 2.2 策略文件格式

生成的策略 JSON 具有以下结构：

```json
{
  "version": 1,
  "speedup_threshold": 1.0,
  "rules": {
    "layernorm_column": {
      "qkv": [{"tp": 1, "min_tokens": 16384, "measured_speedup": 1.03}],
      "fc1": [{"tp": 1, "min_tokens": 4096,  "measured_speedup": 1.18}]
    },
    "row": {
      "proj": [{"tp": 1, "min_tokens": 16384, "measured_speedup": 1.05}],
      "fc2":  [{"tp": 1, "min_tokens": 8192,  "measured_speedup": 1.23}]
    },
    "column_grouped": [
      {"etp": 1, "num_gemms": 64, "min_tokens": 424, "measured_speedup": 1.02}
    ],
    "row_grouped": [
      {"etp": 1, "num_gemms": 64, "min_tokens": 424, "measured_speedup": 1.04}
    ]
  }
}
```

Dense 模块类型（`layernorm_column` / `column` / `row` / `duplicated`）使用嵌套的 `{ub_name: [rules]}` 布局，使相同类型但不同形状的模块（如 `qkv` 与 `fc1`，或 `proj` 与 `fc2`）可以携带不同的阈值。MoE grouped 类型保持扁平列表形式——因为每个专家的 `ub_name` 不是有意义的形状区分符。

**决策逻辑**：

- **Dense 层**（`layernorm_column` / `column` / `row` / `duplicated`）：按 `(module_kind, ub_name, tp)` 查找 → 当 `seq_length × micro_batch_size >= min_tokens` 时启用 FP8。`ub_name` 是模块的 TE tp-comm 缓冲区名称（`qkv` / `proj` / `fc1` / `fc2` / `q_down_proj` / `kv_down_proj`）。
- **MoE 层**（`column_grouped` / `row_grouped`）：按 `(module_kind, etp, num_gemms)` 查找 → token 数量为 `seq_length × micro_batch_size × moe_router_topk`。
- **缺失规则**：如果策略中没有与当前 `(ub_name, tp)` 或 `(etp, num_gemms)` 匹配的条目，则**保守地回退到 BF16**。

---

### 2.3 阶段 2 — 在训练中启用自适应 FP8

#### 2.3.1 独立 LLM（YAML 配置）

在模型 YAML 中添加自适应 FP8 参数：

```yaml
# 示例：configs/models/deepseek3/deepseek_v3_fp8_sel.yaml
_target_: loongforge.models.foundation.DeepseekConfig
defaults:
  - deepseek_v3
  - _self_

fp8: "e4m3"
fp8_recipe: "blockwise"
fp8_param: True
selective_fp8: true
fp8_dynamic_policy_path: "configs/models/deepseek3/fp8_policy_deepseek_v3.json"
```

关键参数：

| 参数 | 说明 |
|------|------|
| `fp8: "e4m3"` | FP8 格式（E4M3） |
| `fp8_recipe: "blockwise"` | 分块量化 recipe |
| `fp8_param: True` | 以 FP8 存储权重（可选，节省内存） |
| `selective_fp8: true` | **启用自适应 FP8** |
| `fp8_dynamic_policy_path` | 策略 JSON 的路径（相对于项目根目录或绝对路径） |

#### 2.3.2 VLM（视觉语言模型）

VLM 模型可以独立配置基座模型（LLM）和图像编码器（ViT）：

```yaml
# 示例：configs/models/qwen3_vl/qwen3_vl_235b_a22b_fp8_sel.yaml
model:
  foundation:
    fp8: "e4m3"
    fp8_recipe: "blockwise"
    fp8_param: True
    selective_fp8: true
    fp8_dynamic_policy_path: "configs/models/qwen3_vl/fp8_policy_235b.json"
  image_encoder:
    fp8: "e4m3"
    fp8_recipe: "blockwise"
    fp8_param: True
    selective_fp8: true
    fp8_dynamic_policy_path: "configs/models/qwen3_vl/fp8_policy_qwen3_vit.json"
```

> **提示**：在多模态模型中，ViT 和 LLM 处理的有效 token 数量不同。请为每个组件显式设置 `fp8_dynamic_num_tokens`，以避免从全局 `seq_length` 进行不准确的自动推断。

#### 2.3.3 启动脚本

使用与全量 FP8 训练相同的命令行参数和 epsilon 防护：

```bash
export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

torchrun --nproc_per_node 8 \
    loongforge/train.py \
    --config-file configs/models/deepseek3/deepseek_v3_fp8_sel.yaml \
    --fp8-format e4m3 \
    --fp8-recipe blockwise \
    --fp8-param-gather \
    ...  # 其他训练参数
```

---

## 3. 工作原理

### 3.1 架构概览

```
启动 → parse_args_from_config
            │
            └─ _register_selective_fp8_decision()
                 │
                 └─ 将 selective_fp8_init_decision 回调注册到 Megatron

模型构建 → 初始化时对每个 TE 模块
            │
            └─ selective_fp8_init_decision(config, te_cls, ub_name, init_kwargs)
                 │
                 ├─ 识别 module_kind（layernorm_column / row / column_grouped / …）
                 ├─ 计算有效 token 数量
                 ├─ 查询策略：should_use_fp8(module_kind, num_tokens, tp, etp)
                 │
                 ├─ True  → 模块保持 FP8
                 └─ False → 模块标记为 _selective_fp8_disabled，以 BF16 运行
```

### 3.2 支持的模块类型

| TE 模块类 | module_kind | 典型用途 |
|-----------|-------------|----------|
| `TELayerNormColumnParallelLinear` | `layernorm_column` | QKV / FC1（融合 LayerNorm） |
| `TEColumnParallelLinear` | `column` | 列并行线性层 |
| `TERowParallelLinear` | `row` | Proj / FC2 行并行 |
| `TEColumnParallelGroupedLinear` | `column_grouped` | MoE 专家 FC1 |
| `TERowParallelGroupedLinear` | `row_grouped` | MoE 专家 FC2 |
| `TELinear` | `duplicated` | MLA 下投影 |

### 3.3 MoE 专家处理

对于 MoE 模型中的专家层：
- `column` / `layernorm_column` 自动提升为 `column_grouped`。
- `row` 自动提升为 `row_grouped`。
- 有效 token 数量乘以 `moe_router_topk` 以反映实际计算量。
- 策略查找使用 `(etp, num_gemms)` 作为键而非 `tp`。

---

## 4. 各场景预期行为

| 场景 | 全量 FP8 | 自适应 FP8 |
|------|----------|------------|
| 大隐藏尺寸（>=8192）的 Dense 模型 | 显著加速 | ≈ 全量 FP8（策略启用所有层） |
| 短序列（<=2048）的 Dense 模型 | 可能回退 | >= BF16（自适应跳过） |
| 小专家的 MoE | 经常回退 | >= BF16（专家层保持 BF16） |
| 高 TP 的 MoE | 明显回退 | 显著优于全量 FP8 |
| VLM（ViT + LLM 混合） | 各组件表现不同 | 各组件独立优化 |

---

## 5. 故障排查

| 症状 | 可能的解决方法 |
|------|----------------|
| loss 或梯度中出现 NaN/Inf | 检查 `FP8_QUANT_*_AMAX_EPS` 变量是否已设置（建议 `1e-12`）。 |
| FP8_SEL 吞吐量低于预期 | 验证策略中的 `min_tokens` 阈值是否匹配实际的 `seq_length × micro_batch_size`。 |
| VLM 中 ViT 的 FP8 决策看起来不正确 | 为 ViT 组件显式设置 `fp8_dynamic_num_tokens`。 |
| 策略中缺少当前 TP/EP 的规则 | 在所需的 TP/EP 下重新运行基准测试并合并到策略中。 |
| FP8_SEL 性能与全量 FP8 相同 | 正常——所有层都从 FP8 中受益，因此策略启用了每个模块。 |
| 基准测试工具 OOM | 减小 `TE_LAYER_PERF_SHAPE_SWEEP` 中的最大 token 数量。 |

---

## 6. 与全量 FP8 的对比

| 方面 | 全量 FP8 | 自适应 FP8 |
|------|----------|------------|
| 配置复杂度 | 低（全局开关） | 中等（需要生成策略） |
| Dense 大模型加速 | 最优 | ≈ 全量 FP8 |
| MoE 安全性 | 有回退风险 | 有保障——永远不比 BF16 慢 |
| 运行时开销 | 无 | 无（初始化时决策；前向路径仅为一次 `getattr` 检查） |
| 最适合 | 已验证均匀受益于 FP8 的模型 | 新模型、混合架构、MoE、VLM |

**建议**：对于新模型或混合架构（MoE、VLM），优先使用自适应 FP8。对于已充分验证全量 FP8 的 Dense 模型，全量 FP8 更简单。
