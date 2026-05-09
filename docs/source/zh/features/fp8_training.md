# FP8 训练

DeepSeek-V3 模型采用 **Blockwise FP8** 训练：
* 更细粒度的缩放（激活采用 tile 级别，权重采用 block 级别）替代了逐张量量化，降低了量化噪声。
* 实时的 amax 统计减少了延迟更新所带来的分布偏移误差。

本节列出在 LoongForge 中启用该方案所需的特性开关/环境变量，给出经过验证的配置方案，并汇总故障排查建议。

---

## 0. 前置条件

| 条目 | 要求 |
|------|------|
| **硬件** | 原生 FP8 支持 |
| **软件** | 框架中已启用 Transformer Engine |
| **注意事项** | FP8 在数值上更严格 → 在调试配置期间请保持 NaN/Inf/溢出监控处于开启状态 |

---

## 1. 特性开关

### 1.1 命令行参数

| 参数 | 含义 |
|------|------|
| `--fp8-format e4m3` | 对 FP8 张量使用 **E4M3**（4 位指数，3 位尾数）格式。必须与 `--fp8-recipe blockwise` 配合使用。 |
| `--fp8-recipe blockwise` | 启用 **block 级 / tile 级量化**和逐 block/tile 的 amax 追踪。需要 `--fp8-format e4m3`。 |
| `--fp8-param-gather` | 在分布式 gather/通信以及参数缓冲区中保持**权重为 FP8**。降低内存和通信开销，但需要完整的收敛性和权重回归测试。 |

### 1.2 环境变量

| 变量 | 用途 |
|------|------|
| `FP8_QUANT_FWD_INP_AMAX_EPS` | **前向激活** amax 的 epsilon 钳位（避免除零 → NaN）。**默认 0，建议 1e-12** |
| `FP8_QUANT_FWD_WEIGHT_AMAX_EPS` | **前向权重** amax 的 epsilon 钳位，作用同上。 |
| `FP8_QUANT_BWD_GRAD_AMAX_EPS` | **反向梯度** amax 的 epsilon 钳位。如果反向传播中出现 NaN，首先检查此项。 |
| `NVTE_FP8_BLOCK_SCALING_FP32_SCALES` | 设为 `1` 时以 **FP32** 而非 E8M0 存储缩放因子。**不要**在 Blackwell 上启用。 |
| `NVTE_FP8_BLOCK_SCALING_FWD_INP_POWER2` | 设为 `1` 时强制前向激活使用 **E8M0** 缩放。 |
| `NVTE_FP8_BLOCK_SCALING_FWD_WEIGHT_POWER2` | 设为 `1` 时强制前向权重使用 **E8M0** 缩放。 |
| `NVTE_FP8_BLOCK_SCALING_BWD_GRAD_POWER2` | 设为 `1` 时强制反向梯度使用 **E8M0** 缩放。 |

---

## 2. 推荐配置方案

### 阶段 1 — 基线（验证稳定性）
```bash
--fp8-format e4m3 \
--fp8-recipe blockwise
```
训练直至 loss/指标与 BF16 参考一致。

### 阶段 2 — 优化（节省内存）
```bash
--fp8-format e4m3 \
--fp8-recipe blockwise \
--fp8-param-gather
```
重新运行完整的收敛性测试 + 下游评估 + 权重往返验证。

### 通用 epsilon 防护（添加到启动脚本顶部）
```bash
export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12
```

---

## 3. 快速故障排查清单

| 症状 | 可能的解决方法 |
|------|----------------|
| loss 或梯度中出现 NaN/Inf | 逐步提高三个 `*_AMAX_EPS` 值（1e-12 → 1e-10）。 |
| 与 BF16 相比发散 | 先禁用 `--fp8-param-gather`；若仍发散，降低学习率 10-20%。 |
| 权重重载失败 | 确保保存权重时使用了相同的 FP8 标志和 epsilon 值。 |

使用上述开关和 epsilon 防护，LoongForge 中的 Blockwise FP8 训练即可用于生产规模的运行。

---

## 4. 相关内容

对于全量 FP8 可能出现性能回退的场景（小型 MoE 专家、高 TP、短序列），请参阅 [自适应 FP8 训练](adaptive_fp8.md)，了解基于基准测试的逐模块精度选择机制。
