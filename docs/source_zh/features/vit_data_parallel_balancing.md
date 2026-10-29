# ViT 编码器 DP 负载均衡
`vit_encoder_dp_balance` 是一种**轻量级、与架构解耦**的工程方案，针对视觉语言模型中**视觉编码器（ViT）内部的 DP 负载不均衡问题**。
它在 ViT 前向计算之前将图像 Token 在 DP 各 rank 之间重新分配，使每个 rank 处理相似的工作量，从而显著提升**大规模 VLM 训练**的 GPU 利用率。

---

## 1. 背景与问题
在 **VLM（视觉语言模型）** 训练中，每个样本可能包含分辨率差异巨大的图像/视频。视觉编码器（ViT）在 LLM 骨干网络**之前**处理这些视觉输入。

ViT 对单张图像的自注意力计算代价与视觉 Token 数量的平方成正比：

* load(image<sub>i</sub>) ∝ (T<sub>i</sub> × H<sub>i</sub> × W<sub>i</sub>)²

其中 **T**、**H**、**W** 分别是 `image_grid_thw` 中的时间、高度和宽度网格维度。

由于不同的 DP rank 接收到不同分辨率的图像，各 rank 之间的 ViT 工作负载可能**高度不均衡**——即使各 rank 的 LLM Token 总数是均衡的。

轻负载的 rank 必须在 ViT 前向计算后的同步屏障处等待重负载的 rank，产生**落后者（straggler）**，降低整体训练吞吐量。

该问题与 VLM 级别的 DP 负载均衡（`--use-vlm-dp-balance`）不同，后者是为 LLM 骨干网络重新排序打包序列。这里的负载不均衡**特指 ViT 编码器阶段**，需要重新分配的是**视觉 Token**而非文本 Token。

---

## 2. 方案概述
核心思想是在 ViT 前向计算之前，**根据各图像的 ViT 计算代价将图像 Token 在 DP 各 rank 之间重新分配**，并在前向计算之后**反向还原分配**，以恢复原始数据布局供下游 LLM 处理。

工作流程分为三个阶段：

1. **收集与规划** — 从所有 DP rank 收集每张图像的 Token 数量，使用二次代价函数估计每张图像的计算代价，求解负载均衡的分配方案。
2. **重新分配与前向计算** — 通过 `all_to_all` 在各 rank 之间重新分配 `pixel_values` 和 `image_grid_thw`，然后在均衡后的数据上执行 ViT 前向计算。
3. **反向还原** — ViT 前向计算完成后，将输出 Embedding（包括 `deepstack_pixel_embeds`，如果存在）反向重新分配回原始 DP rank，并在需要时重新生成 `window_index`。

关键特性：

* **与模型架构解耦** — 仅操作 ViT 的输入/输出张量
* **不影响收敛** — 仅改变哪个 rank 计算哪张图像；数学结果完全相同
* **支持梯度反向传播** — 对需要梯度的张量使用 `torch.distributed.nn.functional.all_to_all`，确保通过重新分配操作的反向传播正确性
* **自动跳过** — 如果不均衡比率低于 20% 或单张图像主导了平均负载，则跳过重新分配以避免不必要的通信开销
* **独立微批次均衡** — 与 VLM 级别均衡不同，ViT 模式不会跨微批次累积代价（`cross_micro_batch_balance=False`），因为每个微批次可以在没有打包约束的情况下独立地良好均衡

---

## 3. 使用方法
在训练启动脚本中添加以下参数：

```bash
--use-vit-dp-balance
```

此功能适用于使用 `OmniEncoderModel` 架构的 **VLM 模型**（例如 Qwen2-VL、Qwen3-VL 及其他基于 `image_grid_thw` 的 ViT 编码器模型）。

它可以**独立使用，也可以与** VLM 级别的 DP 负载均衡（`--use-vlm-dp-balance`）**组合使用**。

### 完整命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `--use-vit-dp-balance` | bool | `False` | 启用 ViT 编码器 DP 负载均衡 |
| `--dp-balance-max-len-ratio-vit` | float | `None` | ViT 级别均衡的最大序列长度比率。限制每个 DP rank 的最大 Token 数为 (avg_len × ratio)。`None` 表示禁用此约束（ViT 模式的默认值）。 |
| `--dp-balance-trigger-threshold-vit` | float | `0.2` | 触发 ViT 级别均衡的最小不均衡比率阈值。当不均衡比率 < 阈值时跳过。 |
| `--dp-balance-verbose` | bool | `False` | 打印每次迭代的诊断信息：不均衡比率、各 DP rank 负载、重排序决策 |

---

## 4. 核心设计

### 4.1 入口点与集成

ViT DP 均衡通过 `dp_balance_vit_encoder()`（定义在 `vit_balance.py` 中）在 `OmniEncoderModel.encode_images()` 内部调用。该函数包装了 ViT 模块调用：

```python
### Inside OmniEncoderModel.encode_images():
if args.use_vit_dp_balance:
    pixel_embeds, window_index, deepstack = dp_balance_vit_encoder(
        vit_module, pixel_values, image_grid_thw
    )
else:
    pixel_embeds, window_index, deepstack = vit_module(pixel_values, image_grid_thw)
```

与 VLM 级别均衡（使用 monkey-patching）不同，ViT 均衡是直接函数调用——**不需要预热阶段或代价模型拟合**，因为 ViT 的代价模型是简单的二次函数，无需校准。


### 4.2 代价估计
每张图像的 ViT 计算代价估计为：

cost(image<sub>i</sub>) = num\_tokens<sub>i</sub>²

其中 num\_tokens<sub>i</sub> = T<sub>i</sub> × H<sub>i</sub> × W<sub>i</sub>，来自 `image_grid_thw`。

该二次模型反映了 ViT 编码器内部的自注意力复杂度。无需预热分析——二次假设是 ViT 架构的直接推论。

### 4.3 负载均衡分配
求解器（`solve_sample_dp_reorder_plan`）使用与 VLM 级别均衡相同的**贪心 LPT（最长处理时间优先）算法**，但在 **ViT 模式**下（无打包约束）：

1. **排序** — 将所有图像按代价降序全局排序
2. **贪心分配** — 将每张图像分配给当前总代价最低的 DP rank（无打包长度约束）
3. **优化** — 通过在最重负载和最轻负载的 rank 之间迭代交换/移动图像，直到不均衡降至容差阈值以下（默认 **5%**，最多 **20** 次迭代）

**跳过条件** — 满足以下条件时跳过重新分配：
- **不均衡比率 < 阈值**：当前分配已经足够均衡（默认阈值：**0.2**，由 `--dp-balance-trigger-threshold-vit` 控制）
- **单张图像主导**：最高的单图代价超过平均 DP 负载，意味着任何重新分配都无法显著改善均衡

求解器与 VLM 级别均衡共享，但配置不同：

| 参数 | VLM 模式 | ViT 模式 |
|-----------|----------|----------|
| `cost_fn` | 校准后的 `a·l²+b·l+c` | 纯 `l²` |
| `pack_len_ratio` | 1.2 (`--dp-balance-max-len-ratio-vlm`) | `None`（无约束，`--dp-balance-max-len-ratio-vit`） |
| `cross_micro_batch_balance` | `True` | `False` |

---

### 4.4 详细的重新分配流程

重新分配通过 `all_to_all` 通信实现，分为 6 个步骤：

**步骤 1 — 计算每张图像的长度并跨 DP 收集：**
```
vit_input_lengths[i] = T[i] * H[i] * W[i]  for each local image
→ gather_sample_info_across_dp() → global lengths, local indices, source ranks
```

**步骤 2 — 求解重排序方案：**
```
solve_sample_dp_reorder_plan(cost_fn=λl: l², cross_micro_batch_balance=False)
→ plan[dst_rank] = [(local_idx, src_rank), ...] or None (skip)
```

**步骤 3 — 前向重新分配 `pixel_values`：**
```
Split pixel_values by per-image lengths
→ redistribute_tensors() via all_to_all_single
→ Concatenate into balanced pixel_values
```

**步骤 4 — 前向重新分配 `image_grid_thw`：**
```
Split image_grid_thw per row → redistribute → concatenate
```

**步骤 5 — ViT 前向计算：**
```
pixel_embeds, window_index, deepstack_pixel_embeds = vit_module(
    balanced_pixel_values, balanced_image_grid_thw
)
```

**步骤 6 — 反向重新分配：**
```
Compute reverse_reorder_plan from forward plan
→ Split pixel_embeds by per-image output lengths
→ redistribute back to original ranks via all_to_all_single
→ For each layer in deepstack_pixel_embeds:
    split by merged spatial dims → redistribute → concatenate
→ Regenerate window_index from original (un-reordered) image_grid_thw
```

> **关于 `deepstack_pixel_embeds` 的说明**：当存在时（非空列表），deep stack 特征的空间维度与 `pixel_embeds` 不同，因为空间合并会将 H 和 W 缩小 `spatial_merge_size` 倍。反向重新分配通过从合并后的网格维度计算单独的特征长度来处理此情况。

> **关于 `window_index` 的说明**：反向重新分配后，`window_index` 从**原始** `image_grid_thw`（而非重排序后的）重新生成，因为下游 LLM 以原始 DP 分配顺序处理图像。

---

### 4.5 梯度支持

对于需要梯度的张量（例如反向传播时的 `pixel_embeds`），重新分配使用 `torch.distributed.nn.functional.all_to_all` 而非 `dist.all_to_all_single`。这确保了通过重新分配操作的梯度流正确性，使 ViT 编码器在反向传播期间能够正确接收梯度。

实现会自动检测发送张量的 `requires_grad` 属性，并选择适当的通信原语。

---

### 4.6 与 VLM DP 负载均衡的关系

| 特性 | VLM DP 均衡 (`--use-vlm-dp-balance`) | ViT DP 均衡 (`--use-vit-dp-balance`) |
|---------|--------------------------------------|------------------------------------------|
| **目标** | LLM 骨干网络（Attention + MLP） | ViT 编码器 |
| **均衡单元** | 打包的文本序列 | 单张图像/视频 |
| **代价模型** | 预热分析校准 (a·l² + b·l + c) | 纯二次 (l²) |
| **需要预热** | 是（默认 10 次迭代） | 否 |
| **约束** | 打包长度比率 ≤ 1.2× (`--dp-balance-max-len-ratio-vlm`) | 无 (`--dp-balance-max-len-ratio-vit`) |
| **跨微批次** | 是（累积代价） | 否（每个微批次独立） |
| **集成方式** | 在 pin_memory_loop 上 monkey-patch | 在 encode_images() 中直接调用 |
| **时机** | `get_batch` 之后，forward 之前 | 编码器 forward 内部 |
| **梯度支持** | 不需要（forward 之前） | 是（带 autograd 的 all_to_all） |
| **触发阈值** | `--dp-balance-trigger-threshold-vlm` (0.2) | `--dp-balance-trigger-threshold-vit` (0.2) |

两个功能可以**同时启用**以获得最大吞吐量提升：ViT 均衡确保 ViT 计算均匀，而 VLM 均衡确保 LLM 计算均匀。

---

## 5. 支持的模型
该功能集成在 `OmniEncoderModel.encode_images()` 中，支持所有使用基于 `image_grid_thw` 的 ViT 编码器的 VLM 模型，包括：

* **Qwen2-VL** / **Qwen3-VL**
* 其他基于 `OmniEncoderModel` 架构构建的模型

---

## 6. 常见问题排查

| 症状 | 可能原因 / 解决方法 |
|---------|-------------------|
| 没有明显加速 | 不均衡比率 < 0.2 → 求解器跳过重新分配（预期行为）。检查图像是否具有相似分辨率。使用 `--dp-balance-verbose` 查看不均衡比率。 |
| 启用后部分 rank OOM | 某个 rank 接收了过多大图像。LPT 求解器下不太可能出现，但极端异常值可能导致。考虑与 `--use-vlm-dp-balance` 组合使用。 |
| ViT 前向计算后形状不匹配 | 确保模型从 ViT 返回 `(pixel_embeds, window_index, deepstack_pixel_embeds)`。自定义 ViT 架构可能需要适配。 |
| 应用了重新分配但没有加速 | 通信开销可能超过了均衡收益。通常在 DP 规模较小（< 8）或所有图像大小相似时发生。 |
| `--dp-balance-verbose` 频繁显示 SKIP | 大多数批次已经足够均衡。这是正常的，表明该功能正确地避免了不必要的开销。 |
