# flash_mla_sparse_bwd 算子使用说明

## 概述

`flash_mla_sparse_bwd` 是稀疏注意力机制的反向传播算子，用于 MLA (Multi-Head Latent Attention) 架构中的预填充阶段反向计算。该算子支持 TopK 稀疏注意力模式，可显著降低计算复杂度。

## 函数签名

```python
def flash_mla_sparse_bwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    dO: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: Optional[float] = None,
    d_v: int = 512,
    topk_length: Optional[torch.Tensor] = None,
    q_start_index_s: int = 0,
    fast_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

## 参数说明

| 参数 | 类型 | 形状 | 数据类型 | 说明 |
|------|------|------|----------|------|
| `q` | Tensor | `[s_q, h_q, d_qk]` | bfloat16 | Query 张量 |
| `kv` | Tensor | `[s_kv, h_kv, d_qk]` | bfloat16 | Key/Value 张量 |
| `o` | Tensor | `[s_q, h_q, d_v]` | bfloat16 | 前向传播输出 |
| `dO` | Tensor | `[s_q, h_q, d_v]` | bfloat16 | 输出梯度 |
| `indices` | Tensor | `[s_q, h_kv, topk]` | int32 | TopK 索引 |
| `lse` | Tensor | `[s_q, h_q]` | float32 | Log-Sum-Exp (来自前向传播) |
| `sm_scale` | float | - | - | Softmax 缩放因子，默认为 `d_qk^(-0.5)` |
| `d_v` | int | - | - | Value 维度，必须为 512 |
| `topk_length` | Tensor | `[s_q]` | int32 | 可选的 TopK 长度 |
| `q_start_index_s` | int | - | - | 当前 chunk 在全局序列中的起始位置 (用于因果掩码) |
| `fast_mode` | bool | - | - | 是否启用双kernel融合模式，详见下文 |

## 返回值

返回元组 `(dQ, dKV)`：

| 返回值 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `dQ` | `[s_q, h_q, d_qk]` | bfloat16 | Query 梯度 |
| `dKV` | `[s_kv, h_kv, d_qk]` | bfloat16 | KV 梯度 |

## 使用示例

```python
import torch
from flash_mla_bwd import flash_mla_sparse_bwd

# 假设已有前向传播的结果
# q: [s_q, h_q, d_qk], kv: [s_kv, h_kv, d_qk]
# o: 前向输出, indices: TopK索引, lse: Log-Sum-Exp

dO = torch.randn_like(o)  # 输出梯度

# 单kernel模式 (默认)
dQ, dKV = flash_mla_sparse_bwd(
    q=q,
    kv=kv,
    o=o,
    dO=dO,
    indices=indices,
    lse=lse,
    d_v=512,
    fast_mode=False
)

# 双kernel融合模式 (更快但额外占用显存)
dQ, dKV = flash_mla_sparse_bwd(
    q=q,
    kv=kv,
    o=o,
    dO=dO,
    indices=indices,
    lse=lse,
    d_v=512,
    fast_mode=True  # 仅支持 h_q=128
)
```

---

## fast_mode 参数详解

### 概述

`fast_mode` 参数控制反向传播的 kernel 执行策略，影响计算性能和显存占用。

### 模式对比

| 模式 | fast_mode 设置 | h_q 要求 | kernel 数量 | 额外显存 | 执行速度 (SM100) |
|------|---------------|----------|-------------|----------|------------------|
| 单kernel模式 | `False` | 128 或 64 | 1 | 无 | ~400 TFLOPS |
| 双kernel融合模式 | `True` | 仅支持 128 | 2 | 有 | ~550 TFLOPS |

### 执行路径详解

#### 单kernel模式 (`fast_mode=False`)

当 `fast_mode=False` 时，使用单个 kernel 完成整个反向传播计算：

```
输入 (q, kv, o, dO, indices, lse)
         │
         ▼
    ┌─────────────────┐
    │  单 Kernel 计算  │
    │  (phase1)       │
    └─────────────────┘
         │
         ▼
输出 (dQ, dKV)
```

**特点**：
- 无额外显存占用
- 支持 `h_q=128` 和 `h_q=64` 两种头数
- 适合显存受限场景

#### 双kernel融合模式 (`fast_mode=True`)

当 `fast_mode=True` 时，反向传播拆分为两个独立的 kernel：

```
输入 (q, kv, o, dO, indices, lse)
         │
         ▼
    ┌─────────────────┐
    │  Kernel 1       │ ──► 存储 s, ds 到显存
    │  (dq_phase)     │      计算 dQ
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │  Kernel 2       │ ◄── 读取 s, ds
    │  (dkv_phase)    │      计算 dKV
    └─────────────────┘
         │
         ▼
输出 (dQ, dKV)
```

**特点**：
- 执行速度更快
- **仅支持 `h_q=128`**
- 需要额外显存存储中间结果

### 中间张量说明

双kernel模式需要存储两个中间张量：

| 张量 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `s` | `[s_q, h_q, topk]` | bfloat16 | Softmax 概率值 |
| `ds` | `[s_q, h_q, topk]` | bfloat16 | Softmax 梯度值 |

- **s (softmax probabilities)**: 在 dq_phase kernel 中计算并存储，供 dkv_phase kernel 读取
- **ds (softmax gradients)**: 在 dq_phase kernel 中计算并存储，供 dkv_phase kernel 读取

### 额外显存计算公式

双kernel模式的额外显存占用计算公式：

```
额外显存 = size(s) + size(ds)
         = s_q × h_q × topk × sizeof(bfloat16) × 2
         = s_q × h_q × topk × 2 bytes × 2
         = 4 × s_q × h_q × topk bytes
```

#### 参数说明

| 参数 | 符号 | 说明 |
|------|------|------|
| `s_q` | Query序列长度 | Query 张量的第一维大小 |
| `h_q` | Query头数 | Query 张量的第二维大小 (必须为 128) |
| `topk` | TopK值 | 稀疏注意力的 TopK 数量 |

#### 计算示例

**示例 1: 典型配置**
- `s_q = 4096` (Query序列长度)
- `h_q = 128` (Query头数)
- `topk = 2048` (TopK值)

```
额外显存 = 4 × 4096 × 128 × 2048 bytes
         = 4,294,967,296 bytes
         = 4 GB
```

**示例 2: 较小配置**
- `s_q = 1024`
- `h_q = 128`
- `topk = 512`

```
额外显存 = 4 × 1024 × 128 × 512 bytes
         = 268,435,456 bytes
         = 256 MB
```

**示例 3: 大规模配置**
- `s_q = 8192`
- `h_q = 128`
- `topk = 4096`

```
额外显存 = 4 × 8192 × 128 × 4096 bytes
         = 17,179,869,184 bytes
         = 16 GB
```

### 快速估算表

| s_q | h_q | topk | 额外显存 |
|-----|-----|------|----------|
| 512 | 128 | 256 | 64 MB |
| 1024 | 128 | 512 | 256 MB |
| 2048 | 128 | 1024 | 1 GB |
| 4096 | 128 | 2048 | 4 GB |
| 8192 | 128 | 4096 | 16 GB |

### 选择建议

| 场景 | 推荐设置 |
|------|----------|
| 显存充足，追求性能 | `fast_mode=True` |
| 显存受限 | `fast_mode=False` |
| `h_q=64` | 必须使用 `fast_mode=False` |
| 大序列长度 + 大 TopK | 谨慎使用 `fast_mode=True`，注意显存占用 |

---

## 注意事项

1. **头数限制**: 算子当前仅支持 `h_q=128` 或 `h_q=64`
2. **Value维度**: `d_v` 参数必须为 512
3. **数据类型**: 输入张量需为 bfloat16，lse 需为 float32
4. **fast_mode限制**: `fast_mode=True` 仅支持 `h_q=128`

## 相关文件

- 接口实现: `flash_mla_bwd/flash_mla_interface.py`
- CUDA绑定: `src/pybind.cpp`
- 参数定义: `src/params.h`
- 双kernel实现: `src/sm100/head128_2kernels/`
