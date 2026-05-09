# 融合线性交叉熵

LoongForge 为模型输出层提供了一种显存优化方案。通过将 `hidden @ weight.T` 线性投影与交叉熵损失融合为单一操作，并结合分块计算，显著降低了词表投影阶段的峰值显存占用。

在标准训练中，输出层生成形状为 `(num_tokens, vocab_size)` 的完整 Logits 张量，该张量在反向传播期间再次被保留，导致显存开销翻倍。对于典型配置（num_tokens=16384, vocab_size=129280），仅 Logits 相关的显存就可能达到 **~40 GB**。本优化通过两步递进方案解决此问题：

- **步骤 1（算子融合）**：将线性投影和交叉熵融合为单一 autograd Function，反向传播由框架控制，仅保存轻量级统计信息（每个 Token 的最大值和指数和），无需在前向和反向传播之间存储完整 Logits
- **步骤 2（分块计算）**：沿词表维度将权重切分为小块（默认 `vocab_per_split=3072`），使用在线 Softmax 算法逐块计算并立即丢弃，因此完整 Logits 张量**永远不会被实例化**

LoongForge 提供两种实现路径，框架**根据 GPU 架构自动选择**；

## 使用方法
在训练启动脚本中添加以下参数：

```bash
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl linear
```

---

## 1. 通用实现

LoongForge 使用混合精度、缓冲区复用、原地操作和 Autograd 等策略实现了纯 PyTorch 通用版本，使此优化能在**任何 CUDA GPU** 上运行，同时相比原生 Torch 实现也具有显著的性能优势。

核心设计：预分配宽度为 `vocab_per_split` 的小缓冲区，将每次矩阵乘法直接写入该缓冲区（通过 `out=` 参数），结果在同一循环迭代中立即被在线 Softmax 消费，并在下一轮被覆盖——完整 Logits 永远不会在 Python 层面累积：

```python
matmul_buf = torch.empty((num_tokens, vocab_per_split), ...)  # allocate only one chunk size

for split_idx in range(num_splits):
    torch.matmul(hidden, weight[v_start:v_end].t(), out=matmul_buf)  # write into reused buffer
    logits_chunk.sub_(new_max.unsqueeze(1)).exp_()                    # in-place, immediately consumed
    accumulate.mul_(torch.exp(maximum - new_max)).add_(chunk_sum)     # update statistics
    maximum = new_max
    # next matmul directly overwrites the buffer, complete logits never existed
```

反向传播同样逐块重计算，使用前向传播保存的 `maximum` 和 `accumulate`（形状均为 `(num_tokens,)`）恢复每个块的 Softmax 概率，无需保存完整 Logits。

### 特性
* 适用于**任何 CUDA GPU**，无硬件限制
* 在 A800 上比原生实现快 **22~26%**
* 在线 Softmax 保证与原生实现**数值完全一致**（Loss/梯度差异 < 1e-5）
* 支持 DP / TP / SP 并行策略，支持 FP8 训练


---

## 2. 调优参数

通过环境变量控制分块大小，平衡显存占用和性能：

```bash
# 默认值 3072，显存与性能的最优平衡（推荐）
export LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE=3072
export LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE=3072

# 显存充足时增大分块大小以提高 GPU 利用率
export LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE=8192

# 显存极度受限时减小分块大小
export LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE=512
```
