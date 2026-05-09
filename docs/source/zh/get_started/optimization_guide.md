# 并行策略及优化指南

LoongForge 基于 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 构建，完全兼容所有现有的 Megatron-LM 优化策略。
在此基础上，我们增加了多项增强。本文档介绍基本的并行策略及其优化的启用方式。
这些策略可以按需组合，以在**数百至数千 GPU**上高效训练**十亿至万亿参数**的模型。

## 1. 并行策略

|策略|并行维度|主要应用场景|
|--------|------------------|----------------|
|数据并行（DP）|批次维度|标准训练；默认启用|
|张量并行（TP）|单层算子 / 权重|大隐藏层尺寸或显存受限场景|
|流水线并行（PP）|模型深度|超深模型（层数多）|
|上下文并行（CP）|序列长度|长序列训练（8K+）|
|专家并行（EP）|MoE 专家|混合专家模型|

---

### 1.1 数据并行（DP）

* **并行对象**：不同的微批次样本
* **核心思想**：每个 rank 保留完整的模型副本；梯度同步

在 DP 中，每个 GPU 处理**批次的一个子集**。根据配置，模型相关状态可以**完全复制**或**沿 DP 维度分片**以节省显存。

#### 标准 DP（无分片）

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

* 每个 GPU 存储**完整的参数、梯度和优化器状态**
* 每个 GPU 仅处理批次的一部分
* 梯度通过 All-Reduce 同步

#### 分片数据并行

使用 `--data-parallel-sharding-strategy` 沿 DP 维度分片部分状态，降低每张 GPU 的显存占用。

```bash
--data-parallel-sharding-strategy {no_shard | optim | optim_grads | optim_grads_params}
```

|策略|每个 DP rank 上保留的内容|
|--------|----------------------------|
|`no_shard`（默认）|参数 + 梯度 + 优化器状态|
|`optim`|参数 + 梯度 + **分片的**优化器状态|
|`optim_grads`|参数 + **分片的**梯度 + **分片的**优化器状态|
|`optim_grads_params`|**分片的**参数 + **分片的**梯度 + **分片的**优化器状态|

---

### 1.2 张量并行（TP）

* **并行对象**：单层内的矩阵运算
* **核心思想**：将大矩阵沿某一维度切分到多个 GPU

```bash
--tensor-model-parallel-size 4   # 4 路张量并行
--sequence-parallel              # 推荐开启
```

`--sequence-parallel` 在 LayerNorm 和 Dropout 中分片序列维度以减少激活显存；通常与 TP 配合使用。

---

### 1.3 流水线并行（PP）

* **并行对象**：模型深度（层维度）
* **核心思想**：不同 GPU 拥有不同的阶段；以微批次流水线方式执行

```bash
--pipeline-model-parallel-size 8              # 8 个阶段
--num-layers-per-virtual-pipeline-stage 4     # 虚拟阶段用于负载均衡
```

---

### 1.4 上下文并行（CP）

* **并行对象**：序列长度（token 维度）
* **核心思想**：将长序列切分到多个 GPU

```bash
--context-parallel-size 2   # 2 路上下文并行
--cp-comm-type p2p          # 通信方式
```

---

### 1.5 专家并行（EP）

* **并行对象**：MoE 层中的专家
* **核心思想**：不同 GPU 持有不同专家；token 通过 All-to-All 分发

```bash
--expert-model-parallel-size 8   # 8 路专家并行
```

---

## 2. 性能优化

### 2.1 通信优化

1. **梯度归约重叠**
   ```bash
   --overlap-grad-reduce
   ```
   将梯度的 All-Reduce 与反向计算重叠。

2. **参数收集重叠**
   ```bash
   --overlap-param-gather
   ```
   将参数的 All-Gather 与前向计算重叠。

3. **TP 通信重叠**
   ```bash
   --tp-comm-overlap
   ```
   将张量并行通信与计算重叠。

4. **EP 通信重叠**
   ```bash
   --overlap-moe-expert-parallel-comm
   ```
   将 MoE All-to-All 与计算重叠。

5. **DeepEP 优化**
   [DeepEP](https://github.com/deepseek-ai/DeepEP) 是来自 DeepSeek 的高性能 MoE token 分发/合并库。它大幅降低**跨节点 All-to-All**的调度和同步开销，是 DeepSeek-V3 及类似模型的推荐配置。

   ```bash
   --moe-token-dispatcher-type=flex   # {allgather | alltoall | flex}
   --moe-enable-deepep
   --moe-deepep-num-sms N             # DeepEP 可使用的 SM 数量
   ```

   DeepEP 仅与 `flex` 分发器配合使用。
   * `allgather`（默认）：通过 All-Gather 收集 token
   * `alltoall`：在专家之间直接交换 token
   * `flex`：允许使用高性能后端如 **DeepEP**

---

### 2.2 流水线负载均衡

**流水线负载均衡**是一种针对 **PP / VPP** 的高级划分机制，允许用户通过显式布局字符串精确指定每层如何映射到流水线阶段。
它解决以下问题：

* 模型结构不均匀（如解码器层不可整除、MTP/loss 层）
* 默认切分导致的阶段间负载不均衡
* 流水线气泡大、GPU 利用率低

使用 `--pipeline-model-parallel-layout` 为每个阶段指定层类型和数量。

```bash
--pipeline-model-parallel-size 16
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

* 布局按 `|` 分隔每个阶段
* `*N` 表示重复一个块 N 次
* 支持的符号
  * `E`：Embedding
  * `t`：Transformer 解码器
  * `m`：MTP 层
  * `L`：Loss 计算

---

### 2.3 算子融合

#### MoE 排列融合
```bash
--moe-permute-fusion
```
融合 token 重排算子以减少显存访问。

---

## 3. 显存优化

### 3.1 重计算（Activation Checkpointing）

在 GPU 显存紧张时，用额外的反向计算换取更低的激活显存。

通过**三个正交旋钮**控制：

|维度|标志|选项|
|---------|----|-------|
|方法|`--recompute-method`|`uniform` / `block`|
|粒度|`--recompute-granularity`|`full` / `selective`|
|层数|`--recompute-num-layers`|正整数|

#### 方法
```bash
--recompute-method uniform   # 将模型均分为等大的重计算单元
--recompute-method block     # 仅重计算选定的 Transformer 层
```

#### 粒度
```bash
--recompute-granularity full       # 重计算整个 Transformer 层
--recompute-granularity selective  # 仅重计算列出的子模块
```

#### 层数
```bash
--recompute-num-layers N
```
* `uniform`：每个重计算单元的层数
* `block`：每个 rank / PP 阶段上重计算的层数

#### 选择性子模块（仅与 `selective` 配合使用）
```bash
--recompute-modules core_attn moe_act mlp
```
支持的模块：
`core_attn`、`mlp`、`moe`、`moe_act`、`shared_experts`、`routed_experts`、
`layernorm`、`mla_up_proj`、
`a2a_overlap_attn`、`a2a_overlap_post_attn`、`a2a_overlap_mlp`（后三个需要 EP A2A 重叠）

---

### 3.2 激活卸载

在前向过程中将选定的激活张量卸载到 CPU，在反向过程中按需取回，以降低峰值 GPU 显存。

四个控制标志：

|维度|标志|选项|
|---------|----|-------|
|启用|`--fine-grained-activation-offloading`|on / off|
|模块|`--offload-modules`|模块列表|
|张量|`--offload-tensors`|张量标签列表|
|最小大小|`--min-offloaded-tensor-size`|字节数（int）|

```bash
--fine-grained-activation-offloading
--offload-modules expert_fc1 core_attn
--offload-tensors dispatched_input
--min-offloaded-tensor-size 1048576
```

支持的模块：
`attn_norm`、`core_attn`、`attn_proj`、`mlp_norm`、`expert_fc1`、`moe_act`

支持的张量标签：
- `dispatched_input`（MoE token 分发输出）
- `pre_mlp_layernorm_output`（MLP 前的 LayerNorm 输出）

---

### 3.3 优化器状态 CPU 卸载

将优化器状态（如 Adam 动量/方差）从 GPU 移至 CPU 内存，大幅减少 GPU 显存，代价是额外的 CPU 与 GPU 间数据传输。
可与重计算、激活卸载和通信重叠配合使用。

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0   # (0, 1]；1.0 = 全部卸载
```

* `fraction < 1.0` 允许在显存节省与开销之间进行权衡

### 3.4 融合线性交叉熵

将输出层的线性投影（`hidden @ weight.T`）与交叉熵损失融合为单一操作，并结合沿词表维度的分块计算，消除完整 logits 张量带来的峰值显存尖峰。对于典型配置（num_tokens=16384, vocab_size=129280），可节省高达 **~40 GB** 的 logits 相关显存。

框架根据 GPU 架构自动选择实现方式：
* **非 Blackwell GPU**：带缓冲区复用和在线 Softmax 的纯 PyTorch 实现——在超过原生 Torch 实现性能的同时显著降低峰值显存

```bash
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl linear
```

词表分块大小可通过环境变量调整（默认 3072）：

```bash
export LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE=3072
export LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE=3072
```

详见[融合线性交叉熵](../features/fused_linear_cross_entropy.md)。
