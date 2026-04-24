# Parallelism Strategies & Optimization Guide

LoongForge is built on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and is fully compatible with all existing Megatron-LM optimization strategies.  
On top of that we have added several enhancements. This document describes the basic parallelism strategies and how to enable their optimizations.  
They can be combined as needed to efficiently train **billion- to trillion-parameter** models on **hundreds to thousands of GPUs**.

## 1. Parallelism Strategies

|Strategy|Parallel Dimension|Primary Use-Case|
|--------|------------------|----------------|
|Data Parallel (DP)|Batch dimension|Standard training; enabled by default|
|Tensor Parallel (TP)|Single-layer ops / weights|Large hidden-size or memory-bound cases|
|Pipeline Parallel (PP)|Model depth|Ultra-deep models with many layers|
|Context Parallel (CP)|Sequence length|Long-sequence training (8 K+)|
|Expert Parallel (EP)|MoE experts|Mixture-of-Experts models|

---

### 1.1 Data Parallelism (DP)

* **What is parallelised**: different mini-batch samples  
* **Key idea**: every rank keeps a full copy of the model; gradients are synchronised

In DP each GPU processes a **subset of the batch**. Depending on the configuration, model-related states can be **fully replicated** or **sharded across the DP dimension** to save memory.

#### Standard DP (no sharding)

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard
```

* Each GPU stores **full parameters, gradients and optimizer states**  
* Each GPU processes only part of the batch  
* Gradients are synchronised with All-Reduce

#### Sharded Data Parallel

Use `--data-parallel-sharding-strategy` to shard some states along the DP dimension and reduce per-GPU memory.

```bash
--data-parallel-sharding-strategy {no_shard | optim | optim_grads | optim_grads_params}
```

|Strategy|What remains on each DP rank|
|--------|----------------------------|
|`no_shard` (default)|params + grads + opt state|
|`optim`|params + grads + **sharded** opt state|
|`optim_grads`|params + **sharded** grads + **sharded** opt state|
|`optim_grads_params`|**sharded** params + **sharded** grads + **sharded** opt state|

---

### 1.2 Tensor Parallelism (TP)

* **What is parallelised**: matrix computations inside one layer  
* **Key idea**: split large matrices along a dimension across GPUs

```bash
--tensor-model-parallel-size 4   # 4-way TP
--sequence-parallel              # recommended
```

`--sequence-parallel` shards the sequence dimension in LayerNorm and Dropout to reduce activation memory; usually used together with TP.

---

### 1.3 Pipeline Parallelism (PP)

* **What is parallelised**: model depth (layer dimension)  
* **Key idea**: different GPUs own different stages; execute with micro-batch pipeline

```bash
--pipeline-model-parallel-size 8              # 8 stages
--num-layers-per-virtual-pipeline-stage 4     # virtual stages for load balance
```

---

### 1.4 Context Parallelism (CP)

* **What is parallelised**: sequence length (token dimension)  
* **Key idea**: split a long sequence across GPUs

```bash
--context-parallel-size 2   # 2-way CP
--cp-comm-type p2p          # communication flavour
```

---

### 1.5 Expert Parallelism (EP)

* **What is parallelised**: experts in an MoE layer  
* **Key idea**: different GPUs hold different experts; tokens are dispatched with All-to-All

```bash
--expert-model-parallel-size 8   # 8-way EP
```

---

## 2. Performance Optimisations

### 2.1 Communication Optimisations

1. **Gradient-Reduction Overlap**
   ```bash
   --overlap-grad-reduce
   ```
   Overlaps All-Reduce of gradients with backward compute.

2. **Parameter-Gather Overlap**
   ```bash
   --overlap-param-gather
   ```
   Overlaps All-Gather of parameters with forward compute.

3. **TP Communication Overlap**
   ```bash
   --tp-comm-overlap
   ```
   Overlaps Tensor-Parallel communication with compute.

4. **EP Communication Overlap**
   ```bash
   --overlap-moe-expert-parallel-comm
   ```
   Overlaps MoE All-to-All with compute.

5. **DeepEP optimisation**  
   [DeepEP](https://github.com/deepseek-ai/DeepEP) is a high-performance MoE token dispatch/combine library from DeepSeek. It greatly reduces scheduling and sync overhead for **cross-node All-to-All**, and is the recommended setup for DeepSeek-V3 and similar models.

   ```bash
   --moe-token-dispatcher-type=flex   # {allgather | alltoall | flex}
   --moe-enable-deepep
   --moe-deepep-num-sms N             # number of SMs DeepEP may use
   ```

   DeepEP only works with the `flex` dispatcher.  
   * `allgather` (default): collect tokens with All-Gather  
   * `alltoall`: exchange tokens directly between experts  
   * `flex`: allows high-performance back-ends such as **DeepEP**

---

### 2.2 Pipeline Load Balancing

**Pipeline-load-balancing** is an advanced partitioning mechanism for **PP / VPP** that lets users specify exactly how each layer is mapped to pipeline stages via an explicit layout string.  
It solves:

* Uneven model structure (e.g. decoder layers not divisible, MTP/loss layers)  
* Load imbalance among stages with default cutting  
* Large pipeline bubbles and low GPU utilisation

Use `--pipeline-model-parallel-layout` to assign layer types and counts per stage.

```bash
--pipeline-model-parallel-size 16
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

* Layout is split by `|` per stage  
* `*N` repeats a block N times  
* Supported symbols  
  * `E` : Embedding  
  * `t` : Transformer decoder  
  * `m` : MTP layer  
  * `L` : Loss computation

---

### 2.3 Operator Fusion

#### MoE permute fusion
```bash
--moe-permute-fusion
```
Fuses token-reordering kernels to reduce memory traffic.

---

## 3. Memory Optimisations

### 3.1 Re-computation (Activation Checkpointing)

Trade extra backward compute for lower activation memory when GPU memory is tight.

Controlled by **three orthogonal knobs**:

|Dimension|Flag|Choices|
|---------|----|-------|
|Method|`--recompute-method`|`uniform` / `block`|
|Granularity|`--recompute-granularity`|`full` / `selective`|
|Layers|`--recompute-num-layers`|positive integer|

#### Method
```bash
--recompute-method uniform   # split model into equal checkpoint units
--recompute-method block     # only recompute selected Transformer layers
```

#### Granularity
```bash
--recompute-granularity full       # checkpoint whole Transformer layer
--recompute-granularity selective  # checkpoint only listed sub-modules
```

#### Number of layers
```bash
--recompute-num-layers N
```
* `uniform`: layers per checkpoint unit  
* `block`: number of layers to checkpoint on each rank / PP stage

#### Selective sub-modules (only with `selective`)
```bash
--recompute-modules core_attn moe_act mlp
```
Supported modules:  
`core_attn`, `mlp`, `moe`, `moe_act`, `shared_experts`, `routed_experts`,  
`layernorm`, `mla_up_proj`,  
`a2a_overlap_attn`, `a2a_overlap_post_attn`, `a2a_overlap_mlp` (last three require EP A2A overlap)

---

### 3.2 Activation Offloading

Offloads selected activation tensors to CPU during forward and brings them back on demand during backward to reduce peak GPU memory.

Four controlling flags:

|Dimension|Flag|Choices|
|---------|----|-------|
|Enable|`--fine-grained-activation-offloading`|on / off|
|Modules|`--offload-modules`|module list|
|Tensors|`--offload-tensors`|tensor tag list|
|Min size|`--min-offloaded-tensor-size`|bytes (int)|

```bash
--fine-grained-activation-offloading
--offload-modules expert_fc1 core_attn
--offload-tensors dispatched_input
--min-offloaded-tensor-size 1048576
```

Supported modules:  
`attn_norm`, `core_attn`, `attn_proj`, `mlp_norm`, `expert_fc1`, `moe_act`

Supported tensor tags:
- `dispatched_input` (MoE token-dispatch output)
- `pre_mlp_layernorm_output` (pre-MLP LayerNorm output)

---

### 3.3 Optimiser-State CPU Offload

Moves optimiser states (e.g. Adam momentum/variance) from GPU to CPU memory, greatly reducing GPU memory at the cost of extra CPU↔GPU traffic.  
Can be combined with recompute, activation offload and communication overlap.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0   # (0, 1]; 1.0 = offload everything
```

* `fraction < 1.0` allows a trade-off between memory saving and overhead

### 3.4 Fused Linear Cross Entropy

Fuses the output-layer linear projection (`hidden @ weight.T`) with the cross-entropy loss into a single operation, combined with chunked computation along the vocabulary dimension, to eliminate the peak memory spike from the full logits tensor. For a typical configuration (num_tokens=16384, vocab_size=129280), this can save up to **~40 GB** of logits-related memory.

The framework automatically selects the implementation based on GPU architecture:
* **GPUs except for Blackwell**: pure PyTorch implementation with buffer reuse and online Softmax — significantly reduces peak memory while outperforming the native Torch implementation

```bash
--cross-entropy-loss-fusion \
--cross-entropy-fusion-impl linear
```

The vocabulary chunk size can be tuned via environment variable (default 3072):

```bash
export LCE_GENERIC_FWD_VOCAB_SPLIT_SIZE=3072
export LCE_GENERIC_BWD_VOCAB_SPLIT_SIZE=3072
```

See [Fused Linear Cross Entropy](../features/fused_linear_cross_entropy.md) for details.