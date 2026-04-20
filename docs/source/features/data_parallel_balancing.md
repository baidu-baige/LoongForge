# DP Load-Balancing
`dp_parallel_balance` is a **low-intrusion, easy-to-adapt, and explainable** engineering solution that targets DP-load-imbalance problem arising from **fixed-length packing + quadratic Attention complexity**.
It significantly improves GPU utilisation and linear-scaling efficiency in **large-scale DP training**.

---

## 1. Background & Problem
In large-model **Data-Parallel (DP)** training we usually apply **fixed-length packing**, concatenating several samples into a constant token length (e.g. 32 K / 64 K / 128 K):

∑<sub>i</sub> len(sample<sub>i</sub>) = L

This guarantees that every DP rank sees almost same **O(n)** compute and memory footprint for embedding, MLP, linear layers, etc.

However, **Attention is O(n²)**. Its cost depends not only on total length of a pack, but also on **length distribution inside the pack**.

Taking \`flash_attn_varlen\` as an example:

* load(sample<sub>i</sub>) ∝ len(sample<sub>i</sub>)²
* load(pack) ∝ ∑<sub>i</sub> len(sample<sub>i</sub>)²

Hence, even when two DP ranks own packs with identical total lengths, their Attention workloads can differ dramatically.

During training, lightly-loaded ranks must wait for heavily-loaded ones at **All-Reduce** barrier, creating **stragglers** that lower GPU utilisation and degrade global throughput.
The issue becomes pronounced when **DP size ≥ 32**.

---

## 2. Solution Overview
The key idea is to **reorder samples across DP ranks** according to their **compute load** before forward pass, so that every rank ends up with a similar workload.
Expected benefits:

* Shorter gradient-sync waiting time
* Mitigated straggler effect
* Higher training throughput and better linear scaling

\`dp_parallel_balance\` achieves this by **data reordering** only:

* Decoupled from model architecture
* Preserves per-iteration randomness → **no convergence impact**
* Main logic runs on **CPU** → **no extra GPU kernels**

---

## 3. Usage
Add the flag in your training launcher:

\`\`\`bash
--use-vlm-dp-balance
\`\`\`

### Full CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| \`--use-vlm-dp-balance\` | bool | \`False\` | Enable VLM-level DP load balancing |
| \`--use-vit-dp-balance\` | bool | \`False\` | Enable ViT encoder DP load balancing (see [ViT DP Balance](vit_data_parallel_balancing.md)) |
| \`--vlm-dp-balance-warmup-iters\` | list[int] | \`[2,3,...,11]\` | Iteration indices used for warm-up profiling; first iteration is skipped as a cold-start exclusion |
| \`--dp-balance-max-len-ratio-vlm\` | float | \`1.2\` | Maximum sequence length ratio for VLM-level balancing. Limits each DP rank's max sequence length to (avg_len × ratio). Set to \`None\` to disable. |
| \`--dp-balance-trigger-threshold-vlm\` | float | \`0.2\` | Minimum imbalance ratio threshold for triggering VLM-level balancing. Skips when imbalance ratio < threshold. |
| \`--dp-balance-verbose\` | bool | \`False\` | Print per-iteration diagnostics: imbalance ratio, per-DP load, reorder decisions |

### Supported Models

The feature supports the following model families:

* **InternVL** — depacks/repacks \`pixel_values\`, \`image_flags\`, \`input_ids\`, \`labels\`, \`loss_weights\`
* **VLM families** (Qwen2-VL, Qwen3-VL, etc.) — depacks/repacks \`tokens\`, \`labels\`, \`attn_mask\`, \`pixel_values_images\`, \`pixel_values_videos\`, along with \`image_grid_thw\` / \`video_grid_thw\`

Model family is auto-detected from \`--model-name\` / \`--model-family\`. Both InternVL and generic VLM paths share the same solver but differ in data pack/depack logic.

---

## 4. Core Design

### 4.1 Architecture Overview

The system is integrated into the training pipeline via **monkey-patching** — no modifications to DataLoader or model code are needed:

1. **\`patches.py\`** — On startup (\`exec_adaptation()\`), when \`--use-vlm-dp-balance\` is enabled:
   - Wraps PyTorch's \`_pin_memory_loop\` with a \`_ResortQueueProxy\` that intercepts pinned batches
   - Replaces Megatron's \`RerunDataIterator\` with an extended version supporting \`__iter__\`
2. **\`pin_memory_hook.py\`** — The \`_ResortQueueProxy\` calls \`reorder_data_across_dp()\` in pin-memory thread after warm-up completes, performing cross-DP data reordering transparently
3. **\`train_hooks.py\`** — Decorators on \`train_step\` and \`training_log\` handle warm-up profiling and coefficient broadcasting

\`\`\`
DataLoader → pin_memory_loop → _ResortQueueProxy → reorder_data_across_dp()
                                                          │
                                    ┌─────────────────────┼─────────────────────┐
                                    ▼                     ▼                     ▼
                             depack batch         solve_reorder_plan      repack batch
                           (per-sample split)    (LPT + refinement)    (redistributed)
\`\`\`

---

### 4.2 Warm-up: Build a Load Model
During the first few iterations (controlled by \`--vlm-dp-balance-warmup-iters\`, default iterations 2–11), we **profile** each DP rank's sample-length distribution and iteration time, then fit the following per-rank model:

calc_load<sub>dp</sub> = a·∑<sub>i</sub> len(sample<sub>i</sub>)²
        + b·∑<sub>i</sub> len(sample<sub>i</sub>)
        + c·sample_num

* 1<sup>st</sup> term — quadratic Attention cost
* 2<sup>nd</sup> term — linear layers / comms cost
* 3<sup>rd</sup> term — fixed kernel-launch overhead

Coefficients **a, b, c** are estimated automatically by minimizing squared error between predicted max-DP load and measured forward latency, using \`scipy.optimize.minimize\` with non-negativity bounds. A smooth \`softmax_max\` approximation is used to model the synchronization bottleneck (the slowest DP rank determines iteration time).

**Warm-up flow:**

1. **\`train_step_decorator\`** — Before each train step during warm-up, peeks at all micro-batches and records per-DP statistics: \`(∑seq_len², ∑seq_len, seq_num)\` via \`set_warmup_groups()\`
2. **\`train_log_decorator\`** — After each warm-up step, records the forward computation latency via \`set_warmup_c1()\`
3. At the iteration immediately after warm-up ends, DP rank 0 fits the cost model via \`solve_computation_coef()\`, then **broadcasts** the coefficients to all DP ranks

> **Note:** The first warm-up iteration is skipped (\`iteration == vlm_dp_balance_warmup_iters[0]\`) to exclude cold-start effects.

---

### 4.3 Runtime: Load-Aware Reordering
After warm-up, every batch is processed through the following pipeline:

1. **Depack** — Split the packed batch into individual samples (\`depack_data_for_intern_vl\` or \`depack_data_for_vlm\`)
2. **Gather** — All-gather per-sample sequence lengths across all DP ranks (\`gather_sample_info_across_dp\`)
3. **Estimate** — Compute per-sample cost using the fitted model: \`cost = a·len² + b·len + c\`
4. **Solve** — Run the LPT solver with iterative refinement (\`solve_sample_dp_reorder_plan\`)
5. **Redistribute** — Execute \`all_to_all_single\` to move tensors to assigned DP ranks
6. **Repack** — Reassemble the redistributed samples into a packed batch

---

### 4.4 LPT Solver Algorithm

The solver (\`solve_sample_dp_reorder_plan\`) uses a **Greedy LPT (Longest Processing Time first)** algorithm with iterative Move/Swap refinement:

**Phase 1 — Greedy Assignment:**
1. Sort all samples globally by cost in descending order
2. For each sample, assign it to the DP rank with the currently lowest total cost
3. In VLM mode, respect the **pack length constraint**: each rank's total sequence length must not exceed \`pack_len_ratio × avg_pack_len\` (default ratio: **1.2**, controlled by \`--dp-balance-max-len-ratio-vlm\`)

**Phase 2 — Iterative Refinement (up to 20 iterations):**
1. Find the max-loaded and min-loaded DP ranks
2. If the gap is below \`swap_tolerance\` (default **5%**), stop
3. Try two operations:
   - **Move**: Transfer the highest-cost sample from max-rank to min-rank
   - **Swap**: Exchange the highest-cost sample on max-rank with the lowest-cost on min-rank
4. Select whichever operation reduces the max–min gap more (respecting pack length constraints)

**Skip conditions** — Rebalancing is skipped (returning \`None\`) when:
- **Imbalance ratio < threshold**: current distribution is already well-balanced (default threshold: **0.2**, controlled by \`--dp-balance-trigger-threshold-vlm\`)
- **Single sample dominates**: the highest single-sample cost exceeds the average DP load, meaning no redistribution can meaningfully improve balance

---

### 4.5 Cross Micro-Batch Balancing

When \`num_microbatches > 1\`, a \`_MicroBatchLoadTracker\` accumulates per-DP costs across micro-batches within the same iteration. The solver receives \`dp_historical_costs\` from previous micro-batches so that greedy assignment and refinement consider the **total iteration load**, not just the current micro-batch. This prevents scenarios where each micro-batch is individually balanced but the total per-rank load across all micro-batches is skewed.

---

### 4.6 Tensor Redistribution

The \`redistribute_tensor_helper\` function performs the actual cross-DP communication:

1. Build send/recv metadata from the reorder plan
2. Flatten all tensors to be sent into a contiguous buffer
3. Execute \`all_to_all_single\` to exchange tensors across DP ranks
4. Split the received buffer and apply \`reconstruct_func\` to restore per-sample tensor shapes

Each tensor type (LLM tokens, labels, pixel values, etc.) is redistributed independently with its own \`reconstruct_func\`, because different tensors may have different element counts per sample.

---

## 5. Diagnostic Output

When \`--dp-balance-verbose\` is enabled, the system prints per-iteration diagnostics on DP rank 0, TP rank 0:

**When rebalancing is skipped:**
\`\`\`
[DP Balance][VLM] SKIP | reason: imbalance 0.1234 < 0.2
  imbalance : 0.1234
  load/dp   : [1200.0, 1100.0, 1150.0, 1180.0]
  cumulative: ViT_rebalance: 0/5 applied, VLM_rebalance: 3/10 applied
\`\`\`

**When rebalancing is applied:**
\`\`\`
[DP Balance][VLM] APPLY
  before    : imbalance=0.3456  load/dp=[1800.0, 1100.0, 1200.0, 1400.0]
  after     : imbalance=0.0234  load/dp=[1375.0, 1380.0, 1370.0, 1375.0]
  cumulative: ViT_rebalance: 2/5 applied, VLM_rebalance: 8/10 applied
\`\`\`

The cumulative counters track how often each type of rebalancing was applied vs. skipped across the entire training run.

---

## 6. Experimental Results
Fixed **tensor-parallel = 4**, InternVL on *** dataset.
Average tokens / GPU / sec (TGS) vs. DP size:

![Average TGS vs. DP size with and without data reorganization](../../assets/images/dp_balancing.png)

* **Small DP (4 / 8 / 16)**
  – With or without reordering: almost identical TGS → imbalance is negligible.

* **Large DP (≥ 32)**
  – Without reordering: TGS drops quickly because of stragglers.
  – With \`dp_parallel_balance\`:
    – Attention load balanced across ranks
    – All-Reduce wait time reduced sharply
    – Throughput degradation largely suppressed; benefit grows with DP scale
