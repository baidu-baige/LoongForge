# Adaptive FP8 Training (Selective FP8)

Adaptive FP8 (also called **Selective FP8**) is a **benchmark-driven dynamic precision selection** mechanism in LoongForge. At model initialization time, it consults a pre-generated performance policy file (FP8 Dynamic Policy) to decide — on a per-module basis — whether to use FP8 or BF16. This retains FP8 speedups where they exist while avoiding regressions in unfavorable scenarios (e.g. small MoE experts, high TP parallelism, short sequences).

---

## 0. Background & Motivation

Full FP8 training ([FP8 Training](fp8_training.md)) delivers significant speedups for Dense models with large hidden sizes and long sequences. However, not every layer or configuration benefits from FP8:

- **MoE Grouped GEMM**: When expert sizes are small, FP8 quantisation overhead can exceed the compute benefit.
- **High TP parallelism**: After tensor-parallel splitting each tile is smaller, weakening FP8 advantage.
- **Short sequences / small batches**: Insufficient token counts prevent FP8 kernels from saturating the hardware.

Adaptive FP8 addresses this by **enabling FP8 only on layers where benchmark data confirms a speedup**, keeping the rest in BF16. The goal is "never slower than BF16, as close to full FP8 as possible".

---

## 1. Prerequisites

| Item | Requirement |
|------|-------------|
| **Hardware** | Native FP8 support on the target FP8 hardware platform |
| **Software** | Transformer Engine enabled in the framework |
| **Baseline** | Full FP8 training verified to work correctly (see [FP8 Training](fp8_training.md)) |

---

## 2. Workflow

Adaptive FP8 usage involves two stages: **Generate Policy** and **Enable in Training**.

### 2.1 Stage 1 — Benchmark to Generate FP8 Policy

Use `tools/benchmark_te_parallel_layers.py` to benchmark TE parallel layers for the target model under different TP/EP configurations and produce a policy file.

#### 2.1.1 Dense Models

```bash
# Step 1: Run benchmarks at each target TP size
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

# Step 2: Merge multi-TP reports into a unified policy
python tools/benchmark_te_parallel_layers.py merge-policy \
    --reports outputs/report_tp1.json outputs/report_tp2.json outputs/report_tp4.json \
    --output configs/models/qwen2.5/fp8_policy_qwen2_5_72b.json \
    --speedup-threshold 1.0
```

#### 2.1.2 MoE Models

MoE models require additional EP coverage:

```bash
# TP=1, EP=4 (requires 4 GPUs)
TE_LAYER_PERF_OMNI_CONFIG_PATH="configs/models/deepseek3/deepseek_v3.yaml" \
TE_LAYER_PERF_TP_SIZE=1 \
TE_LAYER_PERF_EP_SIZE=4 \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
TE_LAYER_PERF_FP8_RECIPE="blockwise" \
TE_LAYER_PERF_REPORT_PATH="outputs/report_tp1_ep4.json" \
TE_LAYER_PERF_WARMUP=5 \
TE_LAYER_PERF_ITERS=5 \
    torchrun --nproc_per_node 4 tools/benchmark_te_parallel_layers.py

# TP=2, EP=4 (requires 8 GPUs, world_size = tp * ep)
TE_LAYER_PERF_TP_SIZE=2 TE_LAYER_PERF_EP_SIZE=4 \
TE_LAYER_PERF_REPORT_PATH="outputs/report_tp2_ep4.json" \
    torchrun --nproc_per_node 8 tools/benchmark_te_parallel_layers.py

# Merge
python tools/benchmark_te_parallel_layers.py merge-policy \
    --reports outputs/report_tp1_ep4.json outputs/report_tp2_ep4.json \
    --output configs/models/deepseek3/fp8_policy_deepseek_v3.json \
    --speedup-threshold 1.0
```

#### 2.1.3 VLM (Vision-Language Models)

The benchmark tool automatically extracts both ViT and LLM components from VLM configs:

```bash
TE_LAYER_PERF_OMNI_CONFIG_PATH="configs/models/qwen3_vl/qwen3_vl_235b_a22b.yaml" \
TE_LAYER_PERF_TP_SIZE=1 \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
TE_LAYER_PERF_FP8_RECIPE="blockwise" \
TE_LAYER_PERF_REPORT_PATH="outputs/report_qwen3_vl_tp1.json" \
    torchrun --nproc_per_node 1 tools/benchmark_te_parallel_layers.py
```

#### 2.1.4 Benchmark Environment Variable Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `TE_LAYER_PERF_OMNI_CONFIG_PATH` | Model YAML config path | — |
| `TE_LAYER_PERF_TP_SIZE` | Tensor parallel size | Model default |
| `TE_LAYER_PERF_EP_SIZE` | Expert parallel size | Model default |
| `TE_LAYER_PERF_ETP_SIZE` | Expert-tensor parallel size | Model default |
| `TE_LAYER_PERF_PRECISIONS` | Precisions to benchmark | `"bf16,fp8"` |
| `TE_LAYER_PERF_FP8_RECIPE` | FP8 recipe | `blockwise` |
| `TE_LAYER_PERF_WARMUP` | Warmup iterations | `10` |
| `TE_LAYER_PERF_ITERS` | Timed iterations | `10` |
| `TE_LAYER_PERF_REPORT_PATH` | Report output path | — |
| `TE_LAYER_PERF_FP8_POLICY_PATH` | Direct policy export path | — |
| `TE_LAYER_PERF_SPEEDUP_THRESHOLD` | Minimum speedup to consider FP8 beneficial | `1.0` |
| `TE_LAYER_PERF_SHAPE_SWEEP` | Custom shape sweep (e.g. `"1024x1,4096x2"`) | Auto-generated |

---

### 2.2 Policy File Format

The generated policy JSON has the following structure:

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

Dense module kinds (`layernorm_column` / `column` / `row` / `duplicated`) use a nested `{ub_name: [rules]}` layout so that same-kind modules with different shapes (e.g. `qkv` vs `fc1`, or `proj` vs `fc2`) can carry distinct thresholds. MoE grouped kinds keep the flat list form — per-expert `ub_name` is not a meaningful shape discriminator.

**Decision logic**:

- **Dense layers** (`layernorm_column` / `column` / `row` / `duplicated`): Lookup by `(module_kind, ub_name, tp)` → enable FP8 when `seq_length × micro_batch_size >= min_tokens`. `ub_name` is the TE tp-comm buffer name of the module (`qkv` / `proj` / `fc1` / `fc2` / `q_down_proj` / `kv_down_proj`).
- **MoE layers** (`column_grouped` / `row_grouped`): Lookup by `(module_kind, etp, num_gemms)` → token count is `seq_length × micro_batch_size × moe_router_topk`.
- **Missing rules**: If the policy has no matching entry for the current `(ub_name, tp)` or `(etp, num_gemms)`, **conservatively fall back to BF16**.

---

### 2.3 Stage 2 — Enable Adaptive FP8 in Training

#### 2.3.1 Standalone LLM (YAML config)

Add adaptive FP8 parameters in the model YAML:

```yaml
# Example: configs/models/deepseek3/deepseek_v3_fp8_sel.yaml
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

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `fp8: "e4m3"` | FP8 format (E4M3) |
| `fp8_recipe: "blockwise"` | Block-wise quantisation recipe |
| `fp8_param: True` | Store weights in FP8 (optional, saves memory) |
| `selective_fp8: true` | **Enable adaptive FP8** |
| `fp8_dynamic_policy_path` | Path to the policy JSON (relative to project root or absolute) |

#### 2.3.2 VLM (Vision-Language Models)

VLM models can configure Foundation (LLM) and Image Encoder (ViT) independently:

```yaml
# Example: configs/models/qwen3_vl/qwen3_vl_235b_a22b_fp8_sel.yaml
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

> **Tip**: In multimodal models the ViT and LLM process different effective token counts. Set `fp8_dynamic_num_tokens` explicitly per component to avoid inaccurate auto-inference from the global `seq_length`.

#### 2.3.3 Launch Script

Use the same CLI flags and epsilon guards as full FP8 training:

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
    ...  # other training arguments
```

---

## 3. How It Works

### 3.1 Architecture Overview

```
Startup → parse_args_from_config
            │
            └─ _register_selective_fp8_decision()
                 │
                 └─ Registers selective_fp8_init_decision callback into Megatron

Model build → For each TE module at init time
            │
            └─ selective_fp8_init_decision(config, te_cls, ub_name, init_kwargs)
                 │
                 ├─ Identify module_kind (layernorm_column / row / column_grouped / …)
                 ├─ Compute effective token count
                 ├─ Query policy: should_use_fp8(module_kind, num_tokens, tp, etp)
                 │
                 ├─ True  → Module keeps FP8
                 └─ False → Module marked _selective_fp8_disabled, runs in BF16
```

### 3.2 Supported Module Types

| TE Module Class | module_kind | Typical Usage |
|----------------|-------------|---------------|
| `TELayerNormColumnParallelLinear` | `layernorm_column` | QKV / FC1 (with fused LayerNorm) |
| `TEColumnParallelLinear` | `column` | Column-parallel linear |
| `TERowParallelLinear` | `row` | Proj / FC2 row-parallel |
| `TEColumnParallelGroupedLinear` | `column_grouped` | MoE expert FC1 |
| `TERowParallelGroupedLinear` | `row_grouped` | MoE expert FC2 |
| `TELinear` | `duplicated` | MLA down-projection |

### 3.3 MoE Expert Handling

For expert layers in MoE models:
- `column` / `layernorm_column` is automatically promoted to `column_grouped`.
- `row` is promoted to `row_grouped`.
- The effective token count is multiplied by `moe_router_topk` to reflect actual compute.
- Policy lookup uses `(etp, num_gemms)` as the key instead of `tp`.

---

## 4. Expected Behaviour by Scenario

| Scenario | Full FP8 | Adaptive FP8 |
|----------|----------|-------------|
| Dense model with large hidden size (>=8192) | Significant speedup | ≈ Full FP8 (all layers enabled by policy) |
| Dense model with short sequences (<=2048) | May regress | >= BF16 (adaptively skipped) |
| MoE with small experts | Often regresses | >= BF16 (expert layers kept in BF16) |
| MoE with high TP | Notable regression | Substantially better than full FP8 |
| VLM (mixed ViT + LLM) | Varies per component | Each component optimised independently |

---

## 5. Troubleshooting

| Symptom | Likely Fix |
|---------|------------|
| NaN/Inf in loss or gradients | Check that `FP8_QUANT_*_AMAX_EPS` variables are set (recommended `1e-12`). |
| FP8_SEL throughput lower than expected | Verify policy `min_tokens` thresholds match your actual `seq_length × micro_batch_size`. |
| ViT FP8 decisions seem wrong in VLM | Set `fp8_dynamic_num_tokens` explicitly for the ViT component. |
| Missing rules in policy for current TP/EP | Re-run benchmark at the needed TP/EP and merge into the policy. |
| FP8_SEL performance identical to full FP8 | Normal — all layers benefit from FP8, so the policy enables every module. |
| Benchmark tool OOM | Reduce the max token count in `TE_LAYER_PERF_SHAPE_SWEEP`. |

---

## 6. Comparison with Full FP8

| Aspect | Full FP8 | Adaptive FP8 |
|--------|----------|-------------|
| Configuration complexity | Low (global switch) | Medium (requires policy generation) |
| Dense large-model speedup | Optimal | ≈ Full FP8 |
| MoE safety | Risk of regression | Protected — never slower than BF16 |
| Runtime overhead | None | None (decision at init; forward path is a single `getattr` check) |
| Best for | Models verified to benefit uniformly from FP8 | New models, mixed architectures, MoE, VLM |

**Recommendation**: For new models or mixed architectures (MoE, VLM), prefer Adaptive FP8. For Dense models where full FP8 has been thoroughly validated, full FP8 is simpler.
