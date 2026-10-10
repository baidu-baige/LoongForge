# TE Parallel Layer Benchmark

Measure how much faster (or slower) FP8 is than BF16 for TransformerEngine
parallel Linear layers, at your model's real shapes and parallelism.

Two things you can do with it:

- **Build an FP8 policy for your model** — the file Adaptive FP8 training reads to
  turn FP8 on only where it actually helps. Do this once per model before training
  with `selective_fp8: true`.
- **Just profile TE layers** — get per-layer latency, throughput, and TFLOPS for
  BF16 vs FP8, with no policy step.

## Before you start

Run it inside the LoongForge training environment (needs `torch`, TransformerEngine,
and `megatron.core`), with Megatron on your path:

```bash
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
```

You point the benchmark at your model with a LoongForge model YAML **by path**, so
`loongforge` itself does not need to be importable.

## Build an FP8 policy for your model

**Step 1 — benchmark at each parallelism you'll train with.** Run once per TP (and
EP, for MoE) size; each run writes one report. The launch world size
(`--nproc_per_node` × `--nnodes`) must be divisible by `TE_LAYER_PERF_TP_SIZE` ×
`TE_LAYER_PERF_EP_SIZE`; the simplest setup is `--nproc_per_node` = `TE_LAYER_PERF_TP_SIZE`,
as below. Here we benchmark TP=1 and TP=4 so Step 2 has two reports to merge:

```bash
# TP=1 -> outputs/report_tp1.json
TE_LAYER_PERF_OMNI_CONFIG_PATH=configs/models/qwen2.5/qwen2_5_72b.yaml \
TE_LAYER_PERF_TP_SIZE=1 \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
TE_LAYER_PERF_REPORT_PATH=outputs/report_tp1.json \
    torchrun --nproc_per_node 1 tools/te_parallel_benchmark/benchmark_te_parallel_layers.py

# TP=4 -> outputs/report_tp4.json
TE_LAYER_PERF_OMNI_CONFIG_PATH=configs/models/qwen2.5/qwen2_5_72b.yaml \
TE_LAYER_PERF_TP_SIZE=4 \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
TE_LAYER_PERF_REPORT_PATH=outputs/report_tp4.json \
    torchrun --nproc_per_node 4 tools/te_parallel_benchmark/benchmark_te_parallel_layers.py
```

**Step 2 — merge the reports into one policy file:**

```bash
python tools/te_parallel_benchmark/benchmark_te_parallel_layers.py merge-policy \
    --reports outputs/report_tp1.json outputs/report_tp4.json \
    --output configs/models/qwen2.5/fp8_policy_qwen2_5_72b.json \
    --speedup-threshold 1.0
```

| Argument | Required | Description |
|----------|:---:|-------------|
| `--reports` | ✅ | The report JSONs from step 1 |
| `--output` | ✅ | Where to write the policy JSON |
| `--speedup-threshold` | – | Only enable FP8 above this FP8/BF16 speedup (default `1.0`) |

**Step 3 — point training at the policy.** `selective_fp8: true` alone is not
enough: without `fp8_dynamic_policy_path` the runtime falls back to the static FP8
whitelist. Set both in the model YAML:

```yaml
# configs/models/qwen2.5/qwen2_5_72b_fp8_sel.yaml
fp8: "e4m3"
fp8_recipe: "blockwise"
selective_fp8: true
fp8_dynamic_policy_path: "configs/models/qwen2.5/fp8_policy_qwen2_5_72b.json"
```

The full `TE_LAYER_PERF_*` reference, the per-TP/EP and VLM recipes, and the policy
file format are in [Adaptive FP8 Training](../../docs/source/features/adaptive_fp8.md).

## Just profile TE layers

Skip the policy step — run the benchmark and read the printed table. With no model
YAML it uses built-in shapes. **Always pick a subset with `TE_LAYER_PERF_CASESET`**
(`vision` / `llm` / `llm_235b`); the default `all` runs every built-in model and
shape — including shapes up to 131072 tokens — which can take a long time or OOM.
You can also collect it under `pytest -s`:

```bash
TE_LAYER_PERF_CASESET=vision \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
    torchrun --nproc_per_node 1 tools/te_parallel_benchmark/benchmark_te_parallel_layers.py
```
