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
EP, for MoE) size; each run writes one report:

```bash
TE_LAYER_PERF_OMNI_CONFIG_PATH=configs/models/qwen2.5/qwen2_5_72b.yaml \
TE_LAYER_PERF_TP_SIZE=4 \
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
TE_LAYER_PERF_REPORT_PATH=outputs/report_tp4.json \
    torchrun --nproc_per_node 4 tools/te_parallel_benchmark/benchmark_te_parallel_layers.py
```

**Step 2 — merge the reports into one policy file**, then point training at it via
`selective_fp8`:

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

The full `TE_LAYER_PERF_*` reference, the per-TP/EP and VLM recipes, and the policy
file format are in [Adaptive FP8 Training](../../docs/source/features/adaptive_fp8.md).

## Just profile TE layers

Skip the policy step — run the benchmark and read the printed table. With no model
YAML it uses built-in shapes (`TE_LAYER_PERF_CASESET=vision|llm|llm_235b` picks a
subset); you can also collect it under `pytest -s`:

```bash
TE_LAYER_PERF_PRECISIONS="bf16,fp8" \
    torchrun --nproc_per_node 1 tools/te_parallel_benchmark/benchmark_te_parallel_layers.py
```
