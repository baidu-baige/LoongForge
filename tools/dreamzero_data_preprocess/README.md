# DreamZero Data Preprocess

Get a LeRobot dataset ready for DreamZero (world-action model) training: write
the DreamZero/GEAR metadata the trainer needs, then optionally precompute
frozen-feature caches (VAE latents, prompt embeddings) so training doesn't
re-run the frozen encoders every step.

| Script | Stage | Output |
|--------|-------|--------|
| `prepare_dataset.py` | Dataset preparation (required) | DreamZero/GEAR metadata under `<dataset>/meta/` |
| `precompute_features.py` | Frozen-feature cache (optional) | Per-sample / sharded feature cache + `manifest.json` |
| `validate_precomputed_feature_artifact.py` | Cache validation (optional) | Pass/fail smoke check of a cache artifact |

## Prerequisites

`prepare_dataset.py` is self-contained — it only needs `numpy`, `pandas`, and
`tqdm`; nothing from LoongForge has to be on the path.

`precompute_features.py` and `validate_precomputed_feature_artifact.py` import
`loongforge.embodied`, so the LoongForge repo root must be importable:

```bash
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-/workspace/LoongForge}
export PYTHONPATH=$LOONGFORGE_PATH:$PYTHONPATH
```

Feature precompute additionally loads the frozen DreamZero backbone (VAE + text
encoder) named by the model config, so the corresponding checkpoints and
tokenizer must be available (e.g. Wan2.2-TI2V-5B or Wan2.1-I2V-14B + UMT5).

Ready-to-run wrappers live at
`examples/embodied/dreamzero/prepare_dreamzero_dataset.sh` and
`examples/embodied/dreamzero/precompute_dreamzero_cache.sh` — prefer them as the
entry points; they wire up paths, distributed launch, and post-run validation.

## 1. Prepare the dataset (required)

Validates the source LeRobot layout and writes only DreamZero/GEAR metadata
under `meta/`. It does **not** copy, transcode, or modify parquet or
image/video payloads.

```bash
python ${LOONGFORGE_PATH}/tools/dreamzero_data_preprocess/prepare_dataset.py \
    --dataset-path   /path/to/droid_lerobot \
    --embodiment-tag oxe_droid
```

**Key arguments:**

| Argument | Required | Description |
|----------|:---:|-------------|
| `--dataset-path` | ✅ | Root of the source LeRobot dataset (must contain `meta/info.json`) |
| `--embodiment-tag` | ✅ | Dataset preset: `oxe_droid`, `libero_sim`, `agibot`, or `yam` |
| `--force` | – | Replace existing modality/embodiment/statistics metadata |
| `--skip-statistics` | – | Write schema metadata only; training may compute missing stats |
| `--max-relative-stat-episodes` | – | Sampled episodes for relative-action stats (default 10000; `<=0` uses all) |
| `--allow-partial-statistics` | – | Allow stats when parquet count differs from `total_episodes` |

The per-embodiment field layout is defined by `PRESETS` in
`prepare_dataset.py` — consult it for the exact state/action/video schema.

## 2. Precompute frozen-feature cache (optional)

Precompute the frozen encoders' outputs once so training reads them from disk
instead of recomputing every step: main-video VAE latents, optional first-frame
latents (I2V), and optional frozen text-encoder prompt embeddings. Runs
distributed under `torchrun`. On completion it prints the
`model.precomputed_cache.*` overrides to add to your training command.

```bash
PYTHONPATH=$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun --nproc_per_node 8 \
        ${LOONGFORGE_PATH}/tools/dreamzero_data_preprocess/precompute_features.py \
            --config-file    configs/models/embodied/dreamzero_wan22_5b.yaml \
            --data-path      /path/to/droid_lerobot \
            --output-dir     /path/to/dreamzero_cache \
            --tokenizer-path /path/to/umt5-xxl \
            --storage-format tensor_shards
```

**Key arguments:**

| Argument | Required | Description |
|----------|:---:|-------------|
| `--config-file` | – | DreamZero model YAML (default `dreamzero_wan22_5b.yaml`) |
| `--data-path` | ✅ | Prepared LeRobot dataset root |
| `--output-dir` | ✅ | Cache output directory (holds tensors + `manifest.json`) |
| `--tokenizer-path` | – | Frozen text tokenizer/encoder path (or set in config) |
| `--start-index` / `--num-samples` | – | Slice of samples to cache (default from index 0, 8 samples) |
| `--indices-file` | – | JSON list of explicit dataset indices to cache |
| `--storage-format` | – | `sample_files` (one `.pt` per sample, default) or `tensor_shards` (mmap shards + index; use for full-dataset caches) |
| `--tensor-shard-size` | – | Max samples per shard for `tensor_shards` (default 4096) |
| `--batch-size` / `--num-workers` / `--prefetch-factor` | – | Dataloader throughput knobs |
| `--dtype` | – | `bf16` (default) or `fp32` |
| `--include-video-latents` | – | Cache main-video VAE latents (default on) |
| `--include-first-frame-latents` | – | Cache I2V first-frame latents (default: auto per backbone) |
| `--include-prompt-embs` | – | Cache raw prompt embeddings (default: auto — on for 5B/TI2V, off for 14B/I2V) |
| `--use-sample-transform-seed` / `--sample-transform-seed` | – | Fix per-sample transform seed (default on, seed 0); **must match training** |
| `--require-full-language-chunks` | – | Skip samples lacking a complete language-conditioned chunk |
| `--overwrite` | – | Regenerate existing cache entries |

> Per-sample transform seeds must be identical between cache generation and
> training, or cached features will not align with the live pipeline.

## 3. Validate the cache (optional)

Smoke-validates a generated cache artifact against its manifest.

```bash
PYTHONPATH=$LOONGFORGE_PATH:$PYTHONPATH \
    python ${LOONGFORGE_PATH}/tools/dreamzero_data_preprocess/validate_precomputed_feature_artifact.py \
        --manifest  /path/to/dreamzero_cache/manifest.json \
        --cache-dir /path/to/dreamzero_cache
```

**Key arguments:**

| Argument | Required | Description |
|----------|:---:|-------------|
| `--manifest` | ✅ | Path to the artifact `manifest.json` |
| `--cache-dir` | – | Override the cache tensor directory |
| `--require-full-coverage` | – | Require the cache to cover the entire dataset (production artifacts) |
| `--sample-count` / `--sample-seed` | – | Number of random samples to deep-check (default 8, seed 0) |
| `--expect-use-sample-transform-seed` / `--expect-sample-transform-seed` | – | Assert the cache was built with the expected seed settings |
| `--check-manifest-file-hash` / `--check-sample-hash` | – | Toggle hash verification (default on) |

## Full option reference & tutorial

```bash
python ${LOONGFORGE_PATH}/tools/dreamzero_data_preprocess/prepare_dataset.py --help
python ${LOONGFORGE_PATH}/tools/dreamzero_data_preprocess/precompute_features.py --help
python ${LOONGFORGE_PATH}/tools/dreamzero_data_preprocess/validate_precomputed_feature_artifact.py --help
```

For the end-to-end DreamZero workflow (data → cache → training → eval) see the
[DreamZero quick start](../../docs/source/embodied_tutorial/quick_start_dreamzero.md).
