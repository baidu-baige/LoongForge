# DreamZero Examples

Run these scripts from the repository root.

| Script | Purpose |
| --- | --- |
| `prepare_dreamzero_dataset.sh` | Validate LeRobot data and generate required DreamZero metadata. |
| `precompute_dreamzero_cache.sh` | Generate and validate DreamZero feature cache artifacts. |
| `finetune_dreamzero_wan22_5b_full_fsdp.sh` | Run Wan2.2 5B full fine-tuning with FSDP. |
| `finetune_dreamzero_wan22_5b_full_ddp_zero1.sh` | Run Wan2.2 5B full fine-tuning with DDP and ZeRO-1. |
| `finetune_dreamzero_wan21_14b_full_fsdp.sh` | Run Wan2.1 14B full fine-tuning with FSDP. |

The shell wrappers call these Python entry points in the same directory:

| Script | Stage | Output |
| --- | --- | --- |
| `prepare_dataset.py` | Dataset preparation (required) | DreamZero/GEAR metadata under `<dataset>/meta/` |
| `precompute_features.py` | Frozen-feature cache (optional) | Per-sample / sharded feature cache + `manifest.json` |
| `validate_precomputed_feature_artifact.py` | Cache validation (optional) | Pass/fail smoke check of a cache artifact |

## Paths

The scripts use `/workspace/dreamzero/{data,checkpoints,cache}` by default.
Set the shared roots when the data and checkpoints are stored elsewhere:

```bash
export DREAMZERO_DATA_ROOT=/path/to/dreamzero/data
export DREAMZERO_CKPT_ROOT=/path/to/dreamzero/checkpoints
export DREAMZERO_CACHE_ROOT=/path/to/dreamzero/cache
```

## Prerequisites

`prepare_dataset.py` only needs `numpy`, `pandas`, and `tqdm`; nothing from
LoongForge has to be on the path.

`precompute_features.py` and `validate_precomputed_feature_artifact.py` import
`loongforge`, so the LoongForge repo root must be importable (the shell
wrappers set this):

```bash
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-/workspace/LoongForge}
export PYTHONPATH=$LOONGFORGE_PATH:$PYTHONPATH
```

Feature precompute also loads the frozen DreamZero backbone (VAE + text
encoder) named by the model config, so the corresponding checkpoints and
tokenizer must be available (e.g. Wan2.2-TI2V-5B or Wan2.1-I2V-14B + UMT5).

## Prepare Dataset
The official preprocessed DROID dataset can be used directly; this command
validates it and fills any missing DreamZero metadata. Standard LIBERO and
DreamZero-compatible AgiBot/YAM LeRobot datasets must run this step before
training.

```bash
EMBODIMENT_TAG=oxe_droid DATA_PATH=/path/to/droid_lerobot \
  bash examples/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=libero_sim DATA_PATH=/path/to/libero_lerobot \
  bash examples/dreamzero/prepare_dreamzero_dataset.sh
```

Use `EMBODIMENT_TAG=agibot` or `EMBODIMENT_TAG=yam` for the corresponding
LeRobot datasets. The preparation step only updates `meta/`; it does not
modify parquet files or images/videos.

Key arguments of `prepare_dataset.py`:

| Argument | Required | Description |
| --- | :---: | --- |
| `--dataset-path` | ✅ | Root of the source LeRobot dataset (must contain `meta/info.json`) |
| `--embodiment-tag` | ✅ | Dataset preset: `oxe_droid`, `libero_sim`, `agibot`, or `yam` |
| `--force` | – | Replace existing modality/embodiment/statistics metadata |
| `--skip-statistics` | – | Write schema metadata only; training may compute missing stats |
| `--max-relative-stat-episodes` | – | Sampled episodes for relative-action stats (default 10000; `<=0` uses all) |
| `--allow-partial-statistics` | – | Allow stats when parquet count differs from `total_episodes` |

The per-embodiment field layout is defined by `PRESETS` in
`prepare_dataset.py`.

## Generate Cache

```bash
MODEL_NAME=dreamzero_full_wan22_5b GPUS_PER_NODE=8 \
  bash examples/dreamzero/precompute_dreamzero_cache.sh

MODEL_NAME=dreamzero_full_wan21_14b GPUS_PER_NODE=8 \
  bash examples/dreamzero/precompute_dreamzero_cache.sh
```

The generated cache defaults to `$DREAMZERO_CACHE_ROOT/$MODEL_NAME`. The
wrapper runs `precompute_features.py` under `torchrun`, then validates the
result. On completion `precompute_features.py` prints the
`model.precomputed_cache.*` overrides to add to the training command.
Key arguments of `precompute_features.py`:

| Argument | Required | Description |
| --- | :---: | --- |
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

## Validate Cache

`precompute_dreamzero_cache.sh` already validates the cache it generates. To
check an existing cache:

```bash
PYTHONPATH=$LOONGFORGE_PATH:$PYTHONPATH \
    python examples/dreamzero/validate_precomputed_feature_artifact.py \
        --manifest  /path/to/dreamzero_cache/manifest.json \
        --cache-dir /path/to/dreamzero_cache
```

| Argument | Required | Description |
| --- | :---: | --- |
| `--manifest` | ✅ | Path to the artifact `manifest.json` |
| `--cache-dir` | – | Override the cache tensor directory |
| `--require-full-coverage` | – | Require the cache to cover the entire dataset (production artifacts) |
| `--sample-count` / `--sample-seed` | – | Number of random samples to deep-check (default 8, seed 0) |
| `--expect-use-sample-transform-seed` / `--expect-sample-transform-seed` | – | Assert the cache was built with the expected seed settings |
| `--check-manifest-file-hash` / `--check-sample-hash` | – | Toggle hash verification (default on) |

Run any entry point with `--help` for the full option list.

## Full Fine-Tuning

Omit `CACHE_DIR` to compute features online. When using a cache, keep
`SAMPLE_TRANSFORM_SEED` consistent with cache generation.

```bash
CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan22_5b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/dreamzero/finetune_dreamzero_wan22_5b_full_fsdp.sh

CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan22_5b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/dreamzero/finetune_dreamzero_wan22_5b_full_ddp_zero1.sh

CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan21_14b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/dreamzero/finetune_dreamzero_wan21_14b_full_fsdp.sh
```

For the end-to-end workflow (data → cache → training → eval) see the
[DreamZero quick start](../../docs/source/embodied_tutorial/quick_start_dreamzero.md).
