# DreamZero Examples

Run these scripts from the repository root.

| Script | Purpose |
| --- | --- |
| `prepare_dreamzero_dataset.sh` | Validate LeRobot data and generate required DreamZero metadata. |
| `precompute_dreamzero_cache.sh` | Generate and validate DreamZero feature cache artifacts. |
| `run_dreamzero_wan22_5b_full_finetune.sh` | Run Wan2.2 5B full fine-tuning with DDP and ZeRO-1. |
| `run_dreamzero_wan21_14b_full_finetune.sh` | Run Wan2.1 14B full fine-tuning with FSDP. |

## Paths

The scripts use `/workspace/dreamzero/{data,checkpoints,cache}` by default.
Set the shared roots when the data and checkpoints are stored elsewhere:

```bash
export DREAMZERO_DATA_ROOT=/path/to/dreamzero/data
export DREAMZERO_CKPT_ROOT=/path/to/dreamzero/checkpoints
export DREAMZERO_CACHE_ROOT=/path/to/dreamzero/cache
```

## Prepare Dataset

The official preprocessed DROID dataset can be used directly; this command
validates it and fills any missing DreamZero metadata. Standard LIBERO and
DreamZero-compatible AgiBot/YAM LeRobot datasets must run this step before
training.

```bash
EMBODIMENT_TAG=oxe_droid DATA_PATH=/path/to/droid_lerobot \
  bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=libero_sim DATA_PATH=/path/to/libero_lerobot \
  bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh
```

Use `EMBODIMENT_TAG=agibot` or `EMBODIMENT_TAG=yam` for the corresponding
LeRobot datasets. The preparation step only updates `meta/`; it does not
modify parquet files or images/videos.

## Generate Cache

```bash
MODEL_NAME=dreamzero_full_wan22_5b GPUS_PER_NODE=8 \
  bash examples/embodied/dreamzero/precompute_dreamzero_cache.sh

MODEL_NAME=dreamzero_full_wan21_14b GPUS_PER_NODE=8 \
  bash examples/embodied/dreamzero/precompute_dreamzero_cache.sh
```

The generated cache defaults to `$DREAMZERO_CACHE_ROOT/$MODEL_NAME`.

## Full Fine-Tuning

Omit `CACHE_DIR` to compute features online. When using a cache, keep
`SAMPLE_TRANSFORM_SEED` consistent with cache generation.

```bash
CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan22_5b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_finetune.sh

CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan21_14b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_full_finetune.sh
```
