# DreamZero Examples

Run these scripts from the repository root.

| Script | Purpose |
| --- | --- |
| `prepare_dreamzero_dataset.sh` | Validate LeRobot data and generate required DreamZero metadata. |
| `precompute_dreamzero_cache.sh` | Generate and validate DreamZero feature cache artifacts. |
| `finetune_dreamzero_wan22_5b_full_fsdp.sh` | Run Wan2.2 5B full fine-tuning with FSDP. |
| `finetune_dreamzero_wan22_5b_full_ddp_zero1.sh` | Run Wan2.2 5B full fine-tuning with DDP and ZeRO-1. |
| `finetune_dreamzero_wan22_5b_lora_fsdp.sh` | Run Wan2.2 5B LoRA fine-tuning with FSDP. |
| `finetune_dreamzero_wan21_14b_full_fsdp.sh` | Run Wan2.1 14B full fine-tuning with FSDP. |
| `finetune_dreamzero_wan21_14b_lora_fsdp.sh` | Run Wan2.1 14B LoRA fine-tuning with FSDP. |

The data-preparation wrappers are thin entry points: they wire up paths,
`torchrun` launch, and post-run validation, then call the Python CLIs in
[`tools/dreamzero_data_preprocess/`](../../tools/dreamzero_data_preprocess/README.md)
(`prepare_dataset.py`, `precompute_features.py`,
`validate_precomputed_feature_artifact.py`). See that README for the full
argument reference and for running the CLIs directly.

## Paths

The scripts use `/workspace/dreamzero/{data,checkpoints,cache}` by default.
Set the shared roots when the data and checkpoints are stored elsewhere:

```bash
export DREAMZERO_DATA_ROOT=/path/to/dreamzero/data
export DREAMZERO_CKPT_ROOT=/path/to/dreamzero/checkpoints
export DREAMZERO_CACHE_ROOT=/path/to/dreamzero/cache
```

Feature precompute loads the frozen DreamZero backbone (VAE + text encoder)
named by the model config, so the corresponding checkpoints and tokenizer must
be available (e.g. Wan2.2-TI2V-5B or Wan2.1-I2V-14B + UMT5).

## Prepare Dataset

The official preprocessed DROID dataset can be used directly; this command
validates it and fills any missing DreamZero metadata. Standard LIBERO and
DreamZero-compatible AgiBot/YAM LeRobot datasets must run this step before
training.

```bash
EMBODIMENT_TAG=oxe_droid DATA_PATH=/path/to/droid_lerobot \
  bash examples/dreamzero/preprocess/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=libero_sim DATA_PATH=/path/to/libero_lerobot \
  bash examples/dreamzero/preprocess/prepare_dreamzero_dataset.sh
```

Use `EMBODIMENT_TAG=agibot` or `EMBODIMENT_TAG=yam` for the corresponding
LeRobot datasets. The preparation step only updates `meta/`; it does not
modify parquet files or images/videos.

## Generate Cache

```bash
MODEL_NAME=dreamzero_full_wan22_5b GPUS_PER_NODE=8 \
  bash examples/dreamzero/preprocess/precompute_dreamzero_cache.sh

MODEL_NAME=dreamzero_full_wan21_14b GPUS_PER_NODE=8 \
  bash examples/dreamzero/preprocess/precompute_dreamzero_cache.sh
```

The generated cache defaults to `$DREAMZERO_CACHE_ROOT/$MODEL_NAME`, and the
wrapper validates it after the `torchrun` job finishes. Keep
`SAMPLE_TRANSFORM_SEED` identical between cache generation and training.

## Full Fine-Tuning

Omit `CACHE_DIR` to compute features online.

```bash
CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan22_5b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/dreamzero/finetuning/finetune_dreamzero_wan22_5b_full_fsdp.sh

CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan22_5b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/dreamzero/finetuning/finetune_dreamzero_wan22_5b_full_ddp_zero1.sh

CACHE_DIR=/path/to/dreamzero/cache/dreamzero_full_wan21_14b \
SAMPLE_TRANSFORM_SEED=0 \
  bash examples/dreamzero/finetuning/finetune_dreamzero_wan21_14b_full_fsdp.sh
```

For the end-to-end workflow (data → cache → training → eval) see the
[DreamZero quick start](../../docs/source/embodied_tutorial/quick_start_dreamzero.md).
