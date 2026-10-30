# Quick Start: FastWAM Training

This guide walks you through launching a **FastWAM** SFT (Supervised Fine-Tuning) job in the LoongForge framework.

## 0. Resource Preparation

### 0.1 Model Weights

#### 0.1.1 Extract the Action DiT Backbone

Extract and linearly interpolate the Action DiT head weights from the Wan2.2 video DiT checkpoint, producing a `.pt` file that training can load directly:

```bash
LOCAL_MODEL_PATH=/data/models \
OUTPUT=$LOONGFORGE_PATH/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    bash examples/embodied/fastwam/preprocess_action_dit_backbone.sh
```

#### 0.1.2 Wan-AI/Wan2.2-TI2V-5B Weights

```bash
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir /workspace/huggingface.co/Wan-AI/Wan2.2-TI2V-5B
```

### 0.2 Tokenizer / Processor

Encode all task instructions in the dataset into text embeds in one pass, then read them on demand during training:

```bash
DATASET_PATH=/data/libero \
TEXT_EMBEDDING_CACHE_DIR=/data/cache/fastwam_text_embeds \
    bash examples/embodied/fastwam/precompute_text_embeds.sh
```

### 0.3 Dataset

```bash
hf download yuanty/LIBERO-fastwam --local-dir /workspace/data/LIBERO-fastwam
cd /workspace/data/LIBERO-fastwam
tar -zxvf libero_spatial_no_noops_lerobot.tar.gz
```

## 1. Data Configuration

Data-related fields are configured mainly through YAML and environment variables.

**Model and data YAML:**

```yaml
model:
  action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt  # Path to the action dit generated above
  redirect_common_files: true
  dtype: bfloat16

data:
  text_embedding_cache_dir: data/text_embeds_cache/libero  # Path to the text embeds generated above
```

**Environment variables:**

```bash
export LOONGFORGE_PATH=/workspace/LoongForge                                                    # Repository root
export DATASET_PATH=/workspace/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot              # LeRobot dataset root
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/huggingface.co/Wan-AI/Wan2.2-TI2V-5B                # Wan2.2 5B base model path
```

## 2. Launch Training

### 2.1 Launch Script

Single-node DDP training example:

```bash
bash examples/embodied/fastwam/run_fastwam_sft_ddp_finetune.sh
```

### 2.2 Correctness Verification

To ensure that training accuracy is not affected by our optimizations, we ran a step-by-step action loss comparison between LoongForge's FastWAM and the official implementation under identical data, weights, and training configurations. The results show that LoongForge's performance optimizations are lossless with respect to training accuracy:
![alt text](../../assets/images/precision/fastwam.png)

## 3. Validated Recipe (LIBERO 4-suite)

The recipe below is the one used for the reference 20k SFT checkpoint. On 8×A800-80GB with per-device batch 16 (global batch 128) it runs at ~1.37 s/step, ~71 GB/GPU, ~8.4 h:

```bash
export DATASET_PATH=/path/to/LIBERO-fastwam/libero_spatial_no_noops_lerobot,/path/to/LIBERO-fastwam/libero_object_no_noops_lerobot,/path/to/LIBERO-fastwam/libero_goal_no_noops_lerobot,/path/to/LIBERO-fastwam/libero_10_no_noops_lerobot

LR_DECAY_STYLE=linear_warmup_cosine_annealing \
CLIP_GRAD=1.0 \
ZERO_MASTER_PARAM_DTYPE=fp32 \
SAVE_FORMAT=dcp ASYNC_SAVE=1 \
GPUS_PER_NODE=8 TRAIN_ITERS=20000 PER_DEVICE_BATCH_SIZE=16 SEED=42 \
EXTRA_MODEL_OVERRIDES="model.mot_compile_blocks=none" \
EXTRA_DATA_OVERRIDES="data.norm_stats_path=/path/to/dataset_stats.json" \
bash examples/embodied/fastwam/run_fastwam_sft_ddp_zero1_finetune.sh \
  --config-file configs/models/embodied/fastwam.yaml \
  --optimizer TorchFusedAdamW \
  --lr-base 1e-4 --lr-warmup-iters 1000 --min-lr 1e-6
```

Notes:

- `ZERO_MASTER_PARAM_DTYPE=fp32` keeps an fp32 master copy of the weights and fp32 Adam state, matching the reference implementation's `BF16_Optimizer`. With bf16 direct updates, most weights receive an exactly-zero update per step once the LR decays (measured: 99.5% of elements at step ~18.5k), so the tail of training stalls. Cost: ~1% step time and a larger checkpoint.
- `CLIP_GRAD=1.0` is the default. Disabling it costs 3.4 pt on `libero_10` in this recipe (McNemar p=0.02); the other three suites are within ±0.4 pt.
- `data.norm_stats_path` must be the `dataset_stats.json` that matches the checkpoint: `normalization_mode: q99` needs its `q01`/`q99` entries, otherwise the data pipeline raises `KeyError: 'q01'`.
- The four suites are sampled jointly by frame count (not per-suite equal weight), which is what the `DATASET_PATH` above produces.

### Results (2000 episodes, 4 suites × 500)

| suite | success |
|---|---|
| libero_spatial | 97.00% (485/500) |
| libero_object | 99.60% (498/500) |
| libero_goal | 97.20% (486/500) |
| libero_10 | 94.20% (471/500) |
| **TOTAL** | **97.00% (1940/2000)** |

## 4. Evaluation

See the [LIBERO benchmark page](../eval_docs/benchmarks/libero.md) for environment setup and protocol notes. Fill the `/path/to/...` placeholders in the suite YAML (checkpoint, Wan2.2 root, `dataset_stats.json`) first:

```bash
CONFIG=examples/embodied/fastwam/eval/configs/libero/libero_10.yaml \
REPO_ROOT=/path/to/LoongForge-VLA \
BENCHMARK_PYTHON=/path/to/libero/bin/python \
DIFFSYNTH_MODEL_BASE_PATH=/path/to/models_root \
bash examples/embodied/fastwam/eval/run_libero_eval.sh
```

The shipped YAMLs use the reference eval protocol: `episodes_per_task: 50`, `max_steps` 400 (`spatial`/`object`/`goal`) and 700 (`libero_10`), action `sigma_shift: 1.0`, `num_steps_wait: 30`, `controller_force_update: false`, `prefer_controller_ee: false`, binarized gripper, `noise_seed: 42`.
