# Quick Start: DreamZero Training

This guide describes how to quickly launch **DreamZero** training within the LoongForge framework, covering weight and data preparation, Full / LoRA training, resume training, and the optional Feature Cache. Both Wan2.2-TI2V-5B and Wan2.1-I2V-14B model specifications are supported. Commands are run by default from the LoongForge repo root; the example uses `/workspace/LoongForge` — replace it with your actual environment.

## 0. Resource Preparation
### 0.1 Environment Variables
```bash
cd /workspace/LoongForge
export LOONGFORGE_PATH=$(pwd)
export DREAMZERO_CKPT_ROOT=/workspace/dreamzero/checkpoints
export DREAMZERO_DATA_ROOT=/workspace/dreamzero/data
export DREAMZERO_CACHE_ROOT=/workspace/dreamzero/cache

export WAN21_CKPT_DIR=$DREAMZERO_CKPT_ROOT/Wan2.1-I2V-14B-480P
export WAN22_CKPT_DIR=$DREAMZERO_CKPT_ROOT/Wan2.2-TI2V-5B
export DREAMZERO_AGIBOT_CKPT_DIR=$DREAMZERO_CKPT_ROOT/DreamZero-AgiBot

mkdir -p "$DREAMZERO_CKPT_ROOT" "$DREAMZERO_DATA_ROOT" "$DREAMZERO_CACHE_ROOT"
```

| Environment Variable | Description |
| --- | --- |
| `LOONGFORGE_PATH` | LoongForge repo root, used to locate training scripts and configs from any directory |
| `DREAMZERO_CKPT_ROOT` | Unified directory for all DreamZero-related weights |
| `DREAMZERO_DATA_ROOT` | Unified directory for all training datasets |
| `DREAMZERO_CACHE_ROOT` | Unified output directory for the offline Feature Cache; has no effect on online training when the cache is not used |
| `WAN21_CKPT_DIR` | Wan2.1-I2V-14B weight directory; also provides the CLIP image encoder for 5B training |
| `WAN22_CKPT_DIR` | Wan2.2-TI2V-5B weight directory |
| `DREAMZERO_AGIBOT_CKPT_DIR` | Initialization weights directory for AgiBot / YAM LoRA training; not required for DROID or LIBERO |

All paths above can be adjusted to your actual storage layout. The training scripts read the corresponding variables and do not require the exact directory structure shown in the example.

### 0.2 Download Weights

| Scenario | Required Weights |
| --- | --- |
| Wan2.2 5B DROID / LIBERO | Full Wan2.2 5B; CLIP image encoder from Wan2.1 |
| Wan2.1 14B DROID | Full Wan2.1 14B |
| Wan2.1 14B AgiBot / YAM LoRA | Full Wan2.1 14B; DreamZero-AgiBot initialization weights |

```bash
python -m pip install -U "huggingface_hub[cli]"

# Wan backbone
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN21_CKPT_DIR"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"

# When training only 5B, Wan2.1 only needs to add CLIP (Wan2.2 repo doesn't include CLIP; 5B still uses it to encode the first frame)
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --local-dir "$WAN21_CKPT_DIR"

# AgiBot / YAM LoRA initialization weights (not required for DROID, LIBERO)
hf download GEAR-Dreams/DreamZero-AgiBot \
    --repo-type model --local-dir "$DREAMZERO_AGIBOT_CKPT_DIR"
```

The tokenizer uses by default `$WAN22_CKPT_DIR/google/umt5-xxl` (5B) / `$WAN21_CKPT_DIR/google/umt5-xxl` (14B) bundled with the Wan repo, so no separate download needed. If a separate shared directory is needed:

```bash
export TOKENIZER_PATH=$DREAMZERO_CKPT_ROOT/umt5-xxl
hf download google/umt5-xxl \
    special_tokens_map.json spiece.model tokenizer.json tokenizer_config.json \
    --local-dir "$TOKENIZER_PATH"
```

### 0.3 Weight Integrity Check
```bash
test -f "$WAN22_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json"
test -f "$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -f "$WAN22_CKPT_DIR/Wan2.2_VAE.pth"
test -f "$WAN21_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
test -f "$WAN22_CKPT_DIR/google/umt5-xxl/spiece.model"

# Additional checks when training 14B:
test -f "$WAN21_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json"
test -f "$WAN21_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
test -f "$WAN21_CKPT_DIR/Wan2.1_VAE.pth"
echo "DreamZero weight check passed"
```

### 0.4 Download Datasets
```bash
# DROID (approx 131GB, already converted to LeRobot v2)
export DROID_DATA_ROOT=$DREAMZERO_DATA_ROOT/droid_lerobot
hf download GEAR-Dreams/DreamZero-DROID-Data \
    --repo-type dataset --local-dir "$DROID_DATA_ROOT"

# LIBERO (for simulation training)
export LIBERO_DATA_ROOT=$DREAMZERO_DATA_ROOT/libero_lerobot
hf download physical-intelligence/libero \
    --repo-type dataset --local-dir "$LIBERO_DATA_ROOT"
```

For AgiBot / YAM, there is currently no directly usable LeRobot v2 training data; the data must be converted per the field spec below.

## 1. Data Configuration
### 1.1 Input Format
```text
dataset_root/
├── data/**/episode_*.parquet
├── videos/                         # required when info.json uses video dtype
└── meta/
    ├── info.json                   # codebase_version must start with v2
    ├── tasks.jsonl
    └── episodes.jsonl
```

| `EMBODIMENT_TAG` | State / Action | Video Fields | Language or Task Fields |
| --- | --- | --- | --- |
| `oxe_droid` | `observation.state` ≥14 dims; `action` ≥21 dims | `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left` | Three `annotation.language.*` fields |
| `libero_sim` | `state` 8 dims; `actions` 7 dims | `image`, `wrist_image` | `task_index` |
| `agibot` | `observation.state` ≥20 dims; `action` ≥22 dims | `top_head`, `hand_left`, `hand_right` | `task_index` |
| `yam` | packed `observation.state` / `action` ≥46 dims | top/left/right three `*-camera-images-rgb` fields | `task_index` |

Refer to `PRESETS` in `tools/data_preprocess/embodied/dreamzero/prepare_dataset.py` for the full field spec.

### 1.2 Generate DreamZero Metadata
```bash
EMBODIMENT_TAG=oxe_droid DATA_PATH="$DROID_DATA_ROOT" \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=libero_sim DATA_PATH="$LIBERO_DATA_ROOT" \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=agibot DATA_PATH=/path/to/agibot_lerobot \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

EMBODIMENT_TAG=yam DATA_PATH=/path/to/yam_lerobot \
    bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh
```

Common Options:

- `FORCE=1`: Overwrite existing metadata
- `SKIP_STATISTICS=1`: Generate schema only, skip statistics; statistics must be filled in before full training

On success, prints `DreamZero dataset preparation complete`.

## 2. Launch Training
### 2.1 Full / LoRA Training
```bash
cd "$LOONGFORGE_PATH"
export DATA_PATH=$DROID_DATA_ROOT
export WANDB_MODE=disabled

# 5B Full with FSDP + Delta-FP8, default 200000 steps
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh

# 5B LoRA
TRAIN_ITERS=10000 SAVE_INTERVAL=1000 \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_lora_fsdp_finetune.sh

# 14B Full
bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_full_fsdp_finetune.sh

# 14B LoRA, DROID
TRAIN_ITERS=10000 SAVE_INTERVAL=1000 \
    bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_lora_fsdp_finetune.sh

# LIBERO 5B Full
MODEL_NAME=dreamzero_libero_wan22_5b DATA_PATH="$LIBERO_DATA_ROOT" \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh

# AgiBot / YAM 14B LoRA (initialized from DreamZero-AgiBot)
MODEL_NAME=dreamzero_agibot_wan21_14b DATA_PATH=/path/to/agibot_lerobot \
    bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_lora_fsdp_finetune.sh

MODEL_NAME=dreamzero_yam_wan21_14b DATA_PATH=/path/to/yam_lerobot \
    bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_lora_fsdp_finetune.sh
```

For the DROID scenario, `MODEL_NAME` does not need to be set; the training script automatically picks the correct config. Only when switching to LIBERO, AgiBot, or YAM do you need to override it as shown.

Defaults:

| Scenario | Distributed Strategy | Per-GPU batch | Default global batch | Default LR |
| --- | --- | --- | --- | --- |
| 5B Full | 8-GPU FSDP + Delta-FP8 | 1 | 8 | `1e-5` |
| 5B LoRA | 8-GPU FSDP | 1 | 8 | `1e-5` |
| 14B Full | 8-GPU FSDP | 1 | 8 | `1e-5` |
| 14B DROID LoRA | 8-GPU FSDP | 1 | 8 | `1e-4` |
| 14B AgiBot / YAM LoRA | 8-GPU FSDP | 1 | 8 | `1e-5` |

Batch sizes can be adjusted via `PER_DEVICE_BATCH_SIZE` and `GLOBAL_BATCH_SIZE`, but must satisfy `GLOBAL_BATCH_SIZE % (GPUS_PER_NODE * NNODES * PER_DEVICE_BATCH_SIZE) == 0`. When fewer than 8 GPUs are available, set `GPUS_PER_NODE`; the default global batch size will scale accordingly.

### 2.2 Resume Training
```bash
TRAIN_ITERS=20000 SAVE_INTERVAL=1000 \
OUTPUT_DIR=/workspace/outputs/dreamzero/droid_wan22_5b_lora \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_lora_fsdp_finetune.sh \
    --resume
```

`--resume` will load the latest checkpoint from `$OUTPUT_DIR/checkpoints`. When resuming, keep the same world size and batch config as when saved; the run aborts if no checkpoint is found.

### 2.3 Offline Feature Cache (Optional)
```bash
# Generate full cache (5B defaults to video_latents + prompt_embs; 14B default does not include prompt_embs)
MODEL_NAME=dreamzero_full_wan22_5b DATA_PATH="$DROID_DATA_ROOT" \
CACHE_OUTPUT_DIR=$DREAMZERO_CACHE_ROOT/dreamzero_full_wan22_5b \
GPUS_PER_NODE=8 SAMPLE_TRANSFORM_SEED=0 VALIDATION_REQUIRE_FULL_COVERAGE=1 \
    bash examples/embodied/dreamzero/precompute_dreamzero_cache.sh

# Train with cache (must match SAMPLE_TRANSFORM_SEED, resolution, and language chunk config used at generation time)
CACHE_DIR=$DREAMZERO_CACHE_ROOT/dreamzero_full_wan22_5b SAMPLE_TRANSFORM_SEED=0 \
    bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh
```

The training script strictly validates the cache at startup. When manifest, `_SUCCESS`, or transform config mismatch, training aborts before the first step.

### 2.4 Performance Optimization Switches
The DROID 5B / 14B configs enable validated default optimizations. The 5B Full FSDP recipe also enables Delta-FP8 AllGather by default on supported CUDA/NCCL devices; pass `--no-fsdp-delta-fp8-allgather` for a native BF16 communication A/B comparison. LIBERO, AgiBot, and YAM use conservative defaults. After adjusting performance switches, we recommend running 3–10 steps first and checking loss, grad norm, and memory usage.

Control group (disable all kernel optimizations, for isolating loss / performance issues):

```bash
# Wan2.2 5B
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh \
    model.flash_attention_dense=false model.compile_causal_attention_block=false \
    model.batch_vae_encode=false 'model.prompt_emb_cache=""'

# Wan2.1 14B
bash examples/embodied/dreamzero/run_dreamzero_wan21_14b_full_fsdp_finetune.sh \
    model.skip_single_state_attention=false model.compile_causal_cross_attention=false \
    model.compile_cross_attention_emulate_precision_casts=false model.compile_block_norm_modulate=false \
    model.qk_rmsnorm_impl=wan model.manual_self_attn_linear_backward=false model.fused_rope=false \
    'model.prompt_emb_cache=""' model.cache_fa_lens=false model.cache_fa_lens_clone=false
```

An empty string must be quoted as `'model.prompt_emb_cache=""'`.

Main switches:

| Switch | Default | Description |
| --- | --- | --- |
| `model.flash_attention_dense` | `true` | dense attention uses FlashAttention directly |
| `model.compile_causal_attention_block` | `true` | compile the causal attention block |
| `model.batch_vae_encode` | `true` | batched VAE encoding; higher throughput but larger peak memory; disable when OOM |
| `model.prompt_emb_cache` | `gpu` | cache frozen Text Encoder outputs; set to `cpu` or empty string to disable |
| `model.compile_causal_cross_attention` (14B) | `true` | `torch.compile` compile causal cross-attention |
| `model.fused_rope` (14B) | `true` | fused RoPE kernel |
| `model.cache_fa_lens` (14B) | `true` | cache FlashAttention cumulative sequence lengths |

### 2.5 Correctness Verification

To ensure training accuracy is not affected by optimizations, we performed a step-by-step loss comparison between LoongForge's DreamZero adaptation and the official implementation under the same data, weights, and training configuration. Results show that LoongForge's performance optimizations produce no loss in training accuracy:

![DreamZero training pipeline overview](../../assets/images/precision/dreamzero.png)
