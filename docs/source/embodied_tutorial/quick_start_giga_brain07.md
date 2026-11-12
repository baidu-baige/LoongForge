# Quick Start: GigaBrain-0.7 Training

This document introduces how to quickly launch **GigaBrain-0.7** SFT (Supervised Fine-Tuning) training in the LoongForge framework.

## 0. Resource Preparation
### 0.1 Model Weights
The GigaBrain-0.7 main weights (PaliGemma2 vision-language backbone + action expert) are specified by `GB0_PRETRAINED` and loaded via `model.pretrained`. Taking the officially released `GigaBrain-0.7-3.5B-Base` weights as an example, download them to a local directory:

```bash
hf download open-gigaai/GigaBrain-0.7-3.5B-Base --local-dir /workspace/ckpt/open-gigaai/GigaBrain-0.7-3.5B-Base
```
### 0.2 Tokenizer
GigaBrain-0.7 uses two tokenizers: the PaliGemma text tokenizer (`GB0_TOKENIZER`, loaded via `--tokenizer-path` and `data.tokenizer_model_path`) and the FAST action tokenizer (`GB0_FAST_TOKENIZER`, loaded via `data.fast_tokenizer_path`):

```bash
hf download google/paligemma-3b-pt-224 --local-dir /workspace/ckpt/paligemma-3b-pt-224
hf download physical-intelligence/fast --local-dir /workspace/ckpt/physical-intelligence/fast
```
### 0.3 Dataset
GigaBrain-0.7 uses datasets in LeRobot v3.0 format; the example is `task_327` from AgiBot World. A normalization statistics file (`norm_stats.json`) matching the embodiment is also required:

```bash
export GB0_DATA_PATH_V30=/workspace/data/agibot-world/AgiBotWorldLerobot_v30/agibotworld/task_327
export GB0_NORM_STATS=/workspace/data/agibot-world/AgiBotWorldLerobot/agibot_a2d_norm_stats.json
```
## 1. Data Configuration
GigaBrain-0.7 reuses LoongForge's shared LeRobot v3 reader via `--dataset-strategy giga_brain07`, with **no extra offline preprocessing required**. GigaBrain-0.7's per-sample transform (multi-view image resizing, state/action normalization, language and FAST action tokenization) and batch collator are performed online during training.

Data defaults live in the `data` section of `configs/models/giga_brain07/giga_brain07.yaml`:

```yaml
data:
  camera_keys:
  - observation.images.top_head
  - observation.images.hand_left
  - observation.images.hand_right
  action_chunk: 50
  video_backend: torchcodec
  robot_type: a2d
  embodiment_override: AGIBOT_G1
  embodiment_id: '1'
  resize_hw: [224, 224]
  max_length: 120
  use_quantiles: true
```
Field descriptions:

- `camera_keys`: keys of multi-view images in the LeRobot dataset; order defines the view order (the first is the primary view).
- `action_chunk`: action chunk length per sample; must match `model.action_horizon`.
- `video_backend`: video decoding backend.
- `robot_type` / `embodiment_override` / `embodiment_id`: embodiment type and ID. `embodiment_id` selects the embodiment-specific parameters in the model, and must match the value used when training the checkpoint.
- `resize_hw`: input image resize size.
- `max_length`: maximum language token length.
- `use_quantiles`: normalize with the quantiles in `norm_stats.json`.

## 2. Launch Training
First set up the paths uniformly (refer to `examples/giga_brain07/run_giga_brain07_torchrun.sh`):

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export GB0_PRETRAINED=/workspace/ckpt/open-gigaai/GigaBrain-0.7-3.5B-Base
export GB0_TOKENIZER=/workspace/ckpt/paligemma-3b-pt-224
export GB0_FAST_TOKENIZER=/workspace/ckpt/physical-intelligence/fast
export GB0_DATA_PATH_V30=/workspace/data/agibot-world/AgiBotWorldLerobot_v30/agibotworld/task_327
export GB0_NORM_STATS=/workspace/data/agibot-world/AgiBotWorldLerobot/agibot_a2d_norm_stats.json
export OUTPUT_DIR=/workspace/outputs/giga_brain07_fsdp
```
### 2.1 Launch Script
Single-node 8-GPU FSDP2 fine-tuning:

```bash
bash examples/giga_brain07/run_giga_brain07_torchrun.sh
```
The script enables the following optimizations by default; they can be disabled or adjusted via environment variables:

```bash
GB0_COMPILE=decoder,vision           # Regional torch.compile on decoder and vision encoder layers; set empty to disable
GB0_KV_SLICE=1                       # Each decoder layer receives only its own KV cache; set 0 to disable
GB0_AC_PATTERNS=...                  # Modules to recompute; defaults cover all 26 decoder layers and every vision encoder layer
```
Other common environment variables: `GB0_BATCH_SIZE` (per-GPU batch size, default 80), `GB0_MAX_STEPS` (training steps, default 100), `GB0_NUM_WORKERS` (DataLoader workers, default 8), `GB0_DTYPE` (training precision, default `float32`).
### 2.2 Key Arguments
The main arguments in the script are grouped as follows.

**Model and Distributed:**

```bash
--model-name giga_brain07            # Mapped to GigaBrain-0.7 config via config_map
--distributed-strategy fsdp          # FSDP2 distributed strategy
--dtype float32                      # Training precision (GB0_DTYPE)
--activation-checkpoint-module-patterns "$GB0_AC_PATTERNS"  # Modules with activation checkpointing
# GPUS_PER_NODE=8                    # Number of GPUs per node (script variable)
```
**Data:**

```bash
--dataset-format lerobot_datasets    # Use LeRobot datasets
--dataset-strategy giga_brain07      # GigaBrain-0.7 dataset build strategy
--dataset-path $GB0_DATA_PATH_V30    # LeRobot v3.0 dataset directory
--tokenizer-path $GB0_TOKENIZER      # PaliGemma tokenizer directory
--num-workers 8                      # Number of DataLoader workers
data.norm_stats_path=$GB0_NORM_STATS # Normalization statistics file
data.fast_tokenizer_path=$GB0_FAST_TOKENIZER  # FAST action tokenizer directory
```
**Training:**

```bash
--trainer-type FinetuneTrainer
--train-iters 100
--per-device-batch-size 80
--gradient-accumulation-steps 1
--optimizer AdamW
--lr-base 2.5e-5                     # Base learning rate
--lr-decay-style giga_warmup_cosine  # Warmup + cosine schedule matching the original GigaBrain
--lr-warmup-iters 10
--lr-decay-iters 300
--min-lr 2.5e-6
--adam-beta1 0.9
--adam-beta2 0.95
--adam-eps 1e-8
--weight-decay 1e-10
--clip-grad 0                        # No gradient clipping
model.pretrained=$GB0_PRETRAINED     # Pretrained weights
--save-interval 0                    # Checkpoint save interval; 0 disables saving
--seed 6666
```

> **Note:** `data.embodiment_id` determines the embodiment parameters used for training/inference. Make sure it matches the embodiment of the loaded checkpoint; `data.norm_stats_path` must also correspond to that embodiment's data statistics.
