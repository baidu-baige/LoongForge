# Mcore-Bridge: Online HF Checkpoint Loading & Saving

This module provides online loading and saving of HuggingFace (HF) format checkpoints. It allows you to directly read HF model weights at training startup without any offline conversion, and optionally export weights back to HF format after training.

> **Credits**: This module is inspired by [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

### What is Bridge?

In the standard workflow, training requires a separate offline conversion step (`HF → Megatron`) before training can begin. Bridge eliminates this step — it detects HF safetensors in your checkpoint directory and converts them to Megatron format **on the fly** during startup. This saves you:

- **Time** — no waiting for offline conversion jobs to finish
- **Storage** — no need to keep both HF and Megatron copies of the same weights
- **Complexity** — fewer scripts to run and fewer paths to manage

## Supported Models

Theoretically, any model supported by offline conversion is also supported online. The following models have been tested and verified:

| Type | Models |
|------|--------|
| **LLM** | LLaMA 3/3.1, Qwen 2.5/3, DeepSeek V2 Lite |
| **VLM** | Qwen2.5-VL, InternVL 2.5/3.5, LLaVA-OV 1.5 |

## Supported Parallelism Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| TP | `--tensor-model-parallel-size` | Tensor Parallelism |
| PP | `--pipeline-model-parallel-size` | Pipeline Parallelism |
| Custom Pipeline | `--custom-pipeline-layers` | Custom pipeline layer distribution |
| EP | `--expert-model-parallel-size` | Expert Parallelism (MoE) |
| ETP | `--expert-tensor-parallel-size` | Expert Tensor Parallelism |
| VPP | `--num-virtual-stages-per-pipeline-rank` | Virtual Pipeline Parallelism |
| Heterogeneous TP | `--encoder-tensor-model-parallel-size` | Different TP for Encoder and Decoder (VLM only) |

## Quick Start

### Prerequisites

Ensure the following environment variables are set and **exported** (not just locally assigned):

```bash
export MEGATRON_PATH=/path/to/Loong-Megatron 
export LOONGFORGE_PATH=/path/to/LoongForge  # This repository
export CHECKPOINT_PATH=/path/to/Qwen2.5-7B-Instruct  # HF model directory containing config.json, model*.safetensors, tokenizer files
```

> **Important**: `LOONGFORGE_PATH` **must** be exported. The model YAML configs rely on this variable to resolve paths at runtime via Hydra's `oc.env` resolver, for example:
>
> ```yaml
> hydra:
>   searchpath:
>     - file://${oc.env:LOONGFORGE_PATH}/configs/models/
>
> convert_file: ${oc.env:LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/internvl_vit_0.3b_convert.yaml
> ```
>
> If `LOONGFORGE_PATH` is not exported, config resolution will fail with a `KeyError` or similar error at startup.

### Step 1: Verify HF Model Directory

Ensure your checkpoint directory contains standard HF weight files:

```
$CHECKPOINT_PATH/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-xxxxx.safetensors
└── tokenizer files...
```

> **Tip**: If you want to keep the original HF directory clean, point `--load` to the HF directory and `--save` to a separate output directory. Bridge will load from `--load` and write MCore checkpoints to `--save`. However, when resuming training, you will need to re-point `--load` to the MCore checkpoint directory.

### Step 2: Write Training Script

Point `--load` to the HF model directory and `--save` to where you want MCore checkpoints stored (can be the same path). A full example is available at `examples/qwen2.5/pretrain/pretrain_qwen2.5_7b_bridge.sh`.

Key parameters:

```bash
TRAINING_ARGS=(
    --load $CHECKPOINT_PATH     # Path to HF model directory
    --save $CHECKPOINT_PATH     # MCore checkpoint output (can differ from --load)
    --save-interval 40          # Save MCore checkpoint every 40 steps
    --save-hf true              # Optional: export HF weights after training
    --save-hf-path /path/to/output  # Optional: default is <save>/release_hf_weights/
    ...
)
```

### Step 3: Launch Training

```bash
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun --nproc_per_node 8 --nnodes 1 \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_ARGS[@]} ${DATA_ARGS[@]} ${TRAINING_ARGS[@]} ...
```

- **First run**: No `latest_checkpointed_iteration.txt` in the directory. Bridge automatically loads HF weights and converts them to MCore format on the fly.
- **Subsequent runs**: `latest_checkpointed_iteration.txt` exists. System automatically loads the latest MCore checkpoint (resumes training).

## Loading Mechanism

The system detects whether to load HF weights or MCore shards based on the presence of `latest_checkpointed_iteration.txt`:

| Directory State | Behavior |
|----------------|----------|
| No `latest_checkpointed_iteration.txt`, has HF weights | **Bridge online loading** — converts HF safetensors to MCore format |
| Has `latest_checkpointed_iteration.txt`, has MCore shards | Loads MCore shards (standard behavior) |
| Both exist | **MCore shards take priority** — HF weights are not reloaded |

### Directory Structure During Training

```
# Before first run (pure HF directory)
$CHECKPOINT_PATH/
├── config.json
├── model.safetensors.index.json
└── model-00001-of-xxxxx.safetensors

# After training saves a checkpoint
$CHECKPOINT_PATH/
├── latest_checkpointed_iteration.txt    # auto-generated
├── config.json                          # original HF files preserved
├── model.safetensors.index.json         # original HF files preserved
├── model-00001-of-xxxxx.safetensors     # original HF files preserved
└── iter_0000040/                        # new MCore checkpoint

# Resume training: run the same script again
# → System detects latest_checkpointed_iteration.txt, resumes from iter_0000040/
```

## Saving Mechanism

### MCore Checkpoint Saving (Default)

MCore checkpoints are saved every `--save-interval` steps. This behavior is identical to the standard training workflow.

### HF Weight Export (Optional)

Controlled by the `--save-hf` flag. Exports weights to HF format after training completes.

```bash
TRAINING_ARGS=(
    --save $CHECKPOINT_PATH         # MCore checkpoint path
    --save-hf true                  # Enable HF export
    --save-hf-path /path/to/output  # Optional: default is <save>/release_hf_weights/
)
```

Resulting directory structure:

```
$CHECKPOINT_PATH/
├── latest_checkpointed_iteration.txt
├── iter_xxxx/                    # MCore checkpoints
├── model-00001-of-xxxxx.safetensors  # original HF weights (if any)
└── release_hf_weights/           # exported HF weights after training
    ├── model.safetensors.index.json
    └── model-00001-of-xxxxx.safetensors
```

> Note: HF export saves model weights only; optimizer states are not included.

## Checkpoint Resumption

Checkpoint resumption is fully automatic — just run the same training script again. No additional parameters are needed.

```
Detect latest_checkpointed_iteration.txt
  → Read latest iteration (e.g., 40)
  → Load model weights, optimizer state, RNG state from iter_0000040/
  → Continue training from iteration 41
```

### Important Notes

- Parallelism configuration (TP, PP, EP, etc.) must remain consistent with the interrupted run
- `--train-iters` should be set to the target total iterations; the system automatically continues from where it left off
- Optional flags:
  - `--ckpt-step 80` — resume from a specific iteration (instead of the latest)
  - `--no-load-optim` — skip optimizer state loading
  - `--no-load-rng` — skip RNG state loading

## VLM Heterogeneous TP Configuration

For Vision-Language Models (VLMs), the Encoder and Decoder can use different TP sizes.

**Step 1**: Configure Encoder TP in the model YAML:

```yaml
# configs/models/<model>/<model>.yaml
model:
  image_encoder:
    tensor_model_parallel_size: 2    # encoder TP = 2
```

**Step 2**: Configure Decoder TP in the training script:

```bash
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 4            # decoder TP = 4
    --encoder-tensor-model-parallel-size 2    # encoder TP = 2
    --pipeline-model-parallel-size 2
)
```

## End-to-End Flow Summary

```
HF Model Directory (safetensors)
  │
  │  First run → bridge online loading, auto-convert to MCore format
  ▼
Training Loop
  │
  │  Every --save-interval steps → save MCore checkpoint
  │  Auto-generate latest_checkpointed_iteration.txt
  ▼
$CHECKPOINT_PATH/iter_XXXXXXX/ (MCore format)
  │
  ├── Interrupted & restart → auto-load latest MCore shard (resume)
  │
  ├── Training complete → [--save-hf true] export HF weights to release_hf_weights/
  ▼
HF Model (safetensors)
```

## FAQ

**Q: Can I keep the original HF model directory untouched?**

Yes. Point `--load` to the original HF directory and `--save` to a new directory. Bridge will load HF weights from `--load` and write MCore checkpoints to `--save`. However, when resuming training, you will need to re-point `--load` to the MCore checkpoint directory.

**Q: Does Bridge support SFT (supervised fine-tuning)?**

Yes. Set `--training-phase sft` in your training arguments. The loading and saving behavior is the same as pretrain.

**Q: Can I use a different TP/PP configuration when resuming training?**

No. Parallelism configuration (TP, PP, EP, etc.) must remain consistent with the interrupted run. Changing parallelism requires re-loading from HF weights.

## Roundtrip Test

### Overview

The Roundtrip Test verifies that HF weights remain numerically consistent after a full `HF → MCore → HF` round-trip conversion. It reuses the exact same model building, loading, and saving pipeline as real training, but executes **zero training steps**. The result directly reflects bridge conversion correctness.

### Test Flow

```
1. initialize_loongforge_megatron    # Initialize Megatron environment (same as training)
       ↓
2. get_model()                  # Build model (same as training)
       ↓
3. load_hf_checkpoint_online()  # Online load original HF checkpoint
       ↓
4. save_hf_checkpoint_online()  # Save back to HF format (roundtrip output)
       ↓
5. compare_weights()            # Compare original vs. roundtrip weights (rank 0 only)
```

### Comparison Criteria

| Level | Tolerance | Meaning |
|-------|-----------|---------|
| **Exact** | `rtol=1e-4, atol=1e-6` | Exact match |
| **Close** | `rtol=1e-3, atol=1e-4` | Approximate match (typical bfloat16 precision difference) |
| **Diff** | Exceeds above tolerance | Mismatch — investigation needed |

**Pass condition**: `num_diff == 0` with no missing keys, extra keys, or shape mismatches.

### Running a Single Model Test

Example with Qwen2.5-0.5B:

```bash
bash tools/dist_checkpoint/test/qwen2.5/0.5b_bridge_roundtrip.sh
```

Key parameters in the test script:

```bash
TRAINING_ARGS=(
    --training-phase pretrain
    --train-iters 0              # No training — load + save only
    --no-load-optim              # Skip optimizer state
    --no-load-rng                # Skip RNG state
    --load $TOKENIZER_PATH       # Original HF checkpoint directory
    --save-hf-path $SAVE_HF_PATH # Roundtrip output directory
    --save-hf true                # Enable HF export
    --bf16                       # Use bf16 precision
)
```

> Note: The entry point is `hf_roundtrip_test.py`, **not** `loongforge/train.py`:
>
> ```bash
> PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
>     torchrun --nproc_per_node 4 \
>     $LOONGFORGE_PATH/tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py \
>     ${MODEL_ARGS[@]} ${TOKENIZER_ARGS[@]} ${TRAINING_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]}
> ```

### Running All Tests for a Model Family

```bash
# Qwen2.5 (all sizes)
bash tools/dist_checkpoint/test/qwen2.5/all.sh

# Qwen3 (all sizes)
bash tools/dist_checkpoint/test/qwen3/all.sh

# InternVL 2.5 (all sizes)
bash tools/dist_checkpoint/test/internvl2.5/all.sh
```

### Output Report

After the test completes, a `roundtrip_comparison.json` is generated in the `--save-hf-path` directory:

```json
{
  "passed": true,
  "num_baseline": 361,
  "num_roundtrip": 361,
  "num_common": 361,
  "missing_keys": [],
  "extra_keys": [],
  "shape_mismatches": [],
  "num_exact_matches": 361,
  "num_close_matches": 0,
  "num_different": 0,
  "mismatched_keys": [],
  "max_abs_diff": 0.0,
  "mean_abs_diff": 0.0
}
```

### Available Test Scripts

Tests are organized by model family under `tools/dist_checkpoint/test/`:

| Model Family | Path |
|-------------|------|
| Qwen 2.5 | `test/qwen2.5/` |
| Qwen 3 | `test/qwen3/` |
| Qwen 2.5-VL | `test/qwen2.5vl/` |
| DeepSeek V2 | `test/deepseek2/` |
| DeepSeek V3 | `test/deepseek3/` |
| LLaMA 3 | `test/llama3/` |
| LLaMA 3.1 | `test/llama3.1/` |
| InternVL 2.5 | `test/internvl2.5/` |
| InternVL 3.5 | `test/internvl3.5/` |
| LLaVA-OV 1.5 | `test/llavaov1.5/` |

### Model Provider Selection

The test code defaults to `omni_model_provider` (compatible with all models). To switch:

- **Pure LLM** (LLaMA, Qwen2.5, DeepSeek V2, etc.): use `llm_model_provider`
- **Multimodal** (Qwen2.5-VL, InternVL, etc.): use `omni_model_provider`

Edit the `get_model()` call in `tools/dist_checkpoint/checkpoint/hf_roundtrip_test.py`:

```python
# For pure LLM:
model = get_model(llm_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

# For multimodal (default):
model = get_model(omni_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
```

### Important Notes

- Roundtrip tests do **not** require training data — only an HF checkpoint directory and tokenizer path
- GPU count must satisfy the parallelism configuration (TP x PP <= available GPUs)
- A cProfile performance report (`profile_stats.prof`) is generated for diagnosing load/save bottlenecks
- If weight mismatches are found, check the `[DIFF]` entries in the log for tensor names and diff values to identify conversion issues

## Example Scripts

| Model | Path |
|-------|------|
| Qwen2.5 7B | `examples/qwen2.5/pretrain/pretrain_qwen2.5_7b_bridge.sh` |
| DeepSeek V2 Lite | `examples/deepseek_v2/pretrain/pretrain_deepseek_v2_lite_group_bridge.sh` |
