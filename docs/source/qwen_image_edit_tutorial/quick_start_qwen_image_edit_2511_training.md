# Quick Start: Qwen-Image-Edit-2511 Training

This guide walks through the complete **Qwen-Image-Edit-2511 DiT pretraining** workflow.

---
## Qwen-Image-Edit-2511 Training Workflow

### 1. Resource Preparation

Before starting, download the required model weights and prepare your dataset.

#### 1.1 Download Model Weights

Download the full Qwen-Image-Edit-2511 model from HuggingFace:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir ./Qwen-Image-Edit-2511
```

The downloaded directory structure:

```text
Qwen-Image-Edit-2511/
├── text_encoder/       # Qwen2.5-VL text encoder (*.safetensors)
├── vae/                # VAE (diffusion_pytorch_model.safetensors)
├── transformer/        # DiT weights (diffusion_pytorch_model-*.safetensors)
├── tokenizer/          # Qwen2 tokenizer config & vocab
└── processor/          # Qwen2VL processor config
```

> **Note:** The preprocessing stage only requires `text_encoder/`, `vae/`, `tokenizer/`, and `processor/`. The training stage requires `transformer/` (DiT weights).

#### 1.2 Prepare Dataset

Organize your dataset directory and metadata file as follows:

```text
my_dataset/
├── metadata.json
└── edit/
    ├── image1.jpg
    ├── image2.jpg
    └── image_color.jpg
```

`metadata.json` example (each record contains a target image, edit reference image(s), and a text prompt):

```json
[
  {
    "image": "edit/image2.jpg",
    "edit_image": ["edit/image1.jpg", "edit/image_color.jpg"],
    "prompt": "Change the color of the dress in Figure 1 to the color of Figure 2"
  }
]
```

Supported metadata formats: `.json`, `.jsonl`, `.csv`.

---

### 2. Preprocess Training Data

The preprocessing stage uses the DiffSynth text encoder + VAE to encode raw images into latent caches required for training.

#### Install Dependencies

```bash
pip install diffsynth==2.0.6 --no-deps
pip install accelerate Pillow pandas tqdm
```

> **Note:** `diffsynth` must be installed with `--no-deps` to avoid pulling in a torchaudio build that is ABI-incompatible with custom torch builds. The preprocessing script internally stubs torchaudio (not needed for image pipelines).

#### Run Preprocessing

```bash
cd examples/qwen_image

QWEN_IMAGE_MODEL_ROOT=/path/to/Qwen-Image-Edit-2511 \
DATASET_BASE_PATH=/path/to/my_dataset \
DATASET_METADATA_PATH=/path/to/my_dataset/metadata.json \
bash preprocess.sh qwen-image-edit-2511 ./data/preprocessed
```

Or auto-install dependencies on first run:

```bash
INSTALL_DEPS=1 \
QWEN_IMAGE_MODEL_ROOT=/path/to/Qwen-Image-Edit-2511 \
DATASET_BASE_PATH=/path/to/my_dataset \
DATASET_METADATA_PATH=/path/to/my_dataset/metadata.json \
bash preprocess.sh qwen-image-edit-2511 ./data/preprocessed
```

Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PIXELS` | 1048576 | Max pixel budget for target image (1MP) |
| `SEED` | 1234 | Base seed for deterministic noise generation |
| `TIMESTEP_ID` | 321 | Fixed timestep index (0..999) |
| `TORCH_DTYPE` | bf16 | Inference precision (bf16/fp16/fp32) |
| `TILED_VAE` | unset | Set to any non-empty value to enable tiled VAE (low-VRAM mode) |

#### Output

Each `.pth` file contains the following keys:

- `input_latents` — Target image VAE latent
- `edit_latents` — Edit reference image VAE latents (list)
- `prompt_emb` — Text encoder embedding
- `prompt_emb_mask` — Text attention mask
- `noise` — Deterministic noise
- `timestep` / `timestep_id` — Timestep
- `latents` — Noised latent
- `training_target` — Training target (noise - input_latents)
- `scale` — Loss weight

---

### 3. Checkpoint Conversion (HF → Megatron)

Convert HuggingFace DiT weights to Megatron DCP (fsdp_dtensor) format:

```bash
cd examples/qwen_image

HF_CKPT=/path/to/Qwen-Image-Edit-2511/transformer \
MCORE_CKPT=/path/to/qwen_image_edit_2511_mcore \
bash convert_qwen_image.sh hg2mcore
```

---

### 4. Launch Training

**Recommended configuration**: FSDP + TP=2, single node with 8 GPUs

#### Configure the Training Script

Edit the key paths in `examples/qwen_image/pretrain_qwen_image_edit_2511.sh`:

```bash
MEGATRON_PATH=/path/to/AIAK-Megatron
LOONGFORGE_PATH=/path/to/AIAK-Training-Omni
DATASET_PATH=data/preprocessed                      # Output from Section 2
MCORE_LOAD_CKPT=/path/to/qwen_image_edit_2511_mcore # Output from Section 3
```

#### Launch

Single node:

```bash
bash examples/qwen_image/pretrain_qwen_image_edit_2511.sh
```

Multi-node — scale via data parallelism:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master_ip> \
bash examples/qwen_image/pretrain_qwen_image_edit_2511.sh
```

Key training parameters (overridable via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `TP_SIZE` | 2 | Tensor parallel size |
| `GLOBAL_BATCH_SIZE` | 8 | Global batch size |
| `LR` | 1e-5 | Learning rate |
| `TRAIN_ITERS` | 50000 | Training iterations |
| `RECOMPUTE_NUM_LAYERS` | 42 | Activation recomputation layers (memory control) |
| `H` / `W` | 1024 / 1024 | Training image resolution |

---

### 5. Export Weights (Megatron → HF)

After training, convert Megatron weights back to HuggingFace format:

```bash
cd examples/qwen_image

MCORE_CKPT=/path/to/trained_mcore_ckpt \
HF_OUT=/path/to/output_hf_transformer \
bash convert_qwen_image.sh mcore2hg
```

The exported weights can be used directly with the DiffSynth inference pipeline.
