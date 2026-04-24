# Quick Start: Wan Model Training  
This section walks through the **Wan Pre-train** pipeline end-to-end.

---
## Wan2.2 I2V-A14B Training Pipeline
### 1. Preprocess Training Data
#### Expected dataset example
```text
dataset
├── metadata.csv
└── train
    ├── EGO_1.mp4
    ├── EGO_2.mp4
    ├── EGO_3.mp4
```
metadata.csv example

```csv
video,prompt
train/EGO_1.mp4,"places the bag of clothes on the floor\nPlan:\n pick up the bag of clothes. Put the bag of clothes on the floor.\nactions :\n1. pick up(bag of clothes)\n2. put on(bag of clothes, floor)"
```

#### Steps

**Step-1** Install dependencies and download model weights

```bash
pip install diffsynth==1.1.8
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan-AI/Wan2.2-I2V-A14B
```

**Step-2** Process the input

```bash
MODEL_BASE=./Wan-AI/Wan2.2-I2V-A14B  # should match --local-dir in Step-1
MODEL_T5=${MODEL_BASE}/models_t5_umt5-xxl-enc-bf16.pth
MODEL_VAE=${MODEL_BASE}/Wan2.1_VAE.pth
# Script location: examples/wan/wan_preprocess.py in LoongForge repo
accelerate launch wan_preprocess.py \
  --dataset_base_path <your_dataset> \
  --dataset_metadata_path <your_dataset>/metadata.csv \
  --height 480 --width 832 --num_frames 49 \
  --model_paths "${MODEL_T5},${MODEL_VAE}" \
  --tokenizer_local_path "${MODEL_BASE}/google/umt5-xxl" \
  --output_path ./data/preprocessed \
  --max_timestep_boundary 0.358 --min_timestep_boundary 0

```

#### Output
Each `.pth` file contains the following three keys:
- `input_latents` – VAE latent of the whole video
- `y` – first-frame VAE latent concatenated with a visibility mask
- `context` – text encoder embedding

(High-/low-noise tensors are **NOT** separated; LoongForge adds noise online later.)

---

### 2. Convert Checkpoints (HF → Megatron)

Inside **LoongForge** repo:

**Step-1** Generate **random Megatron checkpoints** with correct PP split (needed as scaffold).  
- Pick an empty folder, e.g. `<base>/wan2.2/hg2mcore_pp4/high_noise/Megatron_Random`  
- In `examples/wan/pretrain_wan2.2_i2v_a14b.sh` set  
  - `HIGH_NOISE_CHECKPOINT_PATH` → above folder  
  - `LOW_NOISE_CHECKPOINT_PATH` → analogous  
  - `--train-iters 5`  
  - `--save-interval 2`  
- Run once – you will obtain `iter_0000002` folders.

**Step-2** Convert HF weights into Megatron format  
Edit `examples/wan/convert_wan2.2.sh` (section `hg2mcore`):  
- `--load_path` → `iter_0000002` produced in Step-1  
- `--save_path` → final release folder, e.g. `<base>/high_noise/Megatron_Release/`  
- `--checkpoint_path` → original HF `.safetensors` directory  
- `--pp 4` (or 8)  

Run  
```bash
bash examples/wan/convert_wan2.2.sh hg2mcore
```
Repeat for low-noise model.

---

### 3. Launch Training

**Recommended single-node split**: PP=4, CP=2  
Multi-node – scale by **data parallelism**:  
```text
dp = (NNODES × GPUS_PER_NODE) / (pp × cp)
```

| Symbol | Meaning |
|---|---|
| `dp` | Data Parallel degree |
| `pp` | Pipeline Parallel degree |
| `cp` | Context Parallel degree |

**Step-1** Tune `examples/wan/pretrain_wan2.2_i2v_a14b.sh`
- `HIGH_NOISE_CHECKPOINT_PATH` → path to high-noise Megatron checkpoint (from Section 2)
- `LOW_NOISE_CHECKPOINT_PATH` → path to low-noise Megatron checkpoint (from Section 2)
- `DATASET_PATH` → output path from Section 1 (e.g. `./data/preprocessed`)
- `--pipeline-model-parallel-size 4`
- `--context-parallel-size 2`
- `--context-parallel-ulysses-degree 2`

**Step-2** Start  
- Single-node:  
  ```bash
  bash examples/wan/pretrain_wan2.2_i2v_a14b.sh
  ```
- Multi-node: execute the same script on every node – cluster env-vars (`MASTER_ADDR`, `NODE_RANK` …) are picked up automatically.

---

### 4. Export Checkpoints (Megatron → HF)

Edit `examples/wan/convert_wan2.2.sh` (section `mcore2hg`):  
- `--load_path` → Megatron checkpoint after training  
- `--save_path` → target HF folder  
- `--checkpoint_path` → original HF checkpoint directory (used for reading model structure only)
- `--pp 4`  

Run  
```bash
bash examples/wan/convert_wan2.2.sh mcore2hg
```
