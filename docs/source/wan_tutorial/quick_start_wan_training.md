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

Edit `examples/wan/convert_wan2.2.sh` (section `hg2mcore`):
- `--checkpoint_path` → source HF folder (`high_noise_model` / `low_noise_model`)
- `--save_path` → target Megatron checkpoint folder
- `--tp`, `--pp`, `--num_layers`, `--num_checkpoints` → match your conversion setup

Run from `examples/wan` because the script invokes conversion utilities with relative paths:
```bash
cd examples/wan
bash convert_wan2.2.sh hg2mcore
```

For more conversion parameters, run:
```bash
python convert_checkpoint_hg2mcore.py -h
```

---

### 3. Launch Training

**Recommended single-node split**: CP_SIZE=1 CP_ULYSSES_DEGREE=1,
Multi-node – scale by **data parallelism**:  
```text
DP = (NNODES × GPUS_PER_NODE) / CP_SIZE
CP_RING_DEGREE = CP_SIZE / CP_ULYSSES_DEGREE
```

| Symbol | Meaning |
|---|---|
| `DP` | Data Parallel degree |
| `CP_SIZE` | Context Parallel degree |
| `CP_ULYSSES_DEGREE` | Ulysses context parallel degree |
| `CP_RING_DEGREE` | Ring context parallel degree; computed as `CP_SIZE / CP_ULYSSES_DEGREE` |


**Step-1** Tune `examples/wan/pretrain_wan2.2_i2v_a14b.sh`
- `HIGH_NOISE_CHECKPOINT_PATH` → path to high-noise Megatron checkpoint (from Section 2)
- `LOW_NOISE_CHECKPOINT_PATH` → path to low-noise Megatron checkpoint (from Section 2)
- `DATASET_PATH` → output path from Section 1 (e.g. `./data/preprocessed`)
- `--context-parallel-size 4`
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

Run  
```bash
bash examples/wan/convert_wan2.2.sh mcore2hg
```

