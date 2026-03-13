# Quick Start: Wan Model Training  
This section walks through the **Wan Pre-train** pipeline end-to-end.

---
## Quick Start Wan2.2 Model Training
### 1. Data Pre-processing
#### Background  
- DiffSynth-Studio community produces training data **online**.  
- AIAK expects **offline** pre-processed data.  
→ Apply the provided patch `launch_data_process_v118.patch` to community tag `v1.1.8` to enable offline generation.

#### Expected data layout  
Same as the DiffSynth-Studio community format:

```
data/example_video_dataset/
├── metadata.csv
├── video1.mp4
└── video2.mp4
```

**metadata.csv example**  
```
video,prompt
video1.mp4,"from sunset to night, a small town, light, house, river"
video2.mp4,"a dog is running"
```

#### Steps  
**Step-1** Clone DiffSynth-Studio and switch to v1.1.8  
```bash
git checkout -b v1118 v1.1.8
```

**Step-2** Copy `OmniTraining/examples/wan/patch/launch_data_process_v118.patch` into the DiffSynth-Studio repo root.

**Step-3** Apply the patch  
```bash
git apply launch_data_process_v118.patch
```

**Step-4** Edit `examples/wanvideo/model_training/full/Wan2.2-I2V-A14B-DataProcess.sh`  
- `--dataset_base_path` → `<base_path>/data/example_video_dataset`  
- `--dataset_metadata_path` → same folder + `/metadata.csv`  
- `--output_path` → `<base_path>/data_preprocess/Wan2.2-I2V-A14B_full`  
- `--model_paths` → point to **your Hugging-Face high-/low-noise Wan2.2 weights**

**Step-5** Run (8 GPUs recommended – no DeepSpeed in this stage, OOM risk on smaller setups)  
```bash
bash examples/wanvideo/model_training/full/Wan2.2-I2V-A14B-DataProcess.sh
```

#### Output  
One `.tensors.pth` file per video containing three keys:  
- `input_latents` – VAE latent of the whole video  
- `y` – VAE latent of the **first frame**  
- `context` – text encoder embedding  

(High-/low-noise tensors are **NOT** separated; AIAK adds noise online later.)

---

### 2. Checkpoint Conversion (HF → Megatron)

Inside **OmniTraining** repo:

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
```
dp = (NNODES × GPUS_PER_NODE) / (pp × cp)
```

**Step-1** Tune `examples/wan/pretrain_wan2.2_i2v_a14b.sh`  
- `HIGH_NOISE_CHECKPOINT_PATH` → `<base>/hg2mcore_pp4/high_noise/Megatron_Release/`  
- `LOW_NOISE_CHECKPOINT_PATH` → analogous  
- `METADATA_PATH` → original `metadata.csv`  
- `DATASET_BASE_PATH` → `<base>/data_preprocess/Wan2.2-I2V-A14B_full`  
- `--context-parallel-size 2`  
- `--context-parallel-ulysses-degree 2`  
- `--pipeline-model-parallel-size 4`

**Step-2** Start  
- Single-node:  
  ```bash
  bash examples/wan/pretrain_wan2.2_i2v_a14b.sh
  ```
- Multi-node: execute the same script on every node – cluster env-vars (`MASTER_ADDR`, `NODE_RANK` …) are picked up automatically.

---

### 4. Checkpoint Export (Megatron → HF)

Edit `examples/wan/convert_wan2.2.sh` (section `mcore2hg`):  
- `--load_path` → Megatron checkpoint after training  
- `--save_path` → target HF folder  
- `--checkpoint_path` → dummy HF path (structure only)  
- `--pp 4`  

Run  
```bash
bash examples/wan/convert_wan2.2.sh mcore2hg
```

---

### 5. Argument Reference

Full argument list → see `omni_training/train/arguments.py`.
