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

---

## Quick Start Wan2.1 Model Training  

### 1. Data Pre-processing
Wan2.1 also uses **offline** pre-processing.

**Raw data**: jsonl + video  
Example json line  
```json
{"video_path": "/…/EGO_9.mp4",
 "dense_lang": "In our architecture, spatial attention …"}
```

**Steps**  
1. Clone & checkout commit  
   ```bash
   git clone https://github.com/modelscope/DiffSynth-Studio.git
   git checkout ef2a7abad478d6f91c0839be8f491587e36fda9f
   ```
2. Copy `omni/examples/wan/wan2.1_preprocess.py` → `examples/wanvideo/`
3. Run  
   ```bash
   CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/wan2.1_preprocess.py \
     --task data_process \
     --dataset_path fake_dataset_i2v/train \
     --output_path ./models \
     --text_encoder_path …/models_t5_umt5-xxl-enc-bf16.pth \
     --vae_path …/Wan2.1_VAE.pth \
     --tiled \
     --num_frames 81 \
     --height 480 \
     --width 832
   ```

**Output**  
`<video>.mp4.tensors.pth` alongside each video.

---

### 2. Checkpoint Conversion (HF → Megatron)

```bash
bash examples/wan/convert_wan2.1.sh hg2mcore
```

Key parameters inside script  
- `--load_path` – scaffold random checkpoint (create with short run, `--save-interval 2`)  
- `--save_path` – destination for converted Megatron weights  
- `--checkpoint_path` – HF weights folder  
- `--num_checkpoints 7` – #.safetensors files  
- `--tp 1` – tensor-parallel, fixed to 1 for Wan  
- `--pp 4` – pipeline stages  
- `--num_layers 40`

---

### 3. Training Launch

Example single-node command (≥1 node, 8 GPUs):

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=...
AIAK_TRAINING_PATH=...
METADATA_PATH=...
DATASET_BASE_PATH=...
CHECKPOINT_PATH=...
TENSORBOARD_PATH=...

GPUS_PER_NODE=8
MASTER_ADDR=...
MASTER_PORT=6008
NNODES=1
NODE_RANK=0

torchrun --nproc_per_node $GPUS_PER_NODE \
         --nnodes $NNODES --node_rank $NODE_RANK \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  $AIAK_TRAINING_PATH/omni_training/train.py \
  --model-name wan2-1-i2v \
  --tokenizer-type NullTokenizer --vocab-size 0 \
  --data-path $METADATA_PATH --dataset-base-path $DATASET_BASE_PATH \
  --dataloader-type external \
  --num-latent-frames 81 --max-latent-height 480 --max-latent-width 832 \
  --micro-batch-size 1 --global-batch-size 16 \
  --train-iters 50000 --lr 0.0001 --bf16 \
  --load $CHECKPOINT_PATH --save $CHECKPOINT_PATH --save-interval 5000000 \
  --context-parallel-size 4 --context-parallel-ulysses-degree 4 \
  --pipeline-model-parallel-size 2 \
  --tensorboard-dir $TENSORBOARD_PATH
```

**Parallel tuning**:  
Long sequences → attention dominates. Use higher CP (`--context-parallel-ulysses-degree`) to speed up.

---

### 4. Checkpoint Export (Megatron → HF)

```bash
bash examples/wan/convert_wan2.1.sh mcore2hg
```
Same parameter meanings as `hg2mcore`; `--num_layers 40` required.

---

### 5. Argument Reference

See `omni_training/train/arguments.py`.  
Extra Wan2.1 flags:  
- `--max-text-length`  
- `--max-image-length` (fixed after pre-processing, rarely changed)
