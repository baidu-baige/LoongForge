# Quick Start: VLM Model Training  
This guide walks you through the fastest way to train a Vision-Language Model (VLM) with the AIAK-Training-Omni framework.

---

## 1. Environment Setup

### 1.1 Clone the repositories
Make sure you have both this repo and its dependency AIAK-Megatron at the same level:

```bash
# assume we are in /workspace
git clone <git_repo_url>/AIAK-Training-Omni.git
git clone <git_repo_url>/AIAK-Megatron.git
```

### 1.2 Install dependencies
```bash
cd AIAK-Training-Omni
pip install -r requirements.txt
```

Export environment variables:
```bash
export AIAK_TRAINING_PATH=/workspace/AIAK-Training-Omni
export MEGATRON_PATH=/workspace/AIAK-Megatron
export PYTHONPATH=$MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH
```

---

## 2. Prepare Model Weights

Training starts from open-source Hugging-Face checkpoints.  
You need to download them first and then convert to Megatron-Core format.

### 2.1 Download HF checkpoint
Take **QwenVL2.5-7B** as an example—download it from Hugging Face.

### 2.2 Convert to MCore
Use the conversion script inside `examples/`:

```bash
cd $AIAK_TRAINING_PATH/examples/qwen2.5_vl/checkpoint_convert

# edit LOAD (HF path) and SAVE (MCore output path)
vim convert_qwen2.5_vl_7b_mcore_to_hf.sh

bash convert_qwen2.5_vl_7b_mcore_to_hf.sh
```

---

## 3. Prepare Training Data

### 3.1 Raw annotation
We recommend JSON with image paths and multi-turn conversations:

```json
{
  "id": 64,
  "image": [
    "mmdu-45k_pics/Bk1Gh1Y1S2EChGZzde_1_1.jpg",
    "mmdu-45k_pics/Bk1GhVk1RhkB7W80RCTo_1.jpg"
  ],
  "conversations": [
    { "from": "human", "value": "Image1: <image>\nImage2: <image>\nQ: ..." },
    { "from": "gpt",   "value": "The architectural styles observed ..." }
  ]
}
```

### 3.2 Convert to WebDataset
* Tool: `tools/data_preprocess/vlm/convert_to_webdataset.py`  
* See the [VLM data-preprocessing doc](../usage/vlm_dataset_conversion.md) for details.

After conversion the `output_dir` path is what you will assign to `DATA_PATH` in the training script.

---

## 4. Launch SFT Training

We use **Qwen2.5-VL-7B** supervised fine-tuning as an example.

### 4.1 Edit the training script
Open `examples/qwen2.5_vl/sft/sft_qwen2_5_vl_7b.sh` and update:

```bash
DATA_PATH=/path/to/your/dataset
TOKENIZER_PATH=/path/to/hf/model
CHECKPOINT_LOAD_PATH=/path/to/converted/mcore/checkpoint
CHECKPOINT_SAVE_PATH=/path/to/save/checkpoints
```

### 4.2 Run
```bash
# from examples/internvl2.5/
bash sft_internvl2_5_8b.sh
```

### 4.3 Monitor
TensorBoard logs are written to the directory specified by `TENSORBOARD_PATH`; open TensorBoard to watch the curves.

---

## 5. Common Issues

| Problem | Solution |
|---------|----------|
| **Module not found** | Check that `PYTHONPATH` includes both `AIAK-Training-Omni` and `AIAK-Megatron`. |
| **OOM** | Reduce `global_batch_size` or enable activation checkpointing (`--recompute-granularity full --recompute-method uniform`). |