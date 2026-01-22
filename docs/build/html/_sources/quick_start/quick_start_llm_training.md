# Quick Start: LLM Model Training

This guide shows you the fastest way to train a Large Language Model (LLM) with the AIAK-Training-Omni framework.

---

## 1. Environment Setup

### 1.1 Clone the repositories
Put both repos at the same level:

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
Download first, then convert to Megatron-Core format.

### 2.1 Download HF checkpoint
Example: **Qwen3-8B** from Hugging Face.

### 2.2 Convert to MCore
```bash
cd $AIAK_TRAINING_PATH/examples/qwen3/checkpoint_convert

# edit LOAD (HF path) and SAVE (MCore output path)
vim convert_qwen3_8b_hf_to_mcore.sh

bash convert_qwen3_8b_hf_to_mcore.sh
```

---

## 3. Prepare Training Data

### 3.1 Raw annotation
Use JSON with `instruction`, `input`, `output` fields:

```json
[
  {
    "instruction": "Identify and explain two scientific theories in the list: cell theory and heliocentric theory.",
    "input": "",
    "output": "Cell theory states that all living organisms are composed of cells..."
  },
  {
    "instruction": "Generate a slogan for three basketball teams.",
    "input": "Oklahoma City Thunder, Chicago Bulls, Brooklyn Nets",
    "output": "\"Thunder, Bulls and Nets: each shows its prowess!\""
  }
]
```

---

## 4. Launch SFT Training

Example: **Qwen3-8B** supervised fine-tuning.

### 4.1 Edit the training script
Open `examples/qwen3/finetuning/sft_qwen3_8b.sh` and update:

```bash
DATA_PATH=/path/to/your/dataset
TOKENIZER_PATH=/path/to/hf/model
CHECKPOINT_LOAD_PATH=/path/to/converted/mcore/checkpoint
CHECKPOINT_SAVE_PATH=/path/to/save/checkpoints
```

### 4.2 Run
```bash
# from examples/qwen3/finetuning/
bash sft_qwen3_8b.sh
```

### 4.3 Monitor
TensorBoard logs are written to the folder specified by `TENSORBOARD_PATH`; launch TensorBoard to watch the curves.

---

## 5. Common Issues

| Problem | Solution |
|---------|----------|
| **Module not found** | Check that `PYTHONPATH` includes both `AIAK-Training-Omni` and `AIAK-Megatron`. |
| **OOM** | Reduce `global_batch_size` or enable activation checkpointing (`--recompute-granularity full --recompute-method uniform`). |