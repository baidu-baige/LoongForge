# Quick Start: FastWAM Training

This guide walks you through launching a **FastWAM** SFT (Supervised Fine-Tuning) job in the LoongForge framework.

## 0. Resource Preparation

### 0.1 Model Weights

#### 0.1.1 Extract the Action DiT Backbone

Extract and linearly interpolate the Action DiT head weights from the Wan2.2 video DiT checkpoint, producing a `.pt` file that training can load directly:

```bash
LOCAL_MODEL_PATH=/data/models \
OUTPUT=$LOONGFORGE_PATH/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    bash examples/embodied/fastwam/preprocess_action_dit_backbone.sh
```

#### 0.1.2 Wan-AI/Wan2.2-TI2V-5B Weights

```bash
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir /workspace/huggingface.co/Wan-AI/Wan2.2-TI2V-5B
```

### 0.2 Tokenizer / Processor

Encode all task instructions in the dataset into text embeds in one pass, then read them on demand during training:

```bash
DATASET_PATH=/data/libero \
TEXT_EMBEDDING_CACHE_DIR=/data/cache/fastwam_text_embeds \
    bash examples/embodied/fastwam/precompute_text_embeds.sh
```

### 0.3 Dataset

```bash
hf download yuanty/LIBERO-fastwam --local-dir /workspace/data/LIBERO-fastwam
cd /workspace/data/LIBERO-fastwam
tar -zxvf libero_spatial_no_noops_lerobot.tar.gz
```

## 1. Data Configuration

Data-related fields are configured mainly through YAML and environment variables.

**Model and data YAML:**

```yaml
model:
  action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt  # Path to the action dit generated above
  redirect_common_files: true
  dtype: bfloat16

data:
  text_embedding_cache_dir: data/text_embeds_cache/libero  # Path to the text embeds generated above
```

**Environment variables:**

```bash
export LOONGFORGE_PATH=/workspace/LoongForge                                                    # Repository root
export DATASET_PATH=/workspace/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot              # LeRobot dataset root
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/huggingface.co/Wan-AI/Wan2.2-TI2V-5B                # Wan2.2 5B base model path
```

## 2. Launch Training

### 2.1 Launch Script

Single-node DDP training example:

```bash
bash examples/embodied/fastwam/run_fastwam_sft_ddp_finetune.sh
```

### 2.2 Correctness Verification

To ensure that training accuracy is not affected by our optimizations, we ran a step-by-step action loss comparison between LoongForge's FastWAM and the official implementation under identical data, weights, and training configurations. The results show that LoongForge's performance optimizations are lossless with respect to training accuracy:
![alt text](../../assets/images/precision/fastwam.png)
