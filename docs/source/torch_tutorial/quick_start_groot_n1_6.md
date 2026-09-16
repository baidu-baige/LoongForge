# Quick Start: GR00T N1.6 Training

This guide walks you through launching a **GR00T N1.6** SFT (Supervised Fine-tuning) job in the LoongForge framework.

## 0. Resource Preparation
### 0.1 Main Weights
The GR00T-N1.6 main weights are specified by `CHECKPOINT_PATH` and loaded via `--pretrained-checkpoint`. The original HuggingFace safetensors format is supported:

```bash
hf download nvidia/GR00T-N1.6-3B --local-dir /workspace/huggingface.co/nvidia/GR00T-N1.6-3B
```
### 0.2 Tokenizer / Processor
The Eagle processor/tokenizer is specified by `EAGLE_LOCAL_PATH`:

```bash
mkdir -p /workspace/huggingface.co/aravindhs-NV
huggingface-cli download aravindhs-NV/eagle3-processor-groot-n1d6 --local-dir /workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6
```
Note: `EAGLE_LOCAL_PATH` is not the main model weights path. It is only used for loading the Eagle/VLM processor, tokenizer, and related config files locally.

### 0.3 Data
The script uses LIBERO data in the standard LeRobot v3.0 format by default. If the dataset comes from a HuggingFace Dataset, download it to the same directory:

```bash
hf download lerobot/libero_10 --repo-type dataset --local-dir /workspace/libero_10
```
## 1. Data Configuration
For the standard LeRobot v3.0 format, no extra offline preprocessing is needed. GR00T's sample transform and batch collator run online during training.

Default data-related configuration:

```yaml
data:
  embodiment_tag: libero_panda
  groot_preprocess_mode: sample
  preprocess_action_horizon: 16
  preprocess_max_action_dim: 29
  preprocess_max_state_dim: 29
  image_crop_size: [224, 224]
  image_target_size: [224, 224]
  formalize_language: true
  use_relative_action: true
```
## 2. Launch Training
First, set up the paths:

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export CHECKPOINT_PATH=/workspace/huggingface.co/nvidia/GR00T-N1.6-3B
export EAGLE_LOCAL_PATH=/workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6
export DATA_PATH=/workspace/libero_10
export OUTPUT_DIR=/workspace/outputs/groot_n1_6
export TENSORBOARD_PATH=/workspace/tensorboard-log/groot_n1_6
```
### 2.1 Script Without Performance Optimizations
Use this to validate the pipeline first. It disables CUDA Graph and DDP static graph, and uses the plain `AdamW` optimizer:

```bash
bash /workspace/LoongForge/examples/vla/groot_n1_6/run_groot_n1_6_ddp_finetune.sh \
    --optimizer AdamW \
    --cuda-graph-impl none \
    --cuda-graph-pad-length 0 \
    --no-ddp-static-graph
```
### 2.2 Performance Optimizations and How to Enable Them
The default script already enables the main performance optimizations:

```bash
bash /workspace/LoongForge/examples/vla/groot_n1_6/run_groot_n1_6_ddp_finetune.sh
```
Key performance optimization switches:

* Enable the TEFusedAdamW optimizer

```python
--optimizer TEFusedAdamW
```
* Enable CUDA Graph

```python
--cuda-graph-impl local                                 # Enable CUDA Graph
--cuda-graph-scope per_microbatch                       # Capture granularity, choose between full_iteration and per_microbatch. Per-Microbatch keeps performance close to Full-Iteration while offering better loss alignment.
--cuda-graph-warmup-steps 3                             # Number of warmup steps in eager mode
--cuda-graph-pad-length 220                             # Fix the sequence length to 220 so input shapes stay constant and the CUDA Graph can be reused.
--no-cuda-graph-ddp-sync-in-graph                       # Perform DDP gradient all-reduce outside the graph
--cuda-graph-grad-sync-bucket-mb 400                    # Bucket size (MB) for gradient synchronization
--cuda-graph-grad-sync-impl coalesced                   # Synchronize gradients bucket by bucket
--cuda-graph-grad-sync-dtype bf16                       # Data type used for gradient all-reduce communication
```
### 2.3 Correctness Verification

To ensure training accuracy is not affected by the optimizations, we compared LoongForge's GR00T-N1.6 implementation with the official LeRobot implementation step by step on action loss, using identical data, weights, and training configuration. The results show that LoongForge's performance optimizations are lossless with respect to training accuracy:
![alt text](../../assets/images/precision/GR00TN1.6.png)
