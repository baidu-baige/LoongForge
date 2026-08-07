# Quick Start: Cosmos3-Nano Training

This guide walks you through launching an SFT (supervised fine-tuning) job for **Cosmos3-Nano** in the LoongForge framework. Example: DROID Action-Policy SFT, using FSDP2 full shard and bf16 training, aligned with the `launch_sft_action_policy_droid` recipe of the official cosmos-framework. See the corresponding open-source framework training doc: [cosmos-framework action_policy_droid_posttrain.md](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md).

## 0. Resource Preparation

Cosmos3-Nano is an action-policy model with joint vision-language modeling. Training depends on three kinds of external resources: main backbone weights (Cosmos3-Nano + Wan2.2 VAE encoder), the Qwen3-VL tokenizer / processor, and the DROID post-training dataset. Each is described in turn below.

### 0.1 Model Weights

The main weights come from `nvidia/Cosmos3-Nano` on HuggingFace. **The Cosmos3 training pipeline uses the DCP (Distributed Checkpoint) format**, so the downloaded raw HF safetensors weights need a one-off offline conversion. The conversion steps refer to the official framework doc [training.md Step 2](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md#step-2--prepare-checkpoint). The converted DCP weights directory is loaded via `--pretrained-checkpoint`:

```bash
--pretrained-checkpoint $CHECKPOINT_PATH   # Converted DCP weights directory
--init-on-meta                             # Defer weight loading until after FSDP wrap, reducing peak init memory
```

In addition to the main weights, the video branch also requires a VAE encoder. Cosmos3-Nano reuses `Wan2.2_VAE.pth` from Wan2.2-TI2V-5B; its path comes from `VAE_PATH` in the launch script:

- Main weights: [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano/tree/main)
- VAE: [Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth?download=true)

### 0.2 Tokenizer

Cosmos3-Nano's language path is based on Qwen3-VL-8B-Instruct. The tokenizer / processor reuses its HuggingFace directory directly, loaded via `--tokenizer-path`:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /workspace/ckpt/Qwen3-VL-8B-Instruct
export TOKENIZER_PATH=/workspace/ckpt/Qwen3-VL-8B-Instruct
```

### 0.3 Dataset

The example uses the officially released DROID post-training subset from NVIDIA, [nvidia/Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID/tree/main/success) (the `success/` branch, LeRobot format). Download to local:

```bash
hf download nvidia/Cosmos3-DROID --repo-type dataset --local-dir /workspace/data/Cosmos3-DROID
export DATASET_PATH=/workspace/data/Cosmos3-DROID/success
```

## 1. Data Configuration

DROID data requires no extra offline preprocessing. Frame stitching, image augmentation, and action alignment are performed online at training time via the built-in `cosmos3_droid` processing strategy. Enabled via the following paired arguments:

```bash
--dataset-format lerobot_datasets    # Dataset format: LeRobot
--dataset-strategy cosmos3_droid     # Official Cosmos3 DROID data-processing strategy
```

Remaining settings like target resolution, action chunk length, CFG dropout are already set in the data section of `configs/models/embodied/cosmos3/nano.yaml` (defaults: `target_h/target_w=480`, `action_chunk_length=32`, `action_fps=15.0`), and generally don't need overriding.

## 2. Launch Training

Launch script: `examples/embodied/cosmos3/run_cosmos3_nano_droid_fsdp_finetune.sh`. Defaults to single-node 8-GPU FSDP2 + bf16, trains for 500 steps, per-device batch=2.

### 2.1 Environment Variables

Every path below has a default in the script and can be overridden individually:

```bash
cd /workspace/LoongForge

export LOONGFORGE_PATH=/workspace/LoongForge
export TOKENIZER_PATH=/workspace/ckpt/Qwen3-VL-8B-Instruct
export CHECKPOINT_PATH=/workspace/ckpt/Cosmos3-Nano-DCP   # Converted DCP weights directory
export VAE_PATH=/workspace/ckpt/Wan2.2_VAE/Wan2.2_VAE.pth # Wan2.2 VAE encoder weights
export DATA_PATH=/workspace/data/Cosmos3-DROID/success
export OUTPUT_DIR=/workspace/outputs/cosmos3_nano_droid
```

### 2.2 Launch Script

Single-node 8-GPU FSDP2 SFT:

```bash
bash examples/embodied/cosmos3/run_cosmos3_nano_droid_fsdp_finetune.sh
```

### 2.3 Key Arguments

Arguments in the script are grouped by purpose as follows:

**Model and Distributed:**

```bash
--model-name cosmos3_nano            # Mapped to the Cosmos3-Nano DROID recipe via config_map
--distributed-strategy fsdp          # Distributed strategy: FSDP2 full shard
--dtype bfloat16                     # Training precision: bf16
--fsdp-reduce-dtype bf16             # Reduce-scatter gradients in bf16
--fsdp-wrap-modules MoTDecoderLayer  # Wrap one FSDP group per MoT decoder layer
--init-on-meta                       # Reduce peak init memory via meta device
```

**Data:**

```bash
--dataset-format lerobot_datasets    # Dataset format: LeRobot
--dataset-strategy cosmos3_droid     # Official DROID data-processing strategy (frame stitching, image augmentation, etc.)
--dataset-path $DATA_PATH            # DROID data directory
--tokenizer-path $TOKENIZER_PATH     # Qwen3-VL tokenizer / processor directory
--num-workers 4                      # DataLoader worker count
```

**Training and Optimizer:**

```bash
--trainer-type FinetuneTrainer
--train-iters 500                    # Total training steps
--per-device-batch-size 2            # Per-device batch
--gradient-accumulation-steps 1
--disable-tf32                       # Disable TF32 to match numerical precision of the reference implementation
--pretrained-checkpoint $CHECKPOINT_PATH
--save-interval 0                    # 0 = never checkpoint; set a positive value to save
--seed 42
--set-seed-by-rank                   # Offset the seed per rank
```

**Parameter-group LR and Optimizer:**

Cosmos3-Nano's action head (`action2llm` / `llm2action` / `action_modality_embed`) requires a higher learning rate than the vision-language backbone, so the script uses `--lr-group` to group by parameter-name prefix:

```bash
--lr-group net.action2llm=1e-3,net.llm2action=1e-3,net.action_modality_embed=1e-3,net=2e-4
                                     # Action heads at 1e-3, other net params at 2e-4
--lr-decay-style lambda_linear       # Learning-rate decay: linear
--lr-warmup-iters 0
--optimizer TorchFusedAdamW          # Optimizer: fused AdamW
--clip-grad 1.0
--weight-decay 0.05
--adam-beta1 0.9
--adam-beta2 0.99
--adam-eps 1e-8
```

To adjust for actual training scale, the common approach is to:

- Override `--train-iters` and `--save-interval` to control training duration and checkpoint frequency
- Override `--per-device-batch-size` and `--gradient-accumulation-steps` to adjust the global batch
- Override the `net=` entry in `--lr-group` to tune the backbone LR

### 2.4 Performance Switches

Data switches are set in the launch script, or appended on the command line:

```bash
# Run the augmentation tail on the model device instead of the dataloader worker
bash examples/embodied/cosmos3/run_cosmos3_nano_droid_fsdp_finetune.sh data.colorjitter_on_gpu=true
```

- `data.colorjitter_on_gpu` (default `false`) — when on, the worker only decodes video and ColorJitter plus everything after it runs on the GPU.
- `--disable-tf32` — set by the script to match the reference implementation; drop it when optimizing for throughput.
- `--per-device-batch-size` / `--num-workers` / `--gradient-accumulation-steps` — the usual throughput knobs.

### 2.5 Correctness Verification

With identical data, weights and training configuration, the loss was compared step by step against the official cosmos-framework `launch_sft_action_policy_droid` recipe: over 26 steps the loss is **bit-identical (max |Δ| = 0)**.

![cosmos3-nano loss alignment](../../assets/images/precision/cosmos3.png)

To reproduce the comparison, keep the following arguments consistent with the reference:

```bash
--seed 42                            # same seed
--set-seed-by-rank                   # per-rank seed offset, matching the reference behaviour
--dtype bfloat16                     # bf16 training precision
--fsdp-reduce-dtype bf16             # gradient reduce-scatter precision
--fsdp-wrap-modules MoTDecoderLayer  # FSDP grouping granularity
--disable-tf32                       # TF32 off
--deterministic-mode                 # deterministic kernels; also export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`GPUS_PER_NODE`, `--per-device-batch-size` and `--gradient-accumulation-steps` determine the global batch and sampler sharding — do not change them midway through an alignment experiment.
