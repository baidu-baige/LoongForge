# Quick Start: Pi0.5 Training

This document will guide you through the quick start process for **Pi0.5** SFT training under the LoongForge framework.

## 1. Data Preparation

### 1.1 Dataset Format

VLA training uses robot manipulation trajectory data. Each sample typically contains multimodal observations (images, language instructions) and action sequences. LoongForge expects data to be organized in the **[LeRobot dataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)** format, where the path passed via `--data-path` points to the root of your dataset.

LeRobot dataset v3.0 stores episodes as Parquet files with standardized fields for observations, actions, and metadata. The directory structure typically looks like:

### 1.2 Dataset Parameter Description

* `--data-path`: Dataset root directory path.
* `--chat-template empty`: VLA models use the `empty` chat template, as action prediction does not rely on conversational prompt templates.
* `--split 100,0,0`: Ratio split for train/validation/test. Typically all data is used for training.
* `--num-workers`: Number of data loading worker processes (default 16).


## 2. Model Weight Preparation

### 2.2 Convert HF Weights to torch Format

Convert HuggingFace weights to PyTorch format for LoongForge training:

```bash
# Set input/output paths
export LOAD=/path/to/pi05_huggingface/
export SAVE=/path/to/pi05_torch/

sh examples/pi05/checkpoint_convert/convert_pi05_hf_to_torch.sh
```

After conversion, the checkpoint directory structure will be:

```
pi05_torch/
├── latest_checkpointed_iteration.txt
└── release
    └── mp_rank_00
        └── model_optim_rng.pt
```

## 3. Start SFT Training

### 3.1 Parameter Configuration Description

Based on supporting open-source Megatron parameters, LoongForge adds more convenient training startup parameters. Detailed configuration can be found in the `loongforge/train/arguments.py` file. Key Pi0.5-specific parameters are as follows:

**Model & Parallelism:**

* `--model-name pi05`: Selects the Pi0.5 model family, mapping to `configs/models/pi05/pi05.yaml`.


* `--training-phase sft`: Explicitly enables SFT training phase.
* `--ckpt-format torch`: Checkpoint format compatible with FSDP DTensor sharding.
* `--finetune`: Signals that this is a fine-tuning run (resets optimizer/scheduler state from the loaded checkpoint unless overridden).
* `--no-load-optim` / `--no-load-rng`: Do not restore optimizer or RNG state from checkpoint, starting fresh.

* `export CUDA_DEVICE_MAX_CONNECTIONS=8`: Required by FSDP; avoids deadlocks from connection limits.
* `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Enables expandable CUDA memory segments to reduce OOM errors during training.

### 3.2 SFT Training Script

The full Pi0.5 SFT training script is located at [examples/pi05/finetuning/sft_pi05.sh](../../../examples/pi05/finetuning/sft_pi05.sh). Below is an annotated version:

```bash
#!/usr/bin/env bash


# ── Path Configuration ─────────────────────────────────────────────────────
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
DATA_PATH=${DATA_PATH:-"/workspace/libero/"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/paligemma-3b-pt-224/"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/ckpt/"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/mnt/cluster/LoongForge/tensorboard-log/pi05/"}

# ── Environment Variables ───────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=8   # Required by FSDP
export USE_BF16_BUFFER=false           # DTensor does not support BF16 buffer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce OOM risk

# ── Distributed Launch (defaults to single node) ───────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# ── Data & Tokenizer ────────────────────────────────────────────────────────
DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 100,0,0
  --chat-template empty        # VLA does not use conversational prompt templates
  --num-workers 16
)

# ── Core Training Hyperparameters ───────────────────────────────────────────
TRAINING_ARGS=(
    --training-phase sft
    --micro-batch-size 12
    --global-batch-size 96
    --train-iters 30000
    --seq-length 762
    --max-position-embeddings 762
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --no-masked-softmax-fusion
    --ckpt-format torch
    --load $CHECKPOINT_PATH
    --no-load-optim
    --no-load-rng
    --seed 1234
    --lr 2.5e-8
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-iters 0
    --lr-decay-iters 30000
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-eps 1e-8
    --adam-beta2 0.95
    --weight-decay 0.01
    --finetune
    --bf16
    --use-precision-aware-optimizer
    --exp-avg-dtype fp32
    --exp-avg-sq-dtype bf16
    --num-distributed-optimizer-instances 1
    --save $CHECKPOINT_PATH
    --save-interval 30000
)

# ── Model & Distributed Backend ─────────────────────────────────────────────
MODEL_CONFIG_ARGS=(
    --model-name pi05
    --use-distributed-optimizer
    --distributed-backend nccl
)

# ── Logging ─────────────────────────────────────────────────────────────────
LOGGING_ARGS=(
    --log-interval 1
)

# ── Launch ───────────────────────────────────────────────────────────────────
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-} \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```
