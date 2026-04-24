# Quick Start: VLM SFT

This document will guide you through the quick start process for Vision-Language Model (VLM) fine-tuning under the LoongForge framework.

## 1. Data Preparation

### 1.1 Dataset Configuration and Processing

In VLM instruction fine-tuning scenarios, the **multimodal ShareGPT** format (containing `messages` and `images`) is used. LoongForge parses this format through LoongForge/configs/data/sft_dataset_config.yaml. Below is the **ShareGPT format example**:

```yaml
multimodal:
  format: sharegpt
  columns:
    messages: messages
    images: images
  tags:
    role_tag: role
    content_tag: content
    user_tag: user
    assistant_tag: assistant
```

* **role_tag**: In the messages list, the key name representing the "role field" is role.
* **content_tag**: In the messages list, the key name representing the "content field" is content.
* **user_tag**: When the role field value is user, it indicates the message is from the user.
* **assistant_tag**: When the role field value is assistant, it indicates the message is from the assistant.

`tags` tells the parser which field names are used in the message structure and what the role values are. If your data uses different keys or role values, you should update them here accordingly.

### 1.2 Dataset Parameter Description

* `--data-path`: Dataset path (multiple paths can be specified for mixed training).
* `--sft-dataset-config`: Dataset configuration file path, default is sft_dataset_config.yaml.
* `--packing-sft-data`: Enable online packing mode
* `--packing-buffer-size`: Packing batch size, affecting packing efficiency and memory usage

### 1.3 Dataset Preprocessing

The process of converting to **Energon loading format** is the same as the pre-training section, see section 1.2 in [Quick Start: VLM Model Pretrain Training](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html). The framework provides two data preprocessing methods: online packing and offline packing, described below:

* **Online Packing**

Enable under DATA_ARGS in the training script: `--packing-sft-data`, `--packing-buffer-size` to activate online packing mode. This concatenates multiple shorter samples into the same sequence to improve token utilization. The packing processing batch size indicates the number of samples processed in each packing operation. Larger values typically result in better packing effects but higher preprocessing overhead and memory usage.

* **Offline Packing**

Provides an "offline sequence packing" pipeline: groups and rearranges **sample-level** data directories (one `json` per sample + several media files) according to `max_token_len`, generating **packed WebDataset** (`pretrain-*.tar` + Energon metadata) to improve training throughput and reduce padding. For further understanding of offline packing details, refer to: [offline_data_packing.md](https://loongforge.readthedocs.io/en/latest/features/offline_data_packing.html)

## 2. Model Weight Preparation

This section is the same as the pre-training section, see section 2 in [Quick Start: VLM Model Pretrain Training](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html)

## 3. Start SFT Training

### 3.1 Parameter Configuration Description

Based on supporting open-source Megatron parameters, LoongForge adds more convenient training startup parameters. Detailed configuration can be found in the loongforge/train/arguments.py file. Main parameter descriptions are as follows:

* `--training-phase sft`: Explicitly enable SFT training phase.
* `--chat-template qwen2-vl`: Specify SFT conversation template as qwen2-vl for concatenating multi-round dialogue samples into model input
* `+model.image_encoder.freeze=True`: Override through Hydra configuration to freeze image encoder model parameters for training

### 3.2 SFT Training Script

LoongForge currently provides SFT training example scripts for various models. After entering the container, you can find relevant scripts in the `examples/{model}/finetuning/` directory. Below is an example using Qwen3_vl_30b_a3b SFT training script:

```bash
#!/bin/bash
# The script needs to be run on at least 2 nodes.

# Codebase roots added to PYTHONPATH.
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# Dataset root or manifest path used by the external dataloader.
DATA_PATH=${DATA_PATH:-"/path/to/your/dataset"}

# TOKENIZER_PATH: HF tokenizer directory，must match the model vocabulary.
TOKENIZER_PATH=${TOKENIZER_PATH:-"/path/to/your/hf/tokenizer"}

# Paths for loading and saving weights
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/path/to/your/mcore/checkpoint"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/path/to/your/mcore/checkpoint_save"}

# TensorBoard log directory for training metrics.
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/path/to/your/tensorboard"}

# GPU count per node used by torchrun.
GPUS_PER_NODE=8

# Change for multinode config
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

# To specify the model config file
MODEL_CONFIG_PATH=${LOONGFORGE_PATH}/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml

# Data & tokenizer setup
DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path $TOKENIZER_PATH
    --data-path $DATA_PATH
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
)

# Core training hyperparameters
TRAINING_ARGS=(
    --training-phase sft
    --seq-length 32768
    --max-position-embeddings 262144
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 32
    --lr 6.0e-5
    --min-lr 6.0e-5
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 5000
    --eval-iters 0
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH_SAVE
    --save-interval 10000000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
)

# MoE router and expert behavior
MOE_ARGS=(
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --empty-unused-memory-level 2
    --moe-token-dispatcher-type alltoall
)

# Parallelism and distributed training
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
)

MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir ${TENSORBOARD_PATH}
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT}
        --wandb-exp-name ${WANDB_NAME}
    )
fi

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    $LOONGFORGE_PATH/loongforge/train.py \
    ${MODEL_CONFIG_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} 
```

### 3.3 Monitor Logs

The script will output TensorBoard logs to the directory specified by TENSORBOARD_PATH by default. You can view training curves through TensorBoard.