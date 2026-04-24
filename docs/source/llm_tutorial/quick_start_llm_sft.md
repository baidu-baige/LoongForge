# Quick Start: LLM SFT

This guide walks you through launching a **Supervised Fine-Tuning (SFT)** job for Large-Language Models (LLM) in the LoongForge framework.

---

## 1. Prepare the data

### 1.1 Dataset format & configuration
For instruction tuning, two dialogue styles are common: **Alpaca-style** and **ShareGPT-style**.  
LoongForge currently supports **Alpaca-style JSON**, one sample per line:

```json
[
  {
    "instruction": "User instruction",
    "input": "User question",
    "output": "Model answer"
  }
]
```

Field names may differ across datasets, so you must supply a **dataset config file** that tells the loader how to map your columns.  
A template is shipped at  
`/workspace/LoongForge/configs/data/sft_dataset_config.yaml`.

#### (1) File format  
We follow the community convention used by [LlamaFactory](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md).

#### (2) Default section  
If your file already uses the canonical Alpaca names, simply rely on the built-in `default` block:

```yaml
default:
  format: alpaca
  columns:
    prompt: instruction
    query: input
    response: output
```

#### (3) Adding a custom dataset  
Suppose your file is called `custom_dataset_name.json` and contains extra fields:

```json
[
  {
    "instruction": "...",
    "input": "...",
    "output": "...",
    "system": "System prompt",
    "history": [
      ["Q1", "A1"],
      ["Q2", "A2"]
    ]
  }
]
```

Append a new section to the YAML:

```yaml
custom_dataset_name:
  format: alpaca
  columns:
    prompt: instruction
    query: input
    response: output
    system: system
    history: history
```

### 1.2 Dataset arguments
| Argument | Meaning |
|---------|---------|
| `--data-path` | Path to the JSON file(s). **Multiple datasets** are supported; give sampling weights with colon separator: `path1:weight1,path2:weight2`. |
| `--split` | Train/valid/test ratio, e.g. `--split 90,8,2`. |
| `--sft-dataset-config` | Path to the YAML file described above. **Default:** `configs/data/sft_dataset_config.yaml`. |
| `--sft-dataset` | Name of the dataset entry inside the YAML. Must **correspond 1-to-1** with the order in `--data-path` when you use several datasets. |

### 1.3 Pre-processing modes
LoongForge supports **on-the-fly** and **offline** pre-processing.  
Pick **offline** when the dataset is large; it removes tokenisation overhead from the critical path.

#### (1) On-the-fly (default)
```bash
--data-path /path/to/custom_dataset_name.json \
--sft-dataset custom_dataset_name \
--sft-data-streaming   # optional: stream large files
```

#### (2) Offline
Run the helper script once:

```bash
#!/bin/bash
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

TOKENIZER_PATH=/path/to/hf/tokenizer
input_data=/path/to/custom_dataset_name.json
output_path=/path/to/save_dir

PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  python ${LOONGFORGE_PATH}/tools/data_preprocess/llm/preprocess_sft_data.py \
      --input ${input_data} \
      --output ${output_path} \
      --seq-length 2048 \
      --chat-template ${chat_template} \
      --tokenizer-type HFTokenizer \
      --hf-tokenizer-path $TOKENIZER_PATH \
      --workers 50 \
      --split 100,0,0
      --packing-sft-data
```

Key flags  
* `--seq-length` – max token length; longer samples are truncated.  
* `--chat-template` – must match the template you will use during training.  
* `--split` – pre-split the data; you still need `--split` in training for sanity checks.  
* `--packing-sft-data` – pack multiple samples into one sequence (no padding).

After pre-processing, pass the **directory** to training:

```bash
--data-path /path/to/save_dir \
--is-tokenized-data \
--sft-dataset custom_dataset_name
```

---

## 2. Prepare the checkpoint
Same as pre-training – see [Quick Start: LLM Pre-training](https://github.com/baidu-baige/LoongForge/tree/master/docs/source/llm_tutorial/quick_start_llm_pretrain.md).

---

## 3. Launch SFT training

### 3.1 Extra LoongForge arguments (beyond native Megatron)
| Argument | Purpose |
|---------|---------|
| `--training-phase sft` | Switch to fine-tuning stage. |
| `--chat-template` | Which conversation template to apply (`no-template`, `llama3`, `qwen`, …). |
| `--sft-dataset` / `--sft-train-dataset` / `--sft-valid-dataset` | Dataset name(s) in the YAML file. |
| `--packing-sft-data` | Enable sample packing to increase throughput. |
| `--sft-data-streaming` | Stream large JSON files instead of loading everything into RAM. |
| `--sft-num-preprocess-workers` | CPU workers for on-the-fly tokenisation. |
| `--reduce-variable-seq-shape-p2p-comm` | Pad p2p buffers to fixed length (recommended for SFT). |
| `--optimizer-cpu-offload` / `--optimizer-offload-fraction` | Offload optimiser states to CPU to save GPU memory. |

All FP8, pipeline-parallel, recompute, MoE, MTP flags keep the same meaning as in pre-training.

### 3.2 Example SFT script
Ready-to-run scripts are located under `examples/{model}/finetuning/`.  
Below is the FP8 SFT script for DeepSeek-V3.1 (comments added in English):

```bash
#!/bin/bash
# DeepSeek-V3 FP8 supervised fine-tuning

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# ------------- data -------------
DATA_PATH=/path/to/your/data
DATA_CACHE_PATH=/path/to/your/data/cache
DATASET_CONFIG_PATH=/path/to/your/dataset_config   # optional

# ------------- tokenizer & checkpoint -------------
TOKENIZER_PATH=/path/to/hf/tokenizer
CHECKPOINT_PATH=/path/to/mcore/checkpoint
CHECKPOINT_PATH_SAVE=/path/to/save_dir

# ------------- logging -------------
TENSORBOARD_PATH=/path/to/tensorboard

# ------------- FP8 quantisation -------------
export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

GPUS_PER_NODE=8

# ------------- NCCL & NVSHMEM -------------
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
# choose HCA list according to your cluster
export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_11,mlx5_12,mlx5_13
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_IB_TRAFFIC_CLASS=130
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3

# ------------- Transformer Engine -------------
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# ------------- CUDA / PyTorch -------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------- distributed -------------
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

# ------------- model -------------
MODEL_ARGS=(
  --model-name deepseek-v3
  --multi-latent-attention
  --rotary-base 10000
  --original-max-position-embeddings 4096
  --mscale 1.0
  --mscale-all-dim 1.0
  --norm-epsilon 1e-6
  --rotary-scaling-factor 40
  --enable-fa-within-mla
  --use-fp32-dtype-for-param-pattern expert_bias
)

# ------------- data loader -------------
DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 90,8,2
)

# ------------- SFT specific -------------
SFT_ARGS=(
  --chat-template no-template
  --sft-num-preprocess-workers 16
  --no-check-for-nan-in-loss-and-grad
  --packing-sft-data
  --sft-dataset sharegpt
)

# ------------- training hyper-params -------------
TRAINING_ARGS=(
  --training-phase sft
  --seq-length 65536
  --max-position-embeddings 163840
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 128
  --lr 1e-06
  --train-iters 1500
  --lr-decay-iters 5000
  --lr-decay-style cosine
  --min-lr 1.0e-7
  --weight-decay 0.1
  --lr-warmup-fraction 0.002
  --clip-grad 1.0
  --bf16
  --load $CHECKPOINT_PATH
  --save $CHECKPOINT_PATH_SAVE
  --save-interval 1000
  --eval-interval 10
  --eval-iters 1
  --no-load-optim
  --no-load-rng
  --recompute-granularity full
  --recompute-method block
  --custom-pipeline-layers 8,7,8,8,8,8,8,6
  --custom-pipeline-recompute-layers 8,7,8,8,8,8,8,6
  --num-virtual-stages-per-pipeline-rank 2
  --reduce-variable-seq-shape-p2p-comm
  --fp8-format e4m3
  --fp8-recipe blockwise
  --fp8-param-gather
  --enable-fp8-comm
  --distributed-timeout-minutes 60
  --optimizer-cpu-offload
  --optimizer-offload-fraction 1.0
  --enable-experimental
)

# ------------- MoE -------------
MOE_ARGS=(
  --moe-router-load-balancing-type seq_aux_loss
  --moe-router-topk 8
  --moe-aux-loss-coeff 1e-3
  --moe-grouped-gemm
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0.001
  --moe-router-num-groups 8
  --moe-router-group-topk 4
  --moe-router-score-function sigmoid
  --moe-router-topk-scaling-factor 2.5
  --moe-router-dtype fp32
  --empty-unused-memory-level 2
)

# ------------- parallelism & optimiser -------------
MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 8
  --expert-model-parallel-size 32
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --moe-token-dispatcher-type flex
  --moe-enable-deepep
  --use-precision-aware-optimizer
  --exp-avg-dtype bf16
  --exp-avg-sq-dtype bf16
  --use-distributed-optimizer
  --moe-permute-fusion
  --cross-entropy-loss-fusion
  --overlap-grad-reduce
  --overlap-param-gather
)

# ------------- MTP -------------
MTP_ARGS=(
  --mtp-loss-scaling-factor 0.1
)

# ------------- logging -------------
LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --check-weight-hash-across-dp-replicas-interval 30
)

# ------------- launch -------------
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $LOONGFORGE_PATH/loongforge/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${SFT_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  ${MTP_ARGS[@]}
```

---

## 4. Monitoring
TensorBoard logs are written to the directory specified by `TENSORBOARD_PATH`.  
Open TensorBoard to inspect loss, perplexity, memory, throughput, etc.