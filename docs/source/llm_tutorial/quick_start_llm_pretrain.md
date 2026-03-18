# Quick Start: LLM Pre-training

This guide walks you through launching a Large-Language-Model (LLM) pre-training job in the BaigeOmni framework.

---

## 1. Prepare the data

### 1.1 Data pre-processing
Before training you usually need to transform massive raw corpora into a format that maximises training speed.

1. Organise the corpus as **newline-delimited JSON**, one document per line:

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

2. Launch the container and run the built-in tool:

```bash
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
OMNI_PATH=${OMNI_PATH:-"/workspace/BaigeOmni"}

TOKENIZER_PATH=/path/to/your/tokenizer
input_data=/path/to/your/json
output_prefix=/path/to/your/output_prefix

PYTHONPATH=$MEGATRON_PATH:$OMNI_PATH:$PYTHONPATH \
  python ${OMNI_PATH}/tools/data_preprocess/llm/preprocess_pretrain_data.py \
      --input ${input_data} \
      --output-prefix ${output_prefix} \
      --tokenizer-type HFTokenizer \
      --hf-tokenizer-path $TOKENIZER_PATH \
      --json-keys text \
      --workers 50 \
      --append-eod
```

Example scripts for each model family can be found under `examples/{model}/pretrain/`.

---

## 2. Prepare the checkpoint

Training usually starts from an open-source Hugging-Face checkpoint.  
Download it first, then convert it to the Megatron-Core format that the framework expects.

### 2.1 Download HF checkpoint
Take DeepSeek-V3.1 as an example:  
https://huggingface.co/deepseek-ai/DeepSeek-V3.1

### 2.2 Convert the checkpoint
Baige provides a unified converter `tools/convert_checkpoint`.  
Below we convert the original FP8 HF checkpoint to MCore FP8:

```bash
#!/bin/bash
export OMNI_PATH=${OMNI_PATH:-"/workspace/BaigeOmni"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
CONVERT_CHECKPOINT_PATH="$OMNI_PATH/tools/convert_checkpoint"

LOAD=/path/to/hf_checkpoint          # FP8 HF checkpoint
SAVE=/path/to/your/save              # will be MCore FP8

MODEL_CONFIG_FILE=${OMNI_PATH}/configs/models/deepseek3/deepseek_v3.yaml
CONVERT_FILE=${OMNI_PATH}/configs/models/deepseek3/ckpt_convert/deepseek_v3_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --expert_parallel_size=32 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --custom_pipeline_layers 8,7,8,8,8,8,8,6 \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    --amax_epsilon=1e-12
```

Key flags:
* `--amax_epsilon` – FP8 quantisation scale; keep it identical to the FP8_EPS env var used in training.  
* `--custom_pipeline_layers` – number of layers on each PP stage.  
  If VPP is enabled (`--num-virtual-stages-per-pipeline-rank`), list the layers of every vPP chunk in order, e.g.  
  `--custom_pipeline_layers 4,3,4,4,4,4,4,3,4,4,4,4,4,4,4,3` for 2 virtual stages.

See [llm_ckpt_convert.md](https://github.com/baidu-baige/BaigeOmni/tree/master/docs/source/llm_tutorial/llm_ckpt_convert.md) for full details.

---

## 3. Launch pre-training

### 3.1 Extra arguments provided by BaigeOmni
Besides the native Megatron flags, the framework adds convenient options (defined in `baige_omni/train/arguments.py`):

* `--config-file` – path to a YAML file that contains all model hyper-params, e.g. `configs/models/deepseek3/deepseek_v3.yaml`.  
* `--model-name` – short name such as `deepseek-v3`; the system looks up the YAML automatically.  
* `--training-phase` – `pretrain`, `sft`, etc.  
* `--tokenizer-type` – recommend `HFTokenizer` plus `--hf-tokenizer-path`.  
* `--no-create-attention-mask-in-dataloader` – skip attention mask creation to speed up data loading.  
* `--fp8-format` – `e4m3` (same as DeepSeek-V3 paper).  
* `--fp8-recipe` – `blockwise` quantisation.  
* `--fp8-param-gather` – keep parameters in FP8; omit it to keep BF16.  
* `--enable-fp8-comm` – dispatch FP8 tensors during MoE communication.  
* `--custom-pipeline-layers` – per-stage layer assignment, e.g. `19,20,20,21`.  
* `--custom-pipeline-recompute-layers` – per-stage recompute layers, e.g. `10,11,12,13`.  
* `--reduce-variable-seq-shape-p2p-comm` – pad p2p buffers to fixed length (useful for SFT).  
* `--use-fp32-dtype-for-param-pattern` – keep certain params in FP32.

### 3.2 Example pre-training script
Ready-to-run scripts are provided under `examples/{model}/pretrain/`.  
Below is the FP8 pre-training script for DeepSeek-V3.1 (comments added for clarity):

```bash
#!/bin/bash
# DeepSeek-V3 FP8 mixed-precision pre-training

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # allow non-weights-only loading

MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM"}
OMNI_PATH=${OMNI_PATH:-"/workspace/BaigeOmni"}

DATA_PATH=/path/to/your/dataset
TOKENIZER_PATH=/path/to/your/hf/tokenizer
CHECKPOINT_PATH=/path/to/your/mcore/checkpoint
CHECKPOINT_PATH_SAVE=/path/to/your/save_dir
TENSORBOARD_PATH=/path/to/your/tensorboard

# FP8 quantisation EPS (avoid NaN)
export FP8_QUANT_FWD_INP_AMAX_EPS=1e-12
export FP8_QUANT_FWD_WEIGHT_AMAX_EPS=1e-12
export FP8_QUANT_BWD_GRAD_AMAX_EPS=1e-12

GPUS_PER_NODE=8

# NCCL & NVSHMEM tuning for multi-node
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3
export NVSHMEM_HCA_LIST=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3

# Transformer-Engine knobs
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=24
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# CUDA & PyTorch optimisations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Distributed job settings
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"5000"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NNODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

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
  --use-fp32-dtype-for-param-pattern '^expert_bias$' '.+\.expert_bias$'
)

DATA_ARGS=(
  --tokenizer-type HFTokenizer
  --hf-tokenizer-path $TOKENIZER_PATH
  --data-path $DATA_PATH
  --split 99990,8,2
  --no-create-attention-mask-in-dataloader
)

TRAINING_ARGS=(
  --training-phase pretrain
  --seq-length 32768
  --max-position-embeddings 163840
  --init-method-std 0.02
  --no-masked-softmax-fusion
  --micro-batch-size 1
  --global-batch-size 1024
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
  --save-interval 100
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
  --enable-experimental
)

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

MTP_ARGS=(
  --mtp-loss-scaling-factor 0.1
)

LOGGING_ARGS=(
  --log-interval 1
  --tensorboard-dir ${TENSORBOARD_PATH}
  --log-timers-to-tensorboard
  --log-memory-to-tensorboard
  --log-validation-ppl-to-tensorboard
  --check-weight-hash-across-dp-replicas-interval 30
)

PYTHONPATH=$MEGATRON_PATH:$OMNI_PATH:$PYTHONPATH \
  torchrun ${DISTRIBUTED_ARGS[@]} \
  $OMNI_PATH/baige_omni/train.py \
  ${MODEL_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} \
  ${MTP_ARGS[@]}
```

---

## 4. Monitoring

The script writes TensorBoard logs to the directory specified by `TENSORBOARD_PATH`.  
Launch TensorBoard and open the browser to inspect loss curves, throughput, memory usage, etc.