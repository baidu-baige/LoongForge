# Quick Start: VLM Pre-training

This document will guide you through the quick start process for Vision-Language Model (VLM) pre-training under the LoongForge framework.

## 1. Data Preparation

Before model training, you need to process and convert large-scale pre-training data to maximize training speed. The specific process is as follows:

### 1.1 Raw Data

The raw dataset is in JSON/JSONL format, with each piece of data containing image paths and corresponding multi-round dialogue content.

**Sample Data (data.json):**

```json
[
  {
    "messages": [
      {
        "content": "<image>Who are they?",
        "role": "user"
      },
      {
        "content": "They're Kane and Gretzka from Bayern Munich.",
        "role": "assistant"
      },
      {
        "content": "What are they doing?",
        "role": "user"
      },
      {
        "content": "They are celebrating on the soccer field.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg"
    ]
  }
]
```

### 1.2 Convert to **Energon Loading Format**

Considering the diversity of multimodal datasets, the framework adopts the **Energon** loader to improve data processing performance. **Energon** requires datasets to be stored in standard **WebDataset** format. WebDataset stores data in native file formats (jpg, mp4, etc.), which allows various native multimodal datasets to be simply compressed and converted to WebDataset format, then read by Energon.

The conversion script to **WebDataset and adapt to Energon loading format** is as follows:

```bash
python /workspace/LoongForge/tools/data_preprocess/vlm/convert_to_webdataset.py \
    --output_dir /tmp/mllm/wds \
    --json_file /tmp/mllm/mllm_demo.json \
    --image_dir /tmp/mllm/ \
    --video_dir /tmp/vlm/ \
    --media mix \
    --columns_messages messages \
    --maxcount 10000 \
    --maxsize 3000000000 \
    --sample_type multi_mix_qa
```

Converted dataset directory:

```
.
├── .nv-meta
│   ├── .info.yaml
│   ├── dataset.yaml
│   └── split.yaml
├── pretrain-0.tar
└── pretrain-0.tar.idx
```

Function Description:

* `convert_to_webdataset.py` will extract each sample from **json_file** and store it as an independent json file, together with the corresponding images from **image_dir**, compressed into **$output_dir**. Each tar package contains at most **maxcount** samples;
* When starting training later, specify the WebDataset path /tmp/mllm/wds through the --data-path parameter for training data reading;
* Energon format adds yaml files to record dataset information compared to WebDataset, used for subsequent dataloader parsing
    * `.info.yaml`: records the number of samples in each compressed package
    * `dataset.yaml`: records sample information
    * `split.yaml`: records dataset division

For further understanding of various parameters and detailed functions of dataset conversion, refer to [dataset_conversion.md](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/dataset_conversion.html)

## 2. Model Weight Preparation

Training usually starts with open-source Hugging Face weights. We need to download the weights first, then convert them to the format supported by this framework (Megatron-Core format).

### 2.1 Download Hugging Face Model

Take Qwen3-VL-30B-A3B as an example, please download model weights from Hugging Face ([https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)).

### 2.2 Convert Weight Format

LoongForge provides a unified weight conversion tool `tools/convert_checkpoint` for supported models, which can conveniently convert between Huggingface and Mcore formats. Taking Qwen3-VL-30B-A3B as an example, if you need to convert Huggingface weights to MegatronCore format supported by LoongForge, you can refer to the following example:

```bash
#!/bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="${LOONGFORGE_PATH}/tools/convert_checkpoint"

LOAD=/path/to/hf_checkpoint  # the original Qwen3-VL-30B-A3B checkpoint path
SAVE=/path/to/your/save  # the converted checkpoint save path

SAVE_LANGUAGE_MODEL=${SAVE}/tmp/language-mcore
SAVE_VISION_MODEL=${SAVE}/tmp/vision-model-mcore
SAVE_ADAPTER=${SAVE}/tmp/adapter-mcore
SAVE_PATCH=${SAVE}/tmp/patch-mcore

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert_qwen3vl.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen3_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_3_mlp_adapter_convert.yaml

ETP=1
DTP=1
PP=2
EP=8

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --expert_parallel_size=$EP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \
    --moe-grouped-gemm

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# merge
if [ $EP -gt 1 ]; then
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron_expert.py\
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --expert_parallel_size $EP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE 
else
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py\
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE 
fi

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
```

Partial Parameter Description:

* ETP / DTP: Encoder and Decoder tensor parallelism parameters (support heterogeneous parallelism strategies where ETP != DTP)

For further understanding of various parameters and detailed functions of weight conversion, please refer to: [vlm_ckpt_convert.md](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/vlm_ckpt_convert.html)

## 3. Start Pre-training

### 3.1 Parameter Configuration Description

Based on supporting parameters provided by open-source Megatron, LoongForge adds more convenient training startup parameters. Detailed configuration can be found in the loongforge/train/arguments.py file. Main parameter descriptions are as follows:

* `--training-phase`: Specify the training phase as pretrain
* `--add-question-in-pretrain`: When enabled, questions will be concatenated and added to the input for training; when disabled, only answers or other default text fields will be used for training
* `--enable-discard-sample`: When enabled, samples exceeding --seq-length will be directly discarded without truncation or other processing
* `--dataloader-save`: When enabled, the dataloader state will be written to this path during training, facilitating consistent data reading order recovery during checkpoint restart
* `--packing-sft-data`: When enabled, online packing strategy will be activated, concatenating multiple shorter samples into one long sample

### 3.2 Pre-training Script

LoongForge currently provides pre-training example scripts for various models. After entering the container, you can find relevant scripts in the examples/{model}/pretrain/ directory. Below is an example using Qwen3-VL-30B-A3B pre-training script:

```bash
#!/bin/bash
# The script needs to be run on at least 2 nodes.

# Codebase roots added to PYTHONPATH.
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}

# Dataset root or manifest path used by the external dataloader.
DATA_PATH=${DATA_PATH:-"/path/to/your/dataset"}

# TOKENIZER_PATH: HF tokenizer directory，must match the model vocabulary.
TOKENIZER_PATH=${TOKENIZER_PATH:-"/path/to/your/hf/tokenizer"}

# Paths for loading and saving weights
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/path/to/your/mcore/checkpoint"}
CHECKPOINT_PATH_SAVE=${CHECKPOINT_PATH_SAVE:-"/path/to/your/mcore/checkpoint_save"}

# TensorBoard log directory for training metrics.
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"/path/to/your/tensorboard"}

# GPU count per node used by torchrun.
GPUS_PER_NODE=8

# Change for multinode confi,Distributed training rendezvous settings.
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
    --add-question-in-pretrain
    --enable-discard-sample
    --num-workers 16
)

# Core training hyperparameters
TRAINING_ARGS=(
    --training-phase pretrain # options: pretrain, sft
    --seq-length 32768
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 32
    --lr 0.0002
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters 5000
    --eval-iters 0
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH_SAVE
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
)

# Parallelism and distributed training
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 8
    --moe-token-dispatcher-type alltoall
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-backend nccl
)

# Model architecture/config file
MODEL_CONFIG_ARGS=(
    --config-file $MODEL_CONFIG_PATH
)

# Logging & monitoring
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

### Monitor Logs

The script will output TensorBoard logs to the directory specified by TENSORBOARD_PATH by default. You can view training curves through TensorBoard.
