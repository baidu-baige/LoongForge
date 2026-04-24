# LoRA Feature Usage Guide

The LoongForge framework supports LoRA (Low-Rank Adaptation) training to reduce GPU memory consumption and lower the computational resources required for training.

## Using LoRA in LLM
1. Modify the configuration file to enable LoRA

Add LoRA-related configurations to the model configuration file you want to train. The LLM LoRA configuration file is located at `${LoongForge}/configs/models/lora/lora.yaml`. The configurable parameters include:

* **target_modules**: Wildcard patterns for target modules to be replaced with LoRA. The framework matches each module in the model, and modules that match successfully will be replaced with LoRA modules.
* **dim**: Controls the dimension of low-rank matrices.
* **alpha**: Controls the scaling factor of LoRA updates to adjust the adaptation strength.
* **dropout**: Applies dropout during the training of LoRA layers to prevent overfitting.

To enable LoRA in the model, simply include the LoRA configuration in the model's configuration file. For example, to use LoRA in the Qwen3-1.7b model, modify `${LoongForge}/configs/models/qwen3/qwen3_1_7b_lora.yaml` as follows:

```yaml
# qwen3 model configuration
_target_: loongforge.models.foundation.Qwen3Config

defaults:
  - ../../models/lora@peft_config: lora

num_layers: 28
hidden_size: 2048
ffn_hidden_size: 6144
num_attention_heads: 16
vocab_size_in_config_file: 151936
make_vocab_size_divisible_by: 128

group_query_attention: true
num_query_groups: 8
position_embedding_type: "rope"
add_position_embedding: false
rotary_interleaved: false
normalization: "RMSNorm"
swiglu: true
attention_dropout: 0
hidden_dropout: 0
add_bias_linear: false
add_qkv_bias: false
qk_layernorm: true
untie_embeddings_and_output_weights: true
word_embeddings_for_head: "lm_head"
kv_channels: 128
num_experts: null
moe_ffn_hidden_size: null
rotary_emb_func: "RotaryEmbedding"
rotary_base: 1000000
model_type: "qwen"
# variable_seq_lengths: true
```

In the training script, you need to specify additional parameters:

* **--pretrained-checkpoint**: Pre-trained model weights. You need to specify the path to the base model's MCore weights, such as `/workspace/qwen3-1.7b-tp1-pp1`.
* **--load**: Load LoRA weights. Specify the path to load LoRA weights.
* **--save**: Save LoRA weights. Specify the path to save LoRA weights.

For example, add the following to the Qwen3-1.7b model training script:

```bash
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/qwen3-1.7b-tp1-pp1-Dec24"}

LORA_CHECKPOINT_PATH=${LORA_CHECKPOINT_PATH:-"/workspace/qwen3_1.7B_mcore_tp1pp1_lora"}

TRAINING_ARGS=(
    --training-phase pretrain # options: pretrain, sft
    --seq-length 4096
    --max-position-embeddings 32768
    --init-method-std 0.006
    --micro-batch-size 1
    --global-batch-size 32
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-08
    --norm-epsilon 1e-6
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $LORA_CHECKPOINT_PATH
    --save $LORA_CHECKPOINT_PATH
    --pretrained-checkpoint $CHECKPOINT_PATH
    --save-interval 5000
    --eval-interval 1000
    --eval-iters 10
)
```

## Using LoRA in VLM
1. Modify the configuration file to enable LoRA

Add LoRA-related configurations to the model configuration file you want to train. The VLM LoRA configuration file is located at `${LoongForge}/configs/models/lora/vlm_lora.yaml`. In addition to the LoRA configurable parameters mentioned above, it also includes:

* **apply_to_foundation**: Enable LoRA training on the foundation model.
* **apply_to_image_projector**: Enable LoRA training on the image projector.
* **apply_to_image_encoder**: Enable LoRA training on the image encoder.

To enable LoRA in the model, simply include the LoRA configuration in the model's configuration file. For example, to use LoRA in the Qwen2.5-vl-3b model, modify `${LoongForge}/configs/models/qwen2.5/qwen2_5_vl_3b.yaml` as follows:

```yaml
defaults:
  - ../../models/image_encoder@model.image_encoder: qwen2_5_vit
  - ../../models/image_projector@model.image_projector: qwen_mlp_adapter
  - ../../models/qwen2.5@model.foundation: qwen2_5_3b
  - ../../models/lora@model.peft_config: vlm_lora
  - _self_

model:
  model_type: qwen2_5_vl
  position_idx_func: ${position_func:mrope_ids}
  loss_func: ${loss_func:default}
  mix_used_vision_encoder: true
  mix_used_vision_projector: true
  foundation: 
    rotary_emb_func: "Qwen2VLRotaryEmbedding"
    model_spec: ["loongforge.models.foundation.qwen2.qwen_layer_spec", "get_qwen2_vl_layer_with_te_spec"]
    rotary_base: 1000000
    group_query_attention: true
  image_projector:
    activation_func: ${act:gelu}
    freeze: true
  image_encoder:
    freeze: true
  peft_config:
    apply_to_foundation: true
```

In the training script, you need to specify additional parameters:

* **--pretrained-checkpoint**: Pre-trained model weights. You need to specify the path to the base model's MCore weights, such as `/workspace/qwen2.5-vl-3b-tp1-pp1`.
* **--load**: Load LoRA weights. Specify the path to load LoRA weights.
* **--save**: Save LoRA weights. Specify the path to save LoRA weights.

For example, add the following to the Qwen2.5-vl-3b model training configuration:

```bash
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-tp1-pp1"}
LORA_CHECKPOINT_PATH=${LORA_CHECKPOINT_PATH:-"/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-3b-tp1-pp1-lora"}

TRAINING_ARGS=(
    --norm-epsilon 1e-6
    --training-phase pretrain
    --seq-length 1024
    --max-position-embeddings 4096
    --init-method-std 0.02
    --micro-batch-size 1
    --global-batch-size 512
    --lr 0.0002
    --min-lr 1.0e-5
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-05
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load $LORA_CHECKPOINT_PATH
    --save $LORA_CHECKPOINT_PATH
    --pretrained-checkpoint $CHECKPOINT_PATH
    --save-interval 10000000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
)
```

## Merging Base Model and LoRA Weights and Converting to HF Format

Using the offline checkpoint conversion tool provided by the framework, you can merge LoRA into the base model and convert it to Hugging Face format for saving. Here is a usage example:

```bash
#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/LoongForge/qwen3/qwen3-1.7b-tp1-pp1-Dec24/release/
SAVE=/mnt/cluster/LoongForge/qwen3/qwen3-1.7b-hf-Dec24
LOAD_LORA=/mnt/cluster/LoongForge/qwen3/qwen3-1.7b-tp1-pp1-Dec24/iter_0000010/

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/qwen3_1_7b.yaml

CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_convert.yaml

TP=1
PP=1

LORA_ALPHA=32
LORA_DIM=16

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=$TP \
    --pipeline_model_parallel_size=$PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --load_lora_ckpt_path=$LOAD_LORA \
    --lora_alpha=$LORA_ALPHA \
    --lora_dim=$LORA_DIM \
    --safetensors \
    --no_save_optim \
    --no_load_optim
```