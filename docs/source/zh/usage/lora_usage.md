# LoRA 功能使用指南

LoongForge 框架支持 LoRA（Low-Rank Adaptation）训练，以减少 GPU 显存消耗并降低训练所需的计算资源。

## 在 LLM 中使用 LoRA
1. 修改配置文件以启用 LoRA

在你要训练的模型配置文件中添加 LoRA 相关配置。LLM LoRA 配置文件位于 `${LoongForge}/configs/models/lora/lora.yaml`。可配置的参数包括：

* **target_modules**：要替换为 LoRA 的目标模块的通配符模式。框架会匹配模型中的每个模块，匹配成功的模块将被替换为 LoRA 模块。
* **dim**：控制低秩矩阵的维度。
* **alpha**：控制 LoRA 更新的缩放因子，用于调整适应强度。
* **dropout**：在 LoRA 层训练期间应用 dropout 以防止过拟合。

要在模型中启用 LoRA，只需在模型配置文件中包含 LoRA 配置。例如，要在 Qwen3-1.7b 模型中使用 LoRA，修改 `${LoongForge}/configs/models/qwen3/qwen3_1_7b_lora.yaml` 如下：

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

在训练脚本中，你需要指定额外的参数：

* **--pretrained-checkpoint**：预训练模型权重。你需要指定基座模型的 MCore 权重路径，如 `/workspace/qwen3-1.7b-tp1-pp1`。
* **--load**：加载 LoRA 权重。指定加载 LoRA 权重的路径。
* **--save**：保存 LoRA 权重。指定保存 LoRA 权重的路径。

例如，在 Qwen3-1.7b 模型训练脚本中添加以下内容：

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

## 在 VLM 中使用 LoRA
1. 修改配置文件以启用 LoRA

在你要训练的模型配置文件中添加 LoRA 相关配置。VLM LoRA 配置文件位于 `${LoongForge}/configs/models/lora/vlm_lora.yaml`。除了上述 LoRA 可配置参数外，还包括：

* **apply_to_foundation**：在基座模型上启用 LoRA 训练。
* **apply_to_image_projector**：在图像投影层上启用 LoRA 训练。
* **apply_to_image_encoder**：在图像编码器上启用 LoRA 训练。

要在模型中启用 LoRA，只需在模型配置文件中包含 LoRA 配置。例如，要在 Qwen2.5-vl-3b 模型中使用 LoRA，修改 `${LoongForge}/configs/models/qwen2.5/qwen2_5_vl_3b.yaml` 如下：

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

在训练脚本中，你需要指定额外的参数：

* **--pretrained-checkpoint**：预训练模型权重。你需要指定基座模型的 MCore 权重路径，如 `/workspace/qwen2.5-vl-3b-tp1-pp1`。
* **--load**：加载 LoRA 权重。指定加载 LoRA 权重的路径。
* **--save**：保存 LoRA 权重。指定保存 LoRA 权重的路径。

例如，在 Qwen2.5-vl-3b 模型训练配置中添加以下内容：

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

## 合并基座模型与 LoRA 权重并转换为 HF 格式

使用框架提供的离线权重转换工具，可以将 LoRA 合并到基座模型并转换为 Hugging Face 格式保存。以下是使用示例：

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
