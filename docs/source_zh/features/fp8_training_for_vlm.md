# VLM 的 FP8 训练
LoongForge 为多种模型提供 FP8 低精度训练。通过编辑相应的 YAML 配置文件，可以独立地为视觉/音频**编码器**和语言**基座模型**开启/关闭 FP8，实现最佳训练效率。

---

## 1. 支持的模型
已验证支持 FP8 的 VLM 模型：

| 模型 | FP8 支持 |
|------|----------|
| LLaVA-OneVision-1.5 | ✅ |
| Qwen2.5-VL | – |
| Qwen3-VL | ✅ |
| InternVL 3.5 | – |

---

## 2. 如何运行 FP8 训练
下面以 **Qwen3-VL 30 B** 为例。

### 2.1 全局开启 FP8
在 `examples/qwen3_vl/pretrain/pretrain_qwen3_vl_30b_a3b.sh` 中添加 FP8 相关的启动参数：

```bash
TRAINING_ARGS=(
    --training-phase pretrain        # pretrain | sft
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
    --train-iters 50000
    --lr-decay-iters 50000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    #--load $CHECKPOINT_PATH
    #--save $CHECKPOINT_PATH
    --save-interval 10000000
    --ckpt-format torch
    --dataloader-save ${CHECKPOINT_PATH}/dataloader
    # <-- FP8 block-wise GEMM & weights -->
    --fp8-format e4m3
    --fp8-recipe blockwise
    --fp8-param-gather
)
```

使用这些参数后，**整个** Qwen3-VL 模型将以 FP8 进行训练。

---

### 2.2 选择性启用 FP8
如果只想对**编码器**和/或**基座模型**启用 FP8，请编辑 YAML 配置：

`configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml`

```yaml
# hydra:
#   searchpath:
#     - file://configs/

defaults:
  - ../../models/image_encoder@model.image_encoder: qwen3_vit
  - ../../models/image_projector@model.image_projector: qwen_mlp_adapter
  - ../../models/qwen3@model.foundation: qwen3_30b_a3b
  - _self_

model:
  model_type: qwen3_vl
  position_idx_func: ${position_func:rope_ids_qwen3vl}
  loss_func: ${loss_func:default}
  foundation:
    rotary_emb_func: "Qwen3VLRotaryEmbedding"
    mrope_section: [24, 20, 20]
    rotary_base: 1000000
    model_spec: ["loongforge.models.foundation.qwen3.qwen_layer_spec",
                 "get_qwen3_vl_layer_with_te_spec"]
    # <-- FP8 block-wise GEMM & weights -->
    fp8: "e4m3"
    fp8_recipe: "blockwise"
    fp8_param: True
  image_encoder:
    model_spec: ["loongforge.models.encoder.qwen3_vl_vision_models.qwen3_vl_layer_spec",
                 "get_qwen3_vl_vision_model_layer_with_te_spec"]
    # <-- FP8 block-wise GEMM & weights -->
    fp8: "e4m3"
    fp8_recipe: "blockwise"
    fp8_param: True
  image_projector:
    activation_func: ${act:gelu}
    normalization: "LayerNorm"
```

使用以上配置，仅**基座模型**和 **image_encoder** 模块将以 FP8 运行；其他部分（如投影层、损失函数等）保持原始精度。
