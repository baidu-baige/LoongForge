# 灵活组网

**LoongForge** 框架的核心特性之一是**原生支持灵活组网**。
通过简单的 YAML 配置文件，用户可以自由组合视觉/音频编码器、模态对齐投影层和语言模型基座，快速构建定制化的多模态大模型。
只需编辑 `../configs/models` 下的模型配置 YAML 文件，即可在不修改任何代码的情况下重组或切换整个模型结构。

例如，不同的视觉编码器（**Qwen2.5-ViT、Qwen3-ViT、LLaVA-OV-1.5-ViT、InternViT** 等）可以与不同的 LLM 基座（**LLaMA 系列、DeepSeek 系列、Qwen 系列** 等）拼接。
这种**配置驱动组网**极大地降低了探索和定制多模态架构的成本，实现**零代码**模型构建。

---

## 1. 核心组件

### OmniEncoderModel
所有编码器组件的抽象基类。
它将图像、视频、音频或其他模态数据转换为 LLM 可理解的嵌入表示。

* **抽象化**：封装视觉/音频编码器和投影层，并统一管理文本嵌入。
* **关键实现**：
  * **多模态兼容性**：通过简单插入新分支即可添加新模态。

```python
# loongforge/models/omni_models/omni_encoder_model.py
class OmniEncoderModel(torch.nn.Module):
    def __init__(self, config, ...):
        # 文本模态
        self.text_encoder = LanguageModelEmbedding(...)

        # 图像模态
        if hasattr(config, "image_encoder"):
            self.image_encoder = AutoModel.from_config(config.image_encoder, ...)

        # 视频模态（可轻松扩展）
        if hasattr(config, "video_encoder"):
            self.video_encoder = AutoModel.from_config(config.video_encoder, ...)
```

  * **异构张量并行**：使用 hook 机制在异构设备间实现**编码器张量并行**。

```python
# 自动注册 hook，在前向前后切换并行状态
self.image_encoder.register_forward_pre_hook(
    make_encoder_forward_pre_hook("image_encoder")
)
self.image_encoder.register_forward_hook(
    make_encoder_forward_hook("text_decoder")  # 切回解码器状态
)
```

---

### OmniCombinationModel
多模态组合的核心组件。
定义数据**何时以及如何**在模态组件之间流动；**不包含实际计算逻辑**。

* **逻辑解耦**：通过外部配置动态决定是否加载编码器或基座模型，实现组件级解耦。

```python
# loongforge/models/omni_models/omni_combination_model.py
class OmniCombinationModel(BaseMegatronModule):
    def __init__(self, config, ...):
        # 1. 动态初始化编码器模型
        if config.image_encoder is not None and add_encoder:
            self.encoder_model = OmniEncoderModel(config, ...)

        # 2. 动态初始化 LLM
        if config.foundation is not None and add_decoder:
            self.foundation_model = AutoModel.from_config(config.foundation, ...)
```

---

### OmniModelProvider
解析全局参数，处理分布式初始化（如流水线并行拆分），最终实例化 `OmniCombinationModel`。

关键能力：

* **分布式流水线并行**：检测当前流水线并行阶段并按需加载编码器/基座模型，实现**跨 GPU 流水线部署**。

```python
# loongforge/models/omni_models/omni_model_provider.py
def omni_model_provider(...):
    # 自动检测当前 rank 是否为第一个 PP 阶段；决定是否加载编码器
    # 这对于将编码器和解码器放置在不同 GPU 上至关重要
    if args.encoder_pipeline_model_parallel_size in [0, None]:
        add_encoder = mpu.is_pipeline_first_stage()

    # 使用环境感知标志构建模型
    return OmniCombinationModel(..., add_encoder=add_encoder, ...)
```

* **参数桥接与适配**：获取全局训练参数和模型配置，将训练关键超参数（`language_vocab_size`、`language_max_sequence_length` 等）注入模型初始化，保证模型配置与训练环境的一致性。

```python
args = get_args()
model_config = get_model_config()

model = OmniCombinationModel(
    model_config,
    language_vocab_size=args.padded_vocab_size,
    language_max_sequence_length=args.max_position_embeddings,
    fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
    ...
)
```

---

## 2. 使用方法

我们采用基于 **Hydra** 的配置系统，通过 `defaults` 列表组合组件。
以 `configs/models/internvl2.5/internvl2_5_8b.yaml` 为例：

```yaml
defaults:
  # 1. 选择视觉编码器（ViT）
  - ../../models/image_encoder@model.image_encoder: intern_vit_0.3b

  # 2. 选择图像投影层
  - ../../models/image_projector@model.image_projector: intern_mlp_adapter

  # 3. 选择 LLM 基座
  - ../../models/internlm2.5@model.foundation: internlm2_5_8b

  - _self_

model:
  model_type: intern_vl
  # ... 其他全局参数
```

假设你想要一个结合 **InternViT** 与 **Qwen2.5-7B** 的全新 VLM：

1. **定义组件配置**：确保 `configs/models/` 包含 Qwen2.5 配置，且 `configs/models/image_encoder/` 包含 **InternViT** 配置。
2. **创建组合配置**：创建一个新的 YAML，只需在 `defaults` 中列出所需组件。

```yaml
defaults:
  # 1. 视觉编码器
  # 'intern_vit_0.3b' 指向 configs/models/image_encoder/intern_vit_0.3b.yaml
  - ../../models/image_encoder@model.image_encoder: intern_vit_0.3b

  # 2. 图像投影层
  - ../../models/image_projector@model.image_projector: intern_mlp_adapter

  # 3. LLM 基座
  # 'internlm2_5_8b' 指向 configs/models/internlm2.5/internlm2_5_8b.yaml
  - ../../models/internlm2.5@model.foundation: internlm2_5_8b

  - _self_

model:
  # 多模态模型类型
  model_type: intern_vl

  # 基座模型的损失函数
  loss_func: ${loss_func:loss_func_internvl}

  # 显式设置投影层输出维度以匹配 LLM hidden_size
  image_projector:
    hidden_size: 4096
    ffn_hidden_size: 4096

  # LLM 特定设置
  foundation:
    rotary_emb_func: "DynamicRotaryEmbedding"
    rotary_base: 1000000

    # Megatron 层规范，支持 Transformer-Engine 加速
    model_spec: ["loongforge.models.foundation.internlm.internlm_layer_spec",
                 "get_internlm_layer_with_te_spec"]
```

当前的 `configs/models` 目录已体现这种即插即用的组件库：

* `image_encoder/`：视觉编码器配置
* `image_projector/`：投影层配置
* `llama3/`、`qwen2/`、`deepseek2/` 等：LLM 基座配置

通过这种方式，构建多模态大模型就像拼搭 LEGO 积木一样简单。
