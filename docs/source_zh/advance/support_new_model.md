# 支持新模型

本文档介绍如何在 LoongForge 中支持新模型，涵盖 **LLM 模型**、**VLM 模型**和**自定义模型**（以 Wan 模型为例）。通常，你只需添加相应的配置文件并完成注册，无需修改核心代码。

## 1. 支持 LLM 模型
### 1.1 添加新 LLM 配置
如果你的 LLM 是现有架构的新规格（例如从 Llama3-8B 到 Llama3-70B），只需创建新的 YAML 文件。

* **路径**：configs/models/<model_family>/<model_name>.yaml
* **示例**：configs/models/llama3/llama3_70b.yaml：

```yaml
# 继承该模型系列的通用配置类
_target_: loongforge.models.foundation.Llama3Config

# 修改特定参数
num_layers: 80
hidden_size: 8192
ffn_hidden_size: 28672
num_attention_heads: 64
# ... 其他参数
```

### 1.2 注册模型名称
在 `loongforge/models/catalog.py` 的统一 `MODEL_CONFIG_REGISTRY` 的 Megatron
分组中，按模型家族注册，之后即可直接通过名称引用模型（如
`llama3-70b`）。

```python
MODEL_CONFIG_REGISTRY = {
    "megatron": {
        LanguageModelFamilies.LLAMA: {
            # ... 已有模型
            "llama3-70b": "llama3/llama3_70b.yaml",
        },
    },
}
```

## 2. 支持 VLM 模型
VLM 可以看作 **ViT + 投影层 + LLM**。添加新 VLM 模型时，LLM 部分可以复用现有配置（无需重写 LLM 详情），主要添加**视觉编码器**、**模态对齐投影层**和 **VLM 组合配置**。支持 VLM 模型的流程分为三个主要步骤：

    1. **准备组件配置**：定义 LLM 基座、视觉编码器和投影层配置。
    2. **创建组合配置**：编写 VLM 的顶层 YAML 配置文件。
    3. **注册模型名称**：在 catalog.py 中注册新模型。

### 2.1 视觉编码器（ViT）配置
定义 Vision Transformer 参数。

* **路径**：configs/models/image_encoder/<encoder_name>.yaml
* 示例：configs/models/image_encoder/qwen2_5_vit.yaml

```yaml
# 通过此路径找到 Qwen2VisionRMSNormConfig 类，使用以下参数（如 num_layers、hidden_size 等）创建其实例
_target_: loongforge.models.encoder.Qwen2VisionRMSNormConfig

num_layers: 32
hidden_size: 1280
kv_channels: 80
ffn_hidden_size: 3420
patch_size: 14
num_attention_heads: 16
num_query_groups: 16
image_size: [1344, 1344]
# ... 其他参数
```

### 2.2 投影层配置
Projector 的实现与 OmniEncoder 相互关联。每种 VLM 模型配备专用的 Projector。你需要选择 Projector 类型，其维度信息将在模型组合配置中指定。

* **路径**：configs/models/image_projector/<projector_name>.yaml
* **示例**：configs/models/image_projector/qwen_mlp_adapter.yaml

```yaml
# 选择 image_projector 类型
_target_: loongforge.models.encoder.MLPAdapterConfig

# 修改组件特定的配置参数
normalization: "RMSNorm"
add_bias_linear: True
model_type: "qwen2_5_vl_adapter"
```

### 2.3 创建组合（VLM 顶层 YAML）配置
此步骤是定义 VLM 模型的关键。你需要创建一个"组装"上述组件并设置关键对齐参数的 YAML 文件。

* **推荐路径**：`configs/models/<vlm_family>/<my_new_vlm>.yaml`，内容结构：

```yaml
# 1. 使用 defaults 列表导入组件
defaults:
  # 导入 Encoder
  - ../../models/image_encoder@model.image_encoder: qwen2_5_vit

  # 导入 Projector
  - ../../models/image_projector@model.image_projector: qwen_mlp_adapter

  # 导入 LLM
  - ../../models/llama3@model.foundation: llama3_8b
  - _self_

model:
  # 定义全局模型参数
  position_idx_func: ${position_func:mrope_ids}
  loss_func: ${loss_func:default}

  # 对齐基座模型详情
  foundation:
    rotary_base: 1000000
    group_query_attention: true

  # 对齐 image_projector 详情
  image_projector:
    activation_func: ${act:gelu}
```

### 2.4 模型注册
在 `loongforge/models/catalog.py` 的 Megatron 分组中，按对应模型家族添加条目：

```python
MODEL_CONFIG_REGISTRY = {
    "megatron": {
        VisionLanguageModelFamilies.VLM: {
            # === 添加你的新模型 ===
            "my-custom-vlm-8b": "<vlm_family>/my_new_vlm.yaml",
        },
    },
}
```

注册成功后，即可直接通过名称引用模型（如 `my-custom-vlm-8b`）。

### 2.5 添加 Torch 模型

Torch 模型也使用同一个模型目录。Torch YAML 目前只有 `model` 和 `data`
字段，没有 Megatron YAML 使用的 `_target_`；解析器需要这两个类型化配置类
来提供字段类型、默认值和校验。Torch 条目使用三元组
`(yaml_file, model_config_type, data_config_type)`：

```python
MODEL_CONFIG_REGISTRY = {
    "torch": {
        EmbodiedModelFamilies.PI05: {
            "my-torch-model": (
                "my_torch_model/my_torch_model.yaml",
                "loongforge.models.my_torch_model.ModelConfig",
                "loongforge.data.my_torch_model.DataConfig",
            ),
        },
    },
}
```

YAML 中的 `model_type` 必须与 Torch trainer 注册使用的 family 一致。
注册表分组会在解析参数前选择 engine，默认 trainer 再按
`(engine, model_type, training_phase)` 选择。`--trainer-type` 仍是 Torch
可选的按名称覆盖。

## 3. 支持自定义模型（以 Wan 为例）
Wan 系列模型配置位于 `configs/models/wan/`，例如：

* `configs/models/wan/wan2_2_i2v.yaml`

### 3.1 添加新 Wan 配置
如果是 Wan 的新规格或变体，建议复制现有配置并修改必要参数：

```
configs/models/wan/<your_wan_variant>.yaml
```

### 3.2 注册模型名称
```python
MODEL_CONFIG_REGISTRY = {
    "megatron": {
        DiffusionModelFamilies.WAN2_2_I2V: {
            # ... 已有模型
            "my-wan-variant": "wan/<your_wan_variant>.yaml",
        },
    },
}
```
