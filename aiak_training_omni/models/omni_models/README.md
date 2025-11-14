# AIAK-Omni 多模态训练框架

## 概述

AIAK-Omni 是一个灵活的多模态训练框架，专为构建和训练多模态大语言模型（MLLM）而设计。框架采用模块化架构，支持图像、视频、音频等多种模态的编码、理解和生成任务。

## 核心特性

- **模块化设计**: 采用 Encoder-Foundation-Decoder 三层架构，各模块可独立配置和替换
- **多模态支持**: 原生支持图像、视频、音频等多种模态
- **灵活配置**: 通过 JSON 配置文件组合不同的模型组件
- **动态构建**: 支持运行时动态加载和构建模型组件
- **训练友好**: 提供完整的训练接口和损失计算
- **扩展性强**: 易于添加新的模态和模型类型

## 架构设计

### 整体架构

```
输入数据 → Encoder → Foundation Model → Decoder → 输出
   ↓         ↓           ↓              ↓
 多模态    特征编码    语言理解        模态生成
```

### 核心组件

1. **Encoder（编码器）**: 将多模态输入转换为统一的特征表示
2. **Foundation Model（基础模型）**: 基于 Transformer 的语言模型，处理文本和特征
3. **Decoder（解码器）**: 将语言模型输出转换为特定模态的生成结果

## 目录结构

```
omni/
├── __init__.py                    # 模块导出
├── base_mixins.py                 # 基础抽象类
├── configuration.py               # 配置类定义
├── omni_encoder_model.py          # 编码器主干组件实现
├── omni_foundation_model.py       # FoundationModel主干组件实现
├── omni_decoder_model.py          # 解码器主干组件实现
├── omni_combination_model.py      # Combination主干组件
├── omni_model_provider.py         # 模型provider接口
├── projector.py                   # 特征Adapter
└── examples/                      # 配置示例
    ├── intern_vl_omni.json
    └── qwen2_vl_omni.json
```

## 核心模块详解

### 1. 配置类 (`configuration.py`)

提供灵活的配置管理，支持嵌套配置和动态初始化。

#### 主要配置类

- **`OmniConfig`**: 主配置类，包含编码器、基础模型、解码器配置
- **`OmniEncoderConfig`**: 编码器配置，支持图像、视频、音频子配置
- **`OmniDecoderConfig`**: 解码器配置，支持多模态输出

#### 配置示例

```python
config = OmniConfig(
    encoder_config={
        "image_config": {
            "model_type": "qwen2_vl_vision_model",
            "hidden_size": 1536,
            "output_size": 4096,
            "add_projector": True
        }
    },
    foundation_config={
        "model_type": "qwen2_vl_foundation",
        "vocab_size": 152064,
        "hidden_size": 4096,
        "num_layers": 32
    },
    decoder_config={
        "model_type": "aiak_omni_decoder"
    }
)
```

### 2. 基础抽象类 (`base_mixins.py`)

定义了所有组件的统一接口，确保模块间的兼容性。

#### 核心抽象类

- **`BaseEncoderModelMixin`**: 编码器基类
  - `lm_encode()`: 特征编码接口
  - `lm_dummy_encode()`: 虚拟编码（保持梯度连通性）
  - `set_projector_trainable_only()`: 仅训练Adapter

- **`BaseFoundationModelMixin`**: 基础模型基类
  - `get_generation_position_id()`: 获取生成位置ID
  - `get_position_id_func()`: 获取位置ID函数

- **`BaseDecoderModelMixin`**: 解码器基类
  - `forward_loss()`: 计算解码损失

### 3. 编码器模型 (`omni_encoder_model.py`)

负责将多模态输入转换为统一的特征表示。

#### 主要功能

- **多模态编码**: 支持图像、视频、音频同时编码
- **特征融合**: 将不同模态特征映射到统一空间
- **动态构建**: 根据配置动态加载编码器组件
- **参数冻结**: 支持选择性冻结特定模态编码器

#### 使用示例

```python
encoder = OmniEncoderModel(config.encoder_config)

# 编码多模态输入
input_embeds, decoder_inputs = encoder(
    input_ids=text_ids,
    image_inputs=image_data,
    video_inputs=video_data,
    audio_inputs=audio_data
)

# 冻结特定模态
encoder.freeze(modality="image", freeze_text_encoder=True)
```

### 4. 基础模型 (`omni_foundation_model.py`)

基于现有 LLM 的适配器，提供统一的语言理解接口。

#### 主要功能

- **LLM 适配**: 适配现有的语言模型（Qwen、LLaMA 等）
- **损失计算**: 支持文本生成损失计算
- **生成支持**: 提供文本生成相关接口
- **位置编码**: 支持自定义位置编码策略

#### 使用示例

```python
foundation = OmniFoundationModel(config.foundation_config)

# 前向传播
outputs = foundation(
    inputs_embeds=input_embeds,
    attention_mask=attention_mask,
    labels=labels
)

# 计算损失
loss, hidden_states = foundation.forward_loss(
    inputs_embeds=input_embeds,
    labels=labels
)
```

### 5. 解码器模型 (`omni_decoder_model.py`)

将语言模型输出转换为特定模态的生成结果。

#### 主要功能

- **多模态解码**: 支持图像、视频、音频生成
- **损失计算**: 计算各模态的生成损失
- **参数冻结**: 支持选择性冻结解码器

#### 使用示例

```python
decoder = OmniDecoderModel(config.decoder_config)

# 计算解码损失
losses = decoder.forward_loss(
    decoder_inputs=decoder_inputs,
    foundation_outputs=foundation_outputs
)
```

### 6. 组合模型 (`omni_combination_model.py`)

编排整个 Encoder-Foundation-Decoder 的执行流程。

#### 主要功能

- **流程编排**: 管理三个阶段的执行顺序
- **灵活执行**: 支持跳过特定阶段（离线模式）
- **损失聚合**: 汇总各阶段的损失
- **训练控制**: 提供细粒度的训练控制

#### 执行模式

1. **完整模式**: `encoder → foundation → decoder`
2. **离线编码器**: 使用预处理的 `inputs_embeds`
3. **离线基础模型**: 使用预处理的 `output_embeddings`
4. **仅解码器**: 冻结编码器和基础模型

#### 使用示例

```python
model = OmniCombinationModel(config)

# 完整前向传播
outputs = model(
    input_ids=text_ids,
    image_inputs=image_data,
    labels=labels,
    run_encoder=True,
    run_foundation=True,
    run_decoder=True,
    use_text_loss=True,
    use_decoder_loss=True
)

# 离线模式
outputs = model(
    inputs_embeds=precomputed_embeds,
    output_embeddings=precomputed_outputs,
    run_encoder=False,
    run_foundation=False,
    run_decoder=True
)
```

### 7. 模型Provider (`omni_model_provider.py`)

提供与现有训练框架的集成接口。

#### 主要功能

- **模型构建**: 从命令行参数构建模型
- **训练集成**: 提供训练步骤函数
- **框架兼容**: 与 Megatron 等训练框架兼容

#### 使用示例

```python
# 注册模型provider
@register_model_provider(model_family=[VisionLanguageModelFamilies.QWEN2_VL])
def omni_vlm_model_provider():
    return omni_model_provider()

# 训练步骤函数
def forward_step_func(data_iterator, model):
    batch = get_batch(data_iterator)
    outputs = model(**batch)
    return outputs['text_loss'], outputs['decoder_losses']
```

## 配置示例

### Qwen2-VL 配置

```json
{
  "model_type": "aiak_omni",
  "encoder_config": {
    "model_type": "aiak_omni_encoder",
    "image_config": {
      "model_type": "qwen2_vl_vision_model",
      "hidden_size": 1536,
      "output_size": 4096,
      "add_projector": true,
      "patch_size": 14,
      "image_size": 448,
      "num_layers": 24,
      "num_attention_heads": 16
    }
  },
  "foundation_config": {
    "model_type": "qwen2_vl_foundation",
    "vocab_size": 152064,
    "hidden_size": 4096,
    "num_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "image_token_id": 151644,
    "vision_start_token_id": 151645,
    "spatial_merge_size": 2
  },
  "decoder_config": {
    "model_type": "aiak_omni_decoder"
  },
  "initializer_range": 0.02
}
```

### InternVL 配置

```json
{
  "model_type": "aiak_omni",
  "encoder_config": {
    "model_type": "aiak_omni_encoder",
    "image_config": {
      "model_type": "intern_vl_vision_model",
      "hidden_size": 1536,
      "output_size": 4096,
      "add_projector": true,
      "patch_size": 14,
      "image_size": 224,
      "num_layers": 24,
      "num_attention_heads": 16,
      "vision_type": "vit_300m"
    }
  },
  "foundation_config": {
    "model_type": "internlm2_foundation",
    "vocab_size": 103168,
    "hidden_size": 4096,
    "num_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0
  },
  "decoder_config": {
    "model_type": "aiak_omni_decoder"
  },
  "initializer_range": 0.02
}
```

## 使用指南

### 1. 基本使用

```python
from aiak_training_llm.models.omni import OmniCombinationModel, OmniConfig

# 加载配置
config = OmniConfig.from_json_file("config.json")

# 创建模型
model = OmniCombinationModel(config)

# 前向传播
outputs = model(
    input_ids=text_ids,
    image_inputs=image_data,
    labels=labels
)
```

### 2. 自定义编码器

```python
from aiak_training_llm.models.omni import BaseEncoderModelMixin

class CustomEncoder(BaseEncoderModelMixin):
    def __init__(self, config):
        super().__init__(config)
        # 自定义实现
    
    def lm_encode(self, features, **kwargs):
        # 实现编码逻辑
        return {"features": encoded_features}
    
    def set_projector_trainable_only(self):
        # 实现参数冻结逻辑
        pass
```

### 3. 训练配置

```python
# 冻结编码器，仅训练基础模型
model.encoder_model.freeze()

# 仅训练Adapter
model.encoder_model.set_projector_trainable_only()

# 分阶段训练
# 阶段1: 训练编码器Adapter
model.encoder_model.set_projector_trainable_only()
model.foundation_model.freeze()

# 阶段2: 端到端微调
for param in model.parameters():
    param.requires_grad = True
```

## 扩展开发

### 添加新模态

1. **实现编码器**:
```python
class AudioEncoder(BaseEncoderModelMixin):
    def lm_encode(self, audio_features, **kwargs):
        # 音频编码逻辑
        return {"features": audio_embeds}
```

2. **更新配置**:
```python
config = OmniConfig(
    encoder_config={
        "audio_config": {
            "model_type": "custom_audio_encoder",
            "module_path": "your_module.audio_encoder",
            "class_name": "AudioEncoder"
        }
    }
)
```
