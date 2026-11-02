# Model Combination

One of the core features of the **LoongForge** framework is **native support for flexible network assembly**.  
With a simple YAML configuration file, users can freely combine vision/audio encoders, modality-alignment projectors, and language-model backbones to rapidly build customized multimodal large models.  
By editing only the model-configuration YAML files under `../configs/models`, the entire model structure can be re-assembled or switched without touching a single line of code.

For example, different vision encoders (**Qwen2.5-ViT, Qwen3-ViT, LLaVA-OV-1.5-ViT, InternViT**, etc.) can be stitched to different LLM backbones (**LLaMA family, DeepSeek family, Qwen family**, etc.).  
This **configuration-driven assembly** dramatically lowers the cost of exploring and customizing multimodal architectures, enabling **zero-code** model construction.

---

## 1. Core Components

### OmniEncoderModel
Abstract base class for all encoder components.  
It converts image, video, audio or other modality data into embeddings that the LLM can understand.

* **Abstraction**: Encapsulates Vision/Audio Encoders and Projectors, and unifies text-embedding management.  
* **Key implementations**:
  * **Multimodal compatibility**: New modalities are added by simply inserting a new branch.

```python
# loongforge/models/omni_models/omni_encoder_model.py
class OmniEncoderModel(torch.nn.Module):
    def __init__(self, config, ...):
        # text modality
        self.text_encoder = LanguageModelEmbedding(...)

        # image modality
        if hasattr(config, "image_encoder"):
            self.image_encoder = AutoModel.from_config(config.image_encoder, ...)

        # video modality (trivial to extend)
        if hasattr(config, "video_encoder"):
            self.video_encoder = AutoModel.from_config(config.video_encoder, ...)
```

  * **Heterogeneous Tensor Parallel**: Uses hook mechanism to realize **Encoder Tensor Parallel** across heterogeneous devices.

```python
# Automatically register hooks to switch parallel state before/after forward
self.image_encoder.register_forward_pre_hook(
    make_encoder_forward_pre_hook("image_encoder")
)
self.image_encoder.register_forward_hook(
    make_encoder_forward_hook("text_decoder")  # switch back to decoder state
)
```

---

### OmniCombinationModel
Core component for multimodal composition.  
Defines **when and how** data flows among modality components; contains **no actual compute logic**.

* **Logical decoupling**: Dynamically decides whether to load an encoder or a foundation model via external config, achieving component-level decoupling.

```python
# loongforge/models/omni_models/omni_combination_model.py
class OmniCombinationModel(BaseMegatronModule):
    def __init__(self, config, ...):
        # 1. Dynamically init Encoder Model
        if config.image_encoder is not None and add_encoder:
            self.encoder_model = OmniEncoderModel(config, ...)

        # 2. Dynamically init LLM
        if config.foundation is not None and add_decoder:
            self.foundation_model = AutoModel.from_config(config.foundation, ...)
```

---

### OmniModelProvider
Parses global arguments, handles distributed initialization (e.g., Pipeline-Parallel splitting), and finally instantiates `OmniCombinationModel`.

Key capabilities:

* **Distributed Pipeline Parallel**: Detects the current Pipeline-Parallel stage and loads the encoder/foundation model on demand, enabling **cross-GPU pipelined deployment**.

```python
# loongforge/models/omni_models/omni_model_provider.py
def omni_model_provider(...):
    # Auto-detect if current rank is the first PP stage; decide whether to load encoder
    # This is critical for placing encoder and decoder on different GPUs
    if args.encoder_pipeline_model_parallel_size in [0, None]:
        add_encoder = mpu.is_pipeline_first_stage()

    # Build the model with environment-aware flags
    return OmniCombinationModel(..., add_encoder=add_encoder, ...)
```

* **Parameter bridging & adaptation**: Fetches both global training args and model-config, injects training-critical hyper-parameters (`language_vocab_size`, `language_max_sequence_length`, etc.) into the model initialization to guarantee consistency between model config and training environment.

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

## 2. How to use

We adopt a **Hydra-based** configuration system that composes components via the `defaults` list.  
Take `configs/models/internvl2.5/internvl2_5_8b.yaml` as an example:

```yaml
defaults:
  # 1. pick vision encoder (ViT)
  - ../../models/image_encoder@model.image_encoder: intern_vit_0.3b

  # 2. pick image projector
  - ../../models/image_projector@model.image_projector: intern_mlp_adapter

  # 3. pick LLM backbone
  - ../../models/internlm2.5@model.foundation: internlm2_5_8b

  - _self_

model:
  model_type: intern_vl
  # ... other global params
```

Suppose you want a brand-new VLM that combines **InternViT** with **Qwen2.5-7B**:

1. **Define component configs**: make sure `configs/models/` contains Qwen2.5 config and `configs/models/image_encoder/` contains **InternViT** config.  
2. **Create a composition config**: create a new YAML that simply lists the desired components in `defaults`.

```yaml
defaults:
  # 1. Image Encoder
  # 'intern_vit_0.3b' points to configs/models/image_encoder/intern_vit_0.3b.yaml
  - ../../models/image_encoder@model.image_encoder: intern_vit_0.3b

  # 2. Image Projector
  - ../../models/image_projector@model.image_projector: intern_mlp_adapter

  # 3. LLM Foundation
  # 'internlm2_5_8b' points to configs/models/internlm2.5/internlm2_5_8b.yaml
  - ../../models/internlm2.5@model.foundation: internlm2_5_8b

  - _self_

model:
  # multimodal model type
  model_type: intern_vl

  # loss function for the foundation model
  loss_func: ${loss_func:loss_func_internvl}

  # explicitly set projector output dim to match LLM hidden_size
  image_projector:
    hidden_size: 4096
    ffn_hidden_size: 4096

  # LLM-specific settings
  foundation:
    rotary_emb_func: "DynamicRotaryEmbedding"
    rotary_base: 1000000

    # Megatron layer spec, supports Transformer-Engine acceleration
    model_spec: ["loongforge.models.foundation.internlm.internlm_layer_spec",
                 "get_internlm_layer_with_te_spec"]
```

The current `configs/models` directory already mirrors this plug-and-play component library:

* `image_encoder/`: vision encoder configs  
* `image_projector/`: projector configs  
* `llama3/`, `qwen2/`, `deepseek2/`, etc.: LLM backbone configs  

With this approach, building a multimodal large model is as easy as snapping LEGO bricks together.
