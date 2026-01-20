# 4.1 model_combination.md

One of the core features of the AIAK-Training-Omni framework is its native support for **flexible model combination**. This feature allows users to freely combine visual/audio encoders (Encoder), modality alignment projection layers (Projector), and language model bases (Foundation) through configuration files (YAML), thereby quickly building customized multimodal large models. Based on this framework, users only need to modify the model configuration YAML files in the `../configs/models` directory to restructure and switch the model architecture.

For example, different visual encoders (such as **Qwen2.5-ViT, Qwen3-ViT, LLaVA-OV-1.5-ViT, InternViT**, etc.) can be flexibly concatenated with different large language model bases (such as **LLaMA series, DeepSeek series, Qwen series**, etc.). This **configuration-driven model assembly approach** significantly reduces the cost of exploring and customizing multimodal model architectures, allowing users to build models without modifying any code, achieving **zero-code** model combination.

## 1. Core Components
### 1.1 OmniEncoderModel
The core abstraction of the Encoder component, which is responsible for converting modal data such as images, videos, and audio into Embeddings understandable by LLMs.

* **Abstract Definition**: Encapsulates Vision/Audio Encoder and Projector, and manages Text Embeddings uniformly.
* **Core Implementation**:
    * **Multimodal Compatibility**: Uniformly manages Encoders of different modalities. Adding a new modality only requires adding a branch.

```python
# aiak_training_omni/models/omni_models/omni_encoder_model.py
class OmniEncoderModel(torch.nn.Module):
    def __init__(self, config, ...):
        # Text modality
        self.text_encoder = LanguageModelEmbedding(...)
        
        # Image modality
        if hasattr(config, "image_encoder"):
            self.image_encoder = AutoModel.from_config(config.image_encoder, ...)
        
        # Video modality (very easy to extend)
        if hasattr(config, "video_encoder"):
            self.video_encoder = AutoModel.from_config(config.video_encoder, ...)
```
    * **Heterogeneous Tensor Parallel**: Implements heterogeneous parallelism for **Encoder Tensor Parallel** through Hook mechanisms.

```python
# Automatically register Hook to switch Parallel State before and after Forward
self.image_encoder.register_forward_pre_hook(
    make_encoder_forward_pre_hook("image_encoder")
)
self.image_encoder.register_forward_hook(
    make_encoder_forward_hook("text_decoder") # Switch back to Decoder state
)
```

### 1.2 OmniCombinationModel
The core component of multimodal combination models, defining the concatenation and interaction timing of data flow between different modality components. This component does not contain specific computational logic. Key features:

    * **Logical Decoupling**: Dynamically decides whether to load Encoder or Foundation model based on external configuration, achieving component-level decoupling.

```python
# aiak_training_omni/models/omni_models/omni_combination_model.py
class OmniCombinationModel(BaseMegatronModule):
    def __init__(self, config, ...):
        # 1. Dynamically initialize Encoder Model
        if config.image_encoder is not None and add_encoder:
            self.encoder_model = OmniEncoderModel(config, ...)
        
        # 2. Dynamically initialize LLM
        if config.foundation is not None and add_decoder:
            self.foundation_model = AutoModel.from_config(config.foundation, ...)
```

### 1.3 OmniModelProvider
Responsible for parsing global parameters, handling the initialization logic of distributed environments (such as Pipeline Parallel splitting), and ultimately instantiating OmniCombinationModel.

Core functionalities:

* Distributed Pipeline Parallel: Determines the current Pipeline Parallel Stage of the model, and loads Encoder model and Foundation model as needed to support cross-GPU pipeline parallel deployment.

```python
# aiak_training_omni/models/omni_models/omni_model_provider.py
def omni_model_provider(...):
    # Automatically determines whether the current stage is the first stage of the pipeline, deciding whether to load the Encoder
    # This is crucial for placing the Encoder and Decoder on different GPUs for pipeline parallelism
    if args.encoder_pipeline_model_parallel_size in [0, None]:
        add_encoder = mpu.is_pipeline_first_stage()
    
    # Finally builds the model, injecting environment-aware parameters
    return OmniCombinationModel(..., add_encoder=add_encoder, ...)
```
* **Parameter Bridging and Adaptation**: Simultaneously obtains global training parameters (args) and model structure configuration (model_config). Passes model_config to the model for building submodules, while extracting key hyperparameters related to training (such as language_vocab_size, language_max_sequence_length) from args and injecting them into the model initialization process to ensure consistency between model configuration and training environment.

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

## 2. How to Use
We have adopted a configuration system based on Hydra, combining configurations of different components through the defaults list. Take `configs/models/internvl2.5/internvl2_5_8b.yaml` as an example:

```yaml
defaults:
  # 1. Select image encoder (Vit)
  - ../../models/image_encoder@model.image_encoder: intern_vit_0.3b
  
  # 2. Select image projector (Projector)
  - ../../models/image_projector@model.image_projector: intern_mlp_adapter
  
  # 3. Select foundation language model (LLM)
  - ../../models/internlm2.5@model.foundation: internlm2_5_8b
  
  - _self_

model:
  model_type: intern_vl
  # ... Other global parameters
```
If you want to build a new VLM, for example, using **Qwen2.5-7B** as the base and **InternViT** as the encoder to create a new **InternViT-Qwen2.5-7B model**:

1. **Define Component Configurations**: Ensure that there are configurations for Qwen2.5 under `configs/models/` and configurations for **InternViT** under `configs/models/image_encoder/`.
2. **Create Combination Configuration**: Create a new YAML file and combine them through defaults.

```yaml
defaults:
  # 1. Select image encoder (Image Encoder)
  # 'intern_vit_0.3b' corresponds to configs/models/image_encoder/intern_vit_0.3b.yaml (same below)
  - ../../models/image_encoder@model.image_encoder: intern_vit_0.3b
  
  # 2. Select image projector (Projector)
  - ../../models/image_projector@model.image_projector: intern_mlp_adapter
  
  # 3. Select foundation language model (LLM Foundation)
  # 'internlm2_5_8b' corresponds to configs/models/internlm2.5/internlm2_5_8b.yaml
  - ../../models/internlm2.5@model.foundation: internlm2_5_8b
  
  - _self_

model:
  # Specify the type of multimodal model
  model_type: intern_vl
  
  # Specify the loss function of the Foundation model
  loss_func: ${loss_func:loss_func_internvl}
  
  # Explicitly specify the output dimension of the Projector, which must be consistent with the hidden_size of the LLM
  image_projector:
    hidden_size: 4096
    ffn_hidden_size: 4096
  
  # Configure LLM-specific settings
  foundation: 
    rotary_emb_func: "DynamicRotaryEmbedding"
    rotary_base: 1000000
    
    # Specify Megatron layer construction, supporting Transformer Engine acceleration
    model_spec: ["aiak_training_omni.models.foundation.internlm.internlm_layer_spec", "get_internlm_layer_with_te_spec"]
```

The current directory structure of `configs/models` in the framework corresponds to the library of combinable components:

* `image_encoder/`: Stores configurations for various visual encoders.
* `image_projector/`: Stores configurations for various projection layers.
* `llama3/`, `qwen2/`, `deepseek2/`, etc.: Store configurations for various LLM bases.

With this approach, you can flexibly build multimodal large models like building with blocks.