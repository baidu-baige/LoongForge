# Support New Model

This document describes how to support new models in LoongForge, covering **LLM models**, **VLM models**, and **Custom models** (using Wan model as an example). Typically, you only need to add corresponding configuration files and complete registration without modifying core code.

## 1. Supporting LLM Models
### 1.1 Adding New LLM Configuration
If your LLM is a new specification of an existing architecture (e.g., from Llama3-8B to Llama3-70B), simply create a new YAML file.

* **Path**: configs/models/<model_family>/<model_name>.yaml
* **Example**: configs/models/llama3/llama3_70b.yaml:

```yaml
# Inherit the common configuration class for this model family
_target_: loongforge.models.foundation.Llama3Config

# Modify specific parameters
num_layers: 80
hidden_size: 8192
ffn_hidden_size: 28672
num_attention_heads: 64
# ... other parameters
```

### 1.2 Register Model Name
Register in the MODEL_CONFIG_REGISTRY in loongforge/utils/config_map.py, then you can reference the model directly by name (e.g., `llama3-70b`).

```python
MODEL_CONFIG_REGISTRY = {
    # ... existing models
    "llama3-70b": {
            "config_path": "configs/models/llama3",
            "config_name": "llama3_70b",
        },
    }
```

## 2. Supporting VLM Models
VLM can be viewed as **ViT + Projector + LLM**. When adding new VLM models, the LLM part can reuse existing configurations (no need to rewrite LLM details), mainly adding **vision encoder**, **projection layer**, and **VLM combination configuration**. The process to support VLM models is divided into three main steps:

    1. **Prepare component configurations**: Define LLM base, vision encoder, and projector configurations.
    2. **Create combination configuration**: Write the top-level YAML configuration file for VLM.
    3. **Register model name**: Register the new model in config_map.py.

### 2.1 Vision Encoder (ViT) Configuration
Define Vision Transformer parameters.

* **Path**: configs/models/image_encoder/<encoder_name>.yaml 
* Example: configs/models/image_encoder/qwen2_5_vit.yaml

```yaml
# Find the Qwen2VisionRMSNormConfig class through this path, use the following parameters (e.g., num_layers, hidden_size, etc.) to create its instance
_target_: loongforge.models.encoder.Qwen2VisionRMSNormConfig

num_layers: 32
hidden_size: 1280
kv_channels: 80
ffn_hidden_size: 3420
patch_size: 14
num_attention_heads: 16
num_query_groups: 16
image_size: [1344, 1344]
# ... other parameters
```

### 2.2 Projector Configuration
The Projector implementation is interrelated with OmniEncoder. Each type of VLM model is equipped with a dedicated Projector. You need to select the Projector type, and its dimension information will be specified in the model combination configuration.

* **Path**: configs/models/image_projector/<projector_name>.yaml
* **Example**: configs/models/image_projector/qwen_mlp_adapter.yaml

```yaml
# Select image_projector type
_target_: loongforge.models.encoder.MLPAdapterConfig

# Modify component-specific configuration parameters
normalization: "RMSNorm"
add_bias_linear: True
model_type: "qwen2_5_vl_adapter"
```

### 2.3 Create Combination (VLM Top-level YAML) Configuration
This step is the key to defining VLM models. You need to create a YAML file that "assembles" the above components and sets key alignment parameters.

* **Recommended path**: `configs/models/<vlm_family>/<my_new_vlm>.yaml`, content structure:

```yaml
# 1. Use defaults list to import components
defaults:
  # Import Encoder
  - ../../models/image_encoder@model.image_encoder: qwen2_5_vit
  
  # Import Projector
  - ../../models/image_projector@model.image_projector: qwen_mlp_adapter
  
  # Import LLM
  - ../../models/llama3@model.foundation: llama3_8b
  - _self_

model:
  # Define global model parameters
  position_idx_func: ${position_func:mrope_ids}
  loss_func: ${loss_func:default}
  
  # Align foundation model details
  foundation: 
    rotary_base: 1000000
    group_query_attention: true
    
  # Align image_projector details
  image_projector:
    activation_func: ${act:gelu}
```

### 2.4 Model Registration
You need to register in loongforge/utils/config_map.py. Open loongforge/utils/config_map.py and add entries to the MODEL_CONFIG_REGISTRY dictionary:

```python
MODEL_CONFIG_REGISTRY = {
    # ... existing models
    
    # === Add your new model ===
    "my-custom-vlm-8b": {
        "config_path": "configs/models/<vlm_family>",       # Directory where the combination configuration file is located
        "config_name": "my_new_vlm",                        # Combination configuration file name (without .yaml)
    },
}
```

After successful registration, you can reference the model directly by name (e.g., `my-custom-vlm-8b`).

## 3. Supporting Custom Models (Using Wan as Example)
Wan series model configurations are located in `configs/models/wan/`, for example:

* `configs/models/wan/wan2_2_i2v.yaml`

### 3.1 Adding New Wan Configuration
If it's a new specification or variant of Wan, it's recommended to copy the existing configuration and modify necessary parameters:

```
configs/models/wan/<your_wan_variant>.yaml
```

### 3.2 Register Model Name
```python
MODEL_CONFIG_REGISTRY = {
    # ... existing models
    "my-wan-variant": {
        "config_path": "configs/models/wan",
        "config_name": "<your_wan_variant>",
    },
}
```