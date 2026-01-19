# 5.2 support_new_model.md

This document will guide you on how to add support for a new model in the AIAK-Training-Omni framework. Thanks to the flexible networking design of the framework, in most cases, you only need to add configuration files and complete the registration without modifying the core code.

The process of supporting a new model is mainly divided into three steps:

1. **Prepare Component Configurations**: Define the configurations for the LLM base, visual encoder, and projector.
2. **Create Combination Configuration**: Write the top-level YAML configuration file for the VLM.
3. **Register Model Name**: Register the new model in `config_map.py`.

# Step 1: Prepare Component Configurations

## 1. Foundation Model (LLM)
If your LLM is a new specification of an existing architecture (e.g., from Llama3-8B to Llama3-16B), simply create a new YAML file.

* **Path**: `configs/models/<model_family>/<model_name>.yaml` (Example: `configs/models/llama3/llama3_custom_16b.yaml`):

```yaml
# Inherit the common configuration class for this model family
_target_: aiak_training_omni.models.foundation.Llama3Config

# Modify specific parameters
num_layers: 40
hidden_size: 5120
ffn_hidden_size: 13824
num_attention_heads: 40
max_sequence_length: 8192
# ... other parameters
```

## 2. Visual/Audio Encoder
Define the parameters for the Vision Transformer.

* **Path**: `configs/models/image_encoder/<encoder_name>.yaml` (Example: `configs/models/image_encoder/llava_vit.yaml`):

```yaml
# Inherit the common configuration class for this model family
_target_: aiak_training_omni.models.encoder.RiceVisionConfig

num_layers: 24
hidden_size: 1024
ffn_hidden_size: 4096
num_attention_heads: 16
patch_size: 14
image_size: [1344, 1344]
kv_channels: 64
# ... other parameters
```

## 3. Projector
The projector layer binds the implementation of OmniEncoder with the Foundation model. Each VLM model has a corresponding Projector implementation.

(Usually, you only need to select the type, and the dimensions are specified in the combination configuration).

* **Path**: `configs/models/image_projector/<projector_name>.yaml`:

```yaml
# Select the image_projector type
_target_: aiak_training_omni.models.encoder.MLPAdapterConfig

# Modify the specific configuration parameters for this component
normalization: "RMSNorm"
add_bias_linear: True
model_type: "qwen2_5_vl_adapter"
```

# Step 2: Create Combination Configuration
This step is crucial for defining the VLM model. You need to create a YAML file that "assembles" the components together and sets key alignment parameters.

* **Suggested Path**: `configs/models/<vlm_family>/<my_new_vlm>.yaml`, content structure:

```yaml
# 1. Use the defaults list to introduce components
defaults:
  # Introduce Encoder
  - ../../models/image_encoder@model.image_encoder: llava_vit
  
  # Introduce Projector
  - ../../models/image_projector@model.image_projector: qwen_mlp_adapter
  
  # Introduce LLM
  - ../../models/qwen2.5@model.foundation: qwen2_5_7b
  - _self_

model:
  # Define global model parameters
  model_type: qwen2_5_vl
  position_idx_func: ${position_func:mrope_ids}
  loss_func: ${loss_func:default}
  mix_used_vision_encoder: true
  mix_used_vision_projector: true
  
  # Align foundation model details
  foundation: 
    rotary_emb_func: "Qwen2VLRotaryEmbedding"
    model_spec: ["aiak_training_omni.models.foundation.qwen2.qwen_layer_spec", "get_qwen2_vl_layer_with_te_spec"]
    rotary_base: 1000000
    group_query_attention: true
    
  # Align image_projector details
  image_projector:
    activation_func: ${act:gelu}
```

# Step 3: Model Registration
To enable the training script to find your configuration file via the `--model-name` parameter, you need to register it in `aiak_training_omni/utils/config_map.py`. Open `aiak_training_omni/utils/config_map.py` and add an entry to the `MODEL_CONFIG_REGISTRY` dictionary:

```python
MODEL_CONFIG_REGISTRY = {
    # ... existing models
    
    # === Add your new model ===
    "my-custom-vlm-16b": {
        "config_path": "configs/models/model_family",      # Directory where the combination configuration file is located
        "config_name": "my_new_vlm",                 # Combination configuration file name (without .yaml)
    },
}
```

# Verification
After completing the above three steps, you can start training with the new model:

```bash
# Launch the training script
torchrun ... train.py \
    --model-name my-custom-vlm-16b \
    ...
```

**Troubleshooting Tips**:

* If you encounter `KeyError: 'my-custom-vlm-16b'`: Check if `config_map.py` is saved and if the name spelling is consistent.
* If you encounter a projector dimension mismatch error: Check if the `image_projector.hidden_size` in the combination YAML matches the `hidden_size` of the LLM.