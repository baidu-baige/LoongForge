# Model Checkpoint Conversion for VLM

## 1. Parameter Passing Methods

Supports two parameter passing methods: defining in config files and passing through command line args during conversion

### Config
Supports defining relevant parameters directly in the model config file after model construction, for example:

```yaml
# hydra:
#   searchpath:
#     - file://configs/

defaults:
  - ../../models/image_encoder@model.image_encoder: qwen2.5_vit
  - ../../models/image_projector@model.image_projector: qwen_mlp_adapter
  - ../../models/qwen@model.foundation: qwen2_5_7b
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
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 2
  image_projector:
    activation_func: ${act:gelu}
  image_encoder:
    tensor_model_parallel_size: 2
```

Supports heterogeneous TP. If needed, specify different `tensor_model_parallel_size` in `foundation` and `image_encoder`

### Args
While supporting parameter definition in config files, traditional command line args parameter passing is also retained. However, YAML parameter passing has higher priority than args - args parameters only take effect when split strategies are not specified in YAML

*Currently `num_virtual_stages_per_pipeline_rank` is not supported in YAML and needs to be passed through args

## 2. Common Parameters

| **Parameter Name** | **Explanation** | **Optional Values** | **Default Value** |
|-------------------|-----------------|-------------------|------------------|
| load_platform | Platform to load checkpoint from | `huggingface`, `mcore` | `None` |
| save_platform | Platform to save checkpoint to | `huggingface`, `mcore` | `None` |
| load_ckpt_path | Path to load checkpoint | Any valid path | `None` |
| save_ckpt_path | Path to save checkpoint | Any valid path | `None` |
| common_config_path | Path to common config | Any valid path | `None` |
| megatron_path | Base directory of Megatron repository | Any valid path | `None` |
| no_load_optim | Do not convert optimizer | `True`/`False` (action) | `False` |
| no_save_optim | Do not save optimizer | `True`/`False` (action) | `False` |
| safetensors | Use safetensors | `True`/`False` (action) | `False` |
| config_file | Config file for model configuration | Any valid path | `None` |
| convert_file | Convert file for checkpoint conversion | Any valid path | `None` |
| num_virtual_stages_per_pipeline_rank | Number of virtual pipeline stages per pipeline parallelism rank | Any positive integer | `None` |
| tensor_model_parallel_size | Target tensor model parallel size | Any positive integer | `1` |
| pipeline_model_parallel_size | Target pipeline model parallel size | Any positive integer | `1` |
| data_parallel_size | Target data parallel size | Any positive integer | `1` |
| expert_parallel_size | Target expert parallel size | Any positive integer | `None` |
| expert_tensor_parallel_size | Degree of expert model parallelism | Any positive integer | `None` |
| custom_pipeline_layers | Custom pipeline layer distribution | Comma-separated number string | `None` |
| num_layers_per_virtual_pipeline_stage | Number of layers per virtual pipeline stage | Any positive integer | `None` |
| num_experts | Number of Experts in MoE | Any positive integer | `None` |
| max_workers | Thread for checkpoint converting | Any positive integer | `1` |
| moe-grouped-gemm | Use grouped gemm in moe | `True`/`False` (action) | `False` |
| resume-convert | Resume checkpoint converting when failed | `True`/`False` (action) | `False` |

## 3. Script Examples and Parameter Explanation

Below are conversion scripts for Dense and MoE models with parameter explanations

### Dense 

| **Model** | **Split Strategy** |
|-----------|-------------------|
| **Qwen2.5-VL-7B** | TP=4 PP=2 VPP=2 custom_pipeline_layers 6,8,6,8 |

#### HF -> Mcore
```bash
#!/bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"} # Specify LoongForge path
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"} # Specify Megatron backend path
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint" # convert_checkpoint module path, no modification needed

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/ # Specify target model HF weights path
SAVE=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-tp4-pp2-vpp2-custom-Dec12 # Specify target model converted Mcore weights path

# Specify temporary save paths
SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-mcore # Temporary path for saving language model, will be deleted after conversion
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-mcore # Temporary path for saving vision model, will be deleted after conversion
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-mcore # Temporary path for saving adapter, will be deleted after conversion
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-mcore # Temporary path for saving vision patch, will be deleted after conversion

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5vl/qwen2_5_vl_7b.yaml # Specify model configuration file path after model construction

# Specify checkpoint conversion configuration file paths for each module
FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert.yaml # Specify foundation model checkpoint conversion configuration file path
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen2_5_vit_convert.yaml # Specify image encoder checkpoint conversion configuration file path
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_mlp_adapter_convert.yaml # Specify image projector checkpoint conversion configuration file path

ETP=4 # encoder tp, encoder tensor parallelism degree
DTP=4 # decoder tp, decoder tensor parallelism degree, when ETP and DTP are different, heterogeneous TP is enabled
PP=2 # pipeline parallelism degree
VPP=2 # virtual pipeline parallelism degree per rank, requires PP>1 when enabling virtual pipeline
CUSTOM_PIPELINE_LAYERS=6,8,6,8 # Custom pipeline layer split, note: 1) Number of values in CUSTOM_PIPELINE_LAYERS should equal PPxVPP; 2) Sum of values in CUSTOM_PIPELINE_LAYERS should equal model layers

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --num-virtual-stages-per-pipeline-rank=$VPP \
    --custom_pipeline_layers=$CUSTOM_PIPELINE_LAYERS \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# merge
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py \
    --megatron_path $MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL/release \
    --vision_model_path $SAVE_VISION_MODEL/release \
    --vision_patch $SAVE_PATCH/release \
    --adapter_path $SAVE_ADAPTER/release \
    --encoder_tensor_model_parallel_size $ETP \
    --decoder_tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size $PP \
    --save_ckpt_path $SAVE/release \
    --num_virtual_stages_per_pipeline_rank=$VPP \
    --config_file $MODEL_CONFIG_FILE 

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
```

#### MCore -> HF
```bash
#!/bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

SAVE=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-hf-Dec22 # Final saved HF path
LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-tp4-pp2-vpp2-custom-Original/release # Intermediate temporary result of key mapping, used for LoongForge training, can be deleted if not needed
OMNI_LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-tp4-pp2-vpp2-custom-Dec12/release # Mcore checkpoint path to be converted

SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-hf
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-hf
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-hf
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-hf

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5vl/qwen2_5_vl_7b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen2_5_vit_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_mlp_adapter_convert.yaml

PP=2 
ETP=4 
DTP=4
VPP=2
CUSTOM_PIPELINE_LAYERS=6,8,6,8

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser.py \
  --load_omni_ckpt_path $OMNI_LOAD \
  --save_original_ckpt_path $LOAD \
  --decoder_tensor_model_parallel_size=$DTP \
  --pipeline_model_parallel_size=$PP \
  --num_virtual_stages_per_pipeline_rank=$VPP \
  --config_file $MODEL_CONFIG_FILE

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --num-virtual-stages-per-pipeline-rank=$VPP \
    --custom_pipeline_layers=$CUSTOM_PIPELINE_LAYERS \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# vit
if [[ $PP -eq 1 ]]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$ETP;i++)); do
        from=`printf "mp_rank_%02d_000" $i`
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size $PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size $PP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# merge
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/huggingface/merge_huggingface.py \
    --megatron_path $MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL\
    --vision_model_path $SAVE_VISION_MODEL\
    --vision_patch $SAVE_PATCH\
    --adapter_path $SAVE_ADAPTER\
    --save_ckpt_path $SAVE\

rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
```

### MoE

| **Model** | **Split Strategy** |
|-----------|-------------------|
| **internvl3.5_30b-a3b** | TP=2 PP=2 EP=4 Expert_TP=1 |

#### HF -> Mcore
```bash
#!/bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/mnt/cluster/models/InternVL3_5-30B-A3B
SAVE=/mnt/cluster/LoongForge/internvl3.5/internvl3.5-30b-a3b-tp2-pp2-ep4-etp1-Dec15

SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-mcore
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-mcore
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-mcore
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-mcore

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/internvl3.5/internvl3_5_30b_a3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert_intern.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/internvl_vit_0.3b_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/intern_mlp_adapter_convert.yaml

ETP=2
DTP=2
PP=2
EP=4
Expert_TP=1

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --num_experts=128 \
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$Expert_TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py \ # Due to model construction, internvl needs special adapter conversion script
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# merge
if [ $EP -gt 1 ]; then
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron_expert.py \ # MoE needs different merge script
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --expert_parallel_size=$EP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE 
else
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py \ # MoE needs different merge script
        --megatron_path $MEGATRON_PATH \
        --language_model_path $SAVE_LANGUAGE_MODEL/release \
        --vision_model_path $SAVE_VISION_MODEL/release \
        --vision_patch $SAVE_PATCH/release \
        --adapter_path $SAVE_ADAPTER/release \
        --encoder_tensor_model_parallel_size $ETP \
        --decoder_tensor_model_parallel_size $DTP \
        --pipeline_model_parallel_size $PP \
        --save_ckpt_path $SAVE/release \
        --config_file $MODEL_CONFIG_FILE 
fi

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
```

#### Mcore -> HF
```bash
#!/bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

SAVE=/mnt/cluster/LoongForge/internvl3.5/internvl3.5-30b-a3b-hf-Dec23
LOAD=/mnt/cluster/LoongForge/internvl3.5/internvl3.5-30b-a3b-tp2-pp2-ep4-etp1-Original/release
OMNI_LOAD=/mnt/cluster/LoongForge/internvl3.5/internvl3.5-30b-a3b-tp2-pp2-ep4-etp1-Dec15/release

SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-hf
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-hf
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-hf
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-hf

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/internvl3.5/internvl3_5_30b_a3b.yaml

FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen3/ckpt_convert/qwen3_moe_convert_intern.yaml
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/internvl_vit_0.3b_convert.yaml
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/intern_mlp_adapter_convert.yaml

ETP=2
DTP=2
PP=2
EP=4
Expert_TP=1

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser_expert.py \ # MoE needs different key name conversion script
  --load_omni_ckpt_path $OMNI_LOAD \
  --save_original_ckpt_path $LOAD \
  --decoder_tensor_model_parallel_size=$DTP \
  --pipeline_model_parallel_size=$PP \
  --config_file $MODEL_CONFIG_FILE

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $FOUNDATION_CONVERT_FILE \
    --tensor_model_parallel_size=$DTP \
    --pipeline_model_parallel_size=$PP \
    --num_experts=128 \
    --expert_parallel_size=$EP \
    --expert_tensor_parallel_size=$Expert_TP \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE_LANGUAGE_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# vit: vision model
if [ -z "${EP:-}" ]; then
  EP=1
fi
if [ -z "${Expert_TP:-}" ]; then
  Expert_TP=1
fi
if [ $PP -eq 1 ] && [ $EP -eq 1 ]; then
    LOAD_PATH=$LOAD
else
    LOAD_PATH=$LOAD/tmp/
    mkdir -p $LOAD_PATH
    for ((i=0;i<$ETP;i++)); do
        from=`printf "mp_rank_%02d" $i`
        if [ $PP != 1 ]; then
          from+="_000"
        fi
        if [ $EP != 1 ]; then
          from+=`printf "_%03d" $((i/Expert_TP))`
        fi
        to=`printf "mp_rank_%02d" $i`
        cp -r $LOAD/$from $LOAD_PATH/$to
    done
fi

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_VISION_MODEL \
    --safetensors \
    --no_save_optim \
    --no_load_optim \

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_PROJECTOR_CONVERT_FILE \
    --tensor_model_parallel_size $DTP \
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_ADAPTER \
    --safetensors \
    --no_save_optim \
    --no_load_optim

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    --load_platform=mcore\
    --save_platform=huggingface\
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $IMAGE_ENCODER_CONVERT_FILE \
    --tensor_model_parallel_size=$ETP \
    --pipeline_model_parallel_size 1 \
    --load_ckpt_path=$LOAD_PATH \
    --save_ckpt_path=$SAVE_PATCH \
    --safetensors \
    --no_save_optim \
    --no_load_optim

# merge
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/huggingface/merge_huggingface.py \
    --megatron_path $MEGATRON_PATH \
    --language_model_path $SAVE_LANGUAGE_MODEL\
    --vision_model_path $SAVE_VISION_MODEL\
    --vision_patch $SAVE_PATCH\
    --adapter_path $SAVE_ADAPTER\
    --save_ckpt_path $SAVE\

if [[ $LOAD != $LOAD_PATH ]]; then
    rm -rf $LOAD_PATH
fi

rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
```

## Custom Model Construction

### Define Model Construction File and Conversion File
```yaml
# hydra:
#   searchpath:
#     - file://configs/

defaults:
  - ../../models/image_encoder@model.image_encoder: ${Your image encoder}
  - ../../models/image_projector@model.image_projector: ${Your image projector}
  - ../../models/xxx@model.foundation: ${Your foundation model}
  - _self_

...
```

At the same time, you need to define conversion files corresponding to each component

### HF -> Mcore

Compared with existing models, the difference in converting custom model construction lies in different HF weight paths for different components, which need to be specified separately. Meanwhile, model construction files and conversion configuration files also need to be set accordingly, with no other changes.

```bash
...

LOAD_ENCODER= # HF weight path for image encoder
LOAD_PROJECTOR= # HF weight path for image projector
LOAD_FOUNDATION= # HF weight path for foundation model
SAVE=# Converted Mcore weights save path

...

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/... # Specify custom model construction configuration file path

# Specify checkpoint conversion configuration file paths for each module
FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/.../ckpt_convert/..._convert.yaml # Specify foundation model checkpoint conversion configuration file path
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/..._convert.yaml # Specify image encoder checkpoint conversion configuration file path
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/..._convert.yaml # Specify image projector checkpoint conversion configuration file path

...

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    ...
    --load_ckpt_path=$LOAD_FOUNDATION \
    ...

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    ...
    --load_ckpt_path=$LOAD_ENCODER \
    ...

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter.py \
    ...
    --load_ckpt_path=$LOAD_PROJECTOR \
    ...

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/vision_patch.py \
    ...
    --load_ckpt_path=$LOAD_ENCODER \
    ...

# merge
PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py \
    ...

echo release > $SAVE/latest_checkpointed_iteration.txt
rm -rf $SAVE_LANGUAGE_MODEL
rm -rf $SAVE_VISION_MODEL
rm -rf $SAVE_ADAPTER
rm -rf $SAVE_PATCH
```

### Mcore -> HF
Compared with existing models, there is no change in Mcore to HF conversion, just refer to it directly.