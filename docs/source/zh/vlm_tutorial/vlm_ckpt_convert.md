# VLM 模型权重转换

## 1. 参数传递方式

支持两种参数传递方式：在配置文件中定义和通过命令行参数（args）在转换时传递

### Config 方式
支持在模型配置文件中构建模型后直接定义相关参数，例如：

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

支持异构 TP。如需要，可在 `foundation` 和 `image_encoder` 中指定不同的 `tensor_model_parallel_size`

### Args 方式
在支持配置文件参数定义的同时，也保留了传统的命令行参数传递方式。但 YAML 参数传递的优先级高于 args -- 当 YAML 中未指定切分策略时，args 参数才会生效

*目前 `num_virtual_stages_per_pipeline_rank` 不支持在 YAML 中配置，需要通过 args 传递

## 2. 常用参数

| **参数名** | **说明** | **可选值** | **默认值** |
|------------|---------|-----------|-----------|
| load_platform | 加载权重的平台 | `huggingface`, `mcore` | `None` |
| save_platform | 保存权重的平台 | `huggingface`, `mcore` | `None` |
| load_ckpt_path | 加载权重路径 | 任意有效路径 | `None` |
| save_ckpt_path | 保存权重路径 | 任意有效路径 | `None` |
| common_config_path | 通用配置路径 | 任意有效路径 | `None` |
| megatron_path | Megatron 仓库根目录 | 任意有效路径 | `None` |
| no_load_optim | 不转换优化器 | `True`/`False`（action） | `False` |
| no_save_optim | 不保存优化器 | `True`/`False`（action） | `False` |
| safetensors | 使用 safetensors 格式 | `True`/`False`（action） | `False` |
| config_file | 模型配置文件 | 任意有效路径 | `None` |
| convert_file | 权重转换配置文件 | 任意有效路径 | `None` |
| num_virtual_stages_per_pipeline_rank | 每个流水线并行 rank 的虚拟阶段数 | 任意正整数 | `None` |
| tensor_model_parallel_size | 目标张量并行大小 | 任意正整数 | `1` |
| pipeline_model_parallel_size | 目标流水线并行大小 | 任意正整数 | `1` |
| data_parallel_size | 目标数据并行大小 | 任意正整数 | `1` |
| expert_parallel_size | 目标专家并行大小 | 任意正整数 | `None` |
| expert_tensor_parallel_size | 专家张量并行度 | 任意正整数 | `None` |
| custom_pipeline_layers | 自定义流水线层分配 | 逗号分隔的数字字符串 | `None` |
| num_layers_per_virtual_pipeline_stage | 每个虚拟流水线阶段的层数 | 任意正整数 | `None` |
| num_experts | MoE 中的专家数量 | 任意正整数 | `None` |
| max_workers | 权重转换线程数 | 任意正整数 | `1` |
| moe-grouped-gemm | 在 MoE 中使用 grouped gemm | `True`/`False`（action） | `False` |
| resume-convert | 转换失败时恢复继续 | `True`/`False`（action） | `False` |

## 3. 脚本示例与参数说明

以下是 Dense 和 MoE 模型的转换脚本及参数说明

### Dense 模型

| **模型** | **切分策略** |
|---------|-------------|
| **Qwen2.5-VL-7B** | TP=4 PP=2 VP=2 custom_pipeline_layers 6,8,6,8 |

#### HF -> Mcore
```bash
#!/bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"} # 指定 LoongForge 路径
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"} # 指定 Megatron 后端路径
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint" # convert_checkpoint 模块路径，无需修改

LOAD=/mnt/cluster/huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/ # 指定目标模型 HF 权重路径
SAVE=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-tp4-pp2-vpp2-custom-Dec12 # 指定目标模型转换后的 Mcore 权重路径

# 指定临时保存路径
SAVE_LANGUAGE_MODEL=/mnt/cluster/LoongForge/tmp/language-mcore # 语言模型临时保存路径，转换完成后将删除
SAVE_VISION_MODEL=/mnt/cluster/LoongForge/tmp/vision-model-mcore # 视觉模型临时保存路径，转换完成后将删除
SAVE_ADAPTER=/mnt/cluster/LoongForge/tmp/adapter-mcore # adapter 临时保存路径，转换完成后将删除
SAVE_PATCH=/mnt/cluster/LoongForge/tmp/patch-mcore # 视觉 patch 临时保存路径，转换完成后将删除

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5vl/qwen2_5_vl_7b.yaml # 指定模型构建后的配置文件路径

# 指定各模块的权重转换配置文件路径
FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/qwen2.5/ckpt_convert/qwen2_5_convert.yaml # 指定语言模型基座权重转换配置文件路径
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/qwen2_5_vit_convert.yaml # 指定视觉编码器权重转换配置文件路径
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/qwen_mlp_adapter_convert.yaml # 指定投影层权重转换配置文件路径

ETP=4 # encoder tp，视觉编码器张量并行度
DTP=4 # decoder tp，解码器张量并行度，当 ETP 和 DTP 不同时启用异构 TP
PP=2 # 流水线并行度
VPP=2 # 每个 rank 的虚拟流水线并行度，启用虚拟流水线时需要 PP>1
CUSTOM_PIPELINE_LAYERS=6,8,6,8 # 自定义流水线层切分，注意：1) CUSTOM_PIPELINE_LAYERS 中的数值个数应等于 PPxVPP；2) CUSTOM_PIPELINE_LAYERS 中的数值之和应等于模型层数

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

# 合并
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

SAVE=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-hf-Dec22 # 最终保存的 HF 路径
LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-tp4-pp2-vpp2-custom-Original/release # 键映射的中间临时结果，用于 LoongForge 训练，不需要时可删除
OMNI_LOAD=/mnt/cluster/LoongForge/qwen2_5-vl/qwen2_5-vl-7b-tp4-pp2-vpp2-custom-Dec12/release # 待转换的 Mcore 权重路径

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

# 合并
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

### MoE 模型

| **模型** | **切分策略** |
|---------|-------------|
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
    python $CONVERT_CHECKPOINT_PATH/module_convertor/adapter_internvl.py \ # 由于模型构建方式不同，internvl 需要特殊的 adapter 转换脚本
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

# 合并
if [ $EP -gt 1 ]; then
    PYTHONPATH=$MEGATRON_PATH:$LOONGFORGE_PATH:$PYTHONPATH \
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron_expert.py \ # MoE 需要使用不同的合并脚本
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
        python $CONVERT_CHECKPOINT_PATH/mcore/merge_megatron.py \ # MoE 需要使用不同的合并脚本
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
  python $CONVERT_CHECKPOINT_PATH/key_mappings/key_reverser_expert.py \ # MoE 需要使用不同的键名转换脚本
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

# vit：视觉模型
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

# 合并
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

## 自定义模型构建

### 定义模型构建文件与转换文件
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

同时需要定义与各组件对应的转换文件

### HF -> Mcore

与已有模型相比，自定义模型构建在转换时的区别在于不同组件的 HF 权重路径不同，需要分别指定。同时，模型构建文件和转换配置文件也需要相应设置，其余没有变化。

```bash
...

LOAD_ENCODER= # 视觉编码器的 HF 权重路径
LOAD_PROJECTOR= # 投影层的 HF 权重路径
LOAD_FOUNDATION= # 语言模型基座的 HF 权重路径
SAVE=# 转换后的 Mcore 权重保存路径

...

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/... # 指定自定义模型构建配置文件路径

# 指定各模块的权重转换配置文件路径
FOUNDATION_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/.../ckpt_convert/..._convert.yaml # 指定语言模型基座权重转换配置文件路径
IMAGE_ENCODER_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_encoder/ckpt_convert/..._convert.yaml # 指定视觉编码器权重转换配置文件路径
IMAGE_PROJECTOR_CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/image_projector/ckpt_convert/..._convert.yaml # 指定投影层权重转换配置文件路径

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

# 合并
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
与已有模型相比，Mcore 转 HF 的流程没有变化，直接参照即可。
