# Checkpoint Conversion for LLM

## 1. Common Parameters
When performing LLM weight conversion, it is recommended to pass parameters using arguments (args). The following are commonly used parameters:

|**Parameter**|**Description**|
|-|-|
|load_platform|platform to load checkpoint from|
|save_platform|platform to save checkpoint to|
|config_file|Config file for model configuration|
|convert_file|Convert file for checkpoint conversion|
|tensor_model_parallel_size|target tensor model parallel size|
|pipeline_model_parallel_size|target pipeline model parallel size|
|expert_parallel_size|target expert parallel size|
|expert_tensor_parallel_size|Degree of expert model parallelism|
|megatron_path|Base directory of Megatron repository|
|load_ckpt_path|path to load checkpoint|
|save_ckpt_path|path to save checkpoint|
|custom_pipeline_layers|custom pipeline layer distribution|
|safetensors|Use safetensors|
|max_workers|thread for checkpoint converting|
|moe-grouped-gemm|use grouped gemm in moe|
|amax_epsilon|Epsilon value for amax calculation in FP8 conversion; used for FP8 quantization scale, aligned with the FP8 EPS environment variable set during training. Applicable to both `te` and `pt` methods.|
|quant_method|The quantization method to use. Choices: [te, pt], defaults to `te`.|
|force_pow_2_scales|When True (default), uses power-of-2 scaling for FP8 quantization (matching DeepGEMM's get_e4m3_sf_and_sf_inv). When False, uses linear scaling. Applicable to both `te` and `pt` methods.|
|fp8_force_no_requant|skip dequantize + re-quantize in FP8 conversion|

For descriptions of other parameters, please refer to [checkpoint_convert.md](https://loongforge.readthedocs.io/en/latest/llm_tutorial/checkpoint_convert.html).

## 2. Example Scripts
The framework provides weight conversion example scripts for each model. Users can find specific scripts under `configs/models/{model}/ckpt_convert/`.

Below is an example script for converting **DeepSeek V3.1** model weights from **Huggingface FP8** format to **MegatronCore FP8** format. When converting FP8 format weights, it is crucial to set the `amax_epsilon` parameter. This parameter must align with the FP8 EPS environment variables set during training (`export FP8_QUANT_FWD_INP_AMAX_EPS`, `export FP8_QUANT_FWD_WEIGHT_AMAX_EPS`, `export FP8_QUANT_BWD_GRAD_AMAX_EPS`).

If the user is using the **Nvidia B-series GPUs**, the `--quant_method pt` parameter must be added to the conversion script.

```bash
#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/path/to/hf_checkpoint  # the original DeepSeek-V3 checkpoint is FP8 format
SAVE=/path/to/your/save  # the converted checkpoint will be in MCore FP8 format

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/deepseek3/deepseek_v3.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/deepseek3/ckpt_convert/deepseek_v3_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=huggingface \
    --save_platform=mcore \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --expert_parallel_size=32 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --custom_pipeline_layers 8,7,8,8,8,8,8,6 \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    --amax_epsilon=1e-12 \
    # --quant_method pt
```

Below is an example script for converting **DeepSeek V3.1** model weights from **MegatronCore FP8** format to **Huggingface FP8** format:

> **Note**: When converting from BF16/FP32 source to FP8 target, `--fp8_force_no_requant` is necessary to avoid dequantize + re-quantize. On Nvidia B-series GPUs, `--quant_method pt` must also be added.

```bash
#! /bin/bash

export LOONGFORGE_PATH=${LOONGFORGE_PATH:-"/workspace/LoongForge"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Loong-Megatron"}
CONVERT_CHECKPOINT_PATH="$LOONGFORGE_PATH/tools/convert_checkpoint"

LOAD=/path/to/mcore_checkpoint  # the converted checkpoint will be in MCore FP8 format
SAVE=/path/to/your/save  # the original DeepSeek-V3 checkpoint is FP8 format

MODEL_CONFIG_FILE=${LOONGFORGE_PATH}/configs/models/deepseek3/deepseek_v3.yaml
CONVERT_FILE=${LOONGFORGE_PATH}/configs/models/deepseek3/ckpt_convert/deepseek_v3_convert.yaml

PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH \
    python $CONVERT_CHECKPOINT_PATH/module_convertor/model.py \
    --load_platform=mcore \
    --save_platform=huggingface \
    --config_file $MODEL_CONFIG_FILE \
    --convert_file $CONVERT_FILE \
    --tensor_model_parallel_size=8 \
    --pipeline_model_parallel_size=8 \
    --expert_parallel_size=32 \
    --expert_tensor_parallel_size=1 \
    --megatron_path=$MEGATRON_PATH \
    --load_ckpt_path=$LOAD \
    --save_ckpt_path=$SAVE \
    --custom_pipeline_layers 8,7,8,8,8,8,8,6 \
    --safetensors \
    --max_workers=32 \
    --moe-grouped-gemm \
    --fp8_force_no_requant \
    --amax_epsilon=1e-12 \
    # --quant_method pt
```