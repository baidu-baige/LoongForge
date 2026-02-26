# Checkpoint Conversion for LLM

## 1. Supported Models
Supports weight conversion between Hugging Face (HF) and Megatron-Core (Mcore) formats for various mainstream Large Language Models (LLMs), facilitating user experimentation and deployment.

|**Model Series**|**Model**|**Convert File**|**HF->Mcore**|**Mcore->HF**|
|-|-|-|-|-|
|**DeepSeek-V2**|DeepSeek-V2|deepseek_v2_convert|||
||DeepSeek-V2-Lite|deepseek_v2_lite_convert|✅|✅|
|**DeepSeek-V3**|DeepSeek-V3|deepseek_v3_convert|||
|**Llama2**|Llama2_7b|llama2_convert|✅|✅|
||Llama2_13b|llama2_convert|✅|✅|
||Llama2_70b|llama2_convert|✅|✅|
|**Llama3**|Llama3_8b|llama3_convert|✅|✅|
||Llama3_70b|llama3_convert|✅|✅|
|**Llama3.1**|Llama3_8b|llama3_1_convert|✅|✅|
||Llama3_70b|llama3_1_convert|✅|✅|
||Llama3_405b|llama3_1_convert|||
|**Qwen**|qwen_1.8b|qwen_convert|✅|✅|
||qwen_7b|qwen_convert|✅|✅|
||qwen_14b|qwen_convert|✅|✅|
||qwen_72b|qwen_convert|✅|✅|
|**Qwen1.5**|qwen1.5_0.5b|qwen1_5_convert|✅|✅|
||qwen1.5_1.8b|qwen1_5_convert|✅|✅|
||qwen1.5_4b|qwen1_5_convert|✅|✅|
||qwen1.5_7b|qwen1_5_convert|✅|✅|
||qwen1.5_14b|qwen1_5_convert|✅|✅|
||qwen1.5_32b|qwen1_5_convert|✅|✅|
||qwen1.5_72b|qwen1_5_convert|✅|✅|
|**Qwen2**|qwen2_0.5b|qwen2_convert|✅|✅|
||qwe2_1.5b|qwen2_convert|✅|✅|
||qwen2_7b|qwen2_convert|✅|✅|
||qwen2_72b|qwen2_convert|✅|✅|
|**Qwen2.5**|qwen2.5_0.5b|qwen2_5_convert_llm|✅|✅|
||qwe2.5_1.5b|qwen2_5_convert_llm|✅|✅|
||qwen2.5_3b|qwen2_5_convert_llm|✅|✅|
||qwen2.5_7b|qwen2_5_convert_llm|✅|✅|
||qwen2.5_14b|qwen2_5_convert_llm|✅|✅|
||qwen2.5_32b|qwen2_5_convert_llm|✅|✅|
||qwen2.5_72b|qwen2.5_convert_llm|✅|✅|
|**Qwen3**|qwen3_0.6b|qwen3_convert|✅|✅|
||qwen3_1.7b|qwen3_convert|✅|✅|
||qwen3_4b|qwen3_convert|✅|✅|
||qwen3_8b|qwen3_convert|✅|✅|
||qwen3_14b|qwen3_convert|✅|✅|
||qwen3_32b|qwen3_convert|✅|✅|
||qwen3_30b_a3b|qwen3_moe_convert|✅|✅|
||qwen3_coder_30b_a3b|qwen3_moe_convert|✅|✅|
||qwen3_235b_a22b|qwen3_moe_convert|||
||qwen3_480b_a35b|qwen3_moe_convert|||

## 2. Common Parameters
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
|amax_epsilon|Epsilon value for amax calculation in FP8 conversion; used for FP8 quantization scale, aligned with the FP8 EPS environment variable set during training|
|quant_method|The quantization method to use. Choices: [te, pt, aiak]. When using Nvidia B-series GPUs (Blackwell) for weight conversion, this value needs to be set to `pt`|
|fp8_force_no_requant|skip dequantize + re-quantize in FP8 conversion|

For descriptions of other parameters, please refer to [checkpoint_convert.md](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/VPxwT-t6VJ/fj-SCq_ssunsiH?t=mention&mt=doc&dt=doc).

## 3. Example Scripts
The framework provides weight conversion example scripts for each model. Users can find specific scripts under `configs/models/{model}/ckpt_convert/`.

Below is an example script for converting **DeepSeek V3.1** model weights from **Huggingface FP8** format to **MegatronCore FP8** format. When converting FP8 format weights, it is crucial to set the `amax_epsilon` parameter. This parameter must align with the FP8 EPS environment variables set during training (`export FP8_QUANT_FWD_INP_AMAX_EPS`, `export FP8_QUANT_FWD_WEIGHT_AMAX_EPS`, `export FP8_QUANT_BWD_GRAD_AMAX_EPS`).

If the user is using the **Nvidia B-series GPUs**, the `--quant_method pt` parameter must be added to the conversion script.

```bash
#! /bin/bash

export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/OmniTraining"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/path/to/hf_checkpoint  # the original DeepSeek-V3 checkpoint is FP8 format
SAVE=/path/to/your/save  # the converted checkpoint will be in MCore FP8 format

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek3/deepseek_v3.yaml
CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek3/ckpt_convert/deepseek_v3_convert.yaml

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

```bash
#! /bin/bash

export AIAK_TRAINING_PATH=${AIAK_TRAINING_PATH:-"/workspace/OmniTraining"}
MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/AIAK-Megatron"}
CONVERT_CHECKPOINT_PATH="$AIAK_TRAINING_PATH/tools/convert_checkpoint"

LOAD=/path/to/mcore_checkpoint  # the converted checkpoint will be in MCore FP8 format
SAVE=/path/to/your/save  # the original DeepSeek-V3 checkpoint is FP8 format

MODEL_CONFIG_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek3/deepseek_v3.yaml
CONVERT_FILE=${AIAK_TRAINING_PATH}/configs/models/deepseek3/ckpt_convert/deepseek_v3_convert.yaml

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
    --fp8_force_no_requant
```