# Model Checkpoint Conversion

## Overview
LoongForge supports bidirectional conversion between Mcore and Hugging Face (HF) formats for various models. Users can leverage the built-in conversion tools and scripts within the framework to quickly convert model weights and proceed with training or validation.

## Supported Models for Conversion

### VLM Models

| **Model Family** | **Model** | **Convert Files** | **HF → Mcore** | **Mcore → HF** |
|------------------|-----------|-------------------|----------------|----------------|
| **Qwen2.5VL**    | qwen2_5_vl_3b | qwen2_5_vit_convert<br>qwen_mlp_adapter_convert<br>qwen2_5_convert | ✅ | ✅ |
|                  | qwen2_5_vl_7b | qwen2_5_vit_convert<br>qwen_mlp_adapter_convert<br>qwen2_5_convert | ✅ | ✅ |
|                  | qwen2_5_vl_32b | qwen2_5_vit_convert<br>qwen_mlp_adapter_convert<br>qwen2_5_convert | ✅ | ✅ |
|                  | qwen2_5_vl_72b | qwen2_5_vit_convert<br>qwen_mlp_adapter_convert<br>qwen2_5_convert | ✅ | ✅ |
| **InternVL2.5**  | internvl2.5_8b | intern_vit_0.3b_convert<br>intern_mlp_adapter_convert<br>internlm2_5_convert | ✅ | ✅ |
|                  | internvl2.5_26b | intern_vit_6b_convert<br>intern_mlp_adapter_convert<br>internlm2_5_convert | ✅ | ✅ |
|                  | internvl2.5_38b | intern_vit_6b_convert<br>intern_mlp_adapter_convert<br>qwen2_5_convert | ✅ | ✅ |
|                  | internvl2.5_78b | intern_vit_6b_convert<br>intern_mlp_adapter_convert<br>qwen2_5_convert | ✅ | ✅ |
| **InternVL3.5**  | internvl3.5_8b | intern_vit_0.3b_convert<br>intern_mlp_adapter_convert<br>qwen3_convert_intern | ✅ | ✅ |
|                  | internvl3.5_14b | intern_vit_0.3b_convert<br>intern_mlp_adapter_convert<br>qwen3_convert_intern | ✅ | ✅ |
|                  | internvl3.5_30b_a3b | intern_vit_0.3b_convert<br>intern_mlp_adapter_convert<br>qwen3_moe_convert_intern | ✅ | ✅ |
|                  | internvl3.5_38b | intern_vit_6b_convert<br>intern_mlp_adapter_convert<br>qwen3_convert_intern | ✅ | ✅ |
|                  | internvl3.5_241b_a28b | intern_vit_6b_convert<br>intern_mlp_adapter_convert<br>qwen3_moe_convert_intern | ❌ | ❌ |
| **LLaVA-OV1.5**  | llava_ov_1_5_30b_a3b | llava_vit_convert<br>llava_mlp_adapter_convert<br>qwen3_moe_convert | ❌ | ❌ |
|                  | llava_ov_4b | llava_vit_convert<br>llava_mlp_adapter_convert<br>qwen3_convert_llava | ✅ | ✅ |
| **Qwen3.6**      | qwen3_6_27b | qwen3_5_vit_convert<br>qwen_3_mlp_adapter_convert<br>qwen3_6_dense_convert | ✅ | ✅ |
|                  | qwen3_6_35b_a3b | qwen3_5_vit_convert<br>qwen_3_mlp_adapter_convert<br>qwen3_6_moe_convert | ✅ | ✅ |

---

### LLM Models

| **Model Family** | **Model** | **Convert Files** | **HF → Mcore** | **Mcore → HF** |
|------------------|-----------|-------------------|----------------|----------------|
| **DeepSeek-V2**  | DeepSeek-V2 | deepseek_v2_convert | ❌ | ❌ |
|                  | DeepSeek-V2-Lite | deepseek_v2_lite_convert | ✅ | ✅ |
| **DeepSeek-V3**  | DeepSeek-V3 | deepseek_v3_convert | ❌ | ❌ |
| **Llama2**       | Llama2_7b | llama2_convert | ✅ | ✅ |
|                  | Llama2_13b | llama2_convert | ✅ | ✅ |
|                  | Llama2_70b | llama2_convert | ✅ | ✅ |
| **Llama3**       | Llama3_8b | llama3_convert | ✅ | ✅ |
|                  | Llama3_70b | llama3_convert | ✅ | ✅ |
| **Llama3.1**     | Llama3_8b | llama3_1_convert | ✅ | ✅ |
|                  | Llama3_70b | llama3_1_convert | ✅ | ✅ |
|                  | Llama3_405b | llama3_1_convert | ❌ | ❌ |
| **Qwen**         | qwen_1.8b | qwen_convert | ✅ | ✅ |
|                  | qwen_7b | qwen_convert | ✅ | ✅ |
|                  | qwen_14b | qwen_convert | ✅ | ✅ |
|                  | qwen_72b | qwen_convert | ✅ | ✅ |
| **Qwen1.5**      | qwen1.5_0.5b | qwen1_5_convert | ✅ | ✅ |
|                  | qwen1.5_1.8b | qwen1_5_convert | ✅ | ✅ |
|                  | qwen1.5_4b | qwen1_5_convert | ✅ | ✅ |
|                  | qwen1.5_7b | qwen1_5_convert | ✅ | ✅ |
|                  | qwen1.5_14b | qwen1_5_convert | ✅ | ✅ |
|                  | qwen1.5_32b | qwen1_5_convert | ✅ | ✅ |
|                  | qwen1.5_72b | qwen1_5_convert | ✅ | ✅ |
| **Qwen2**        | qwen2_0.5b | qwen2_convert | ✅ | ✅ |
|                  | qwen2_1.5b | qwen2_convert | ✅ | ✅ |
|                  | qwen2_7b | qwen2_convert | ✅ | ✅ |
|                  | qwen2_72b | qwen2_convert | ✅ | ✅ |
| **Qwen2.5**      | qwen2.5_0.5b | qwen2_5_convert_llm | ✅ | ✅ |
|                  | qwen2.5_1.5b | qwen2_5_convert_llm | ✅ | ✅ |
|                  | qwen2.5_3b | qwen2_5_convert_llm | ✅ | ✅ |
|                  | qwen2.5_7b | qwen2_5_convert_llm | ✅ | ✅ |
|                  | qwen2.5_14b | qwen2_5_convert_llm | ✅ | ✅ |
|                  | qwen2.5_32b | qwen2_5_convert_llm | ✅ | ✅ |
|                  | qwen2.5_72b | qwen2_5_convert_llm | ✅ | ✅ |
| **Qwen3**        | qwen3_0.6b | qwen3_convert | ✅ | ✅ |
|                  | qwen3_1.7b | qwen3_convert | ✅ | ✅ |
|                  | qwen3_4b | qwen3_convert | ✅ | ✅ |
|                  | qwen3_8b | qwen3_convert | ✅ | ✅ |
|                  | qwen3_14b | qwen3_convert | ✅ | ✅ |
|                  | qwen3_32b | qwen3_convert | ✅ | ✅ |
|                  | qwen3_30b_a3b | qwen3_moe_convert | ✅ | ✅ |
|                  | qwen3_coder_30b_a3b | qwen3_moe_convert | ✅ | ✅ |
|                  | qwen3_235b_a22b | qwen3_moe_convert | ❌ | ❌ |
|                  | qwen3_480b_a35b | qwen3_moe_convert | ❌ | ❌ |
| **Glm5**         | glm5 | glm5_convert | ✅ | ✅ |
---

## Usage Example

To convert a Qwen3-14B model using the framework:

1. Download the HF model weights from [https://huggingface.co/Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B).
2. Confirm the model parallelism strategy, e.g.:

| **Model** | **Parallelism Strategy** |
|-----------|--------------------------|
| Qwen3_14B | TP=2, PP=4 |

3. Use the built-in script to perform HF → Mcore conversion. For Qwen3-14B:

```bash
# Modify the save/load paths and parallelism settings in the script
# load path: HF checkpoint, save path: Mcore checkpoint
sh examples/qwen3/checkpoint_convert/convert_qwen3_14b_hf_to_mcore.sh
```

4. After training, use the script to convert back to HF format:

```bash
# Modify the save/load paths and parallelism settings
# load path: Mcore checkpoint, save path: HF checkpoint
sh examples/qwen3/checkpoint_convert/convert_qwen3_14b_mcore_to_hf.sh
```

---

## Conversion Tool Parameters Summary

| **Parameter** | **Description** | **Options** | **Default** |
|---------------|------------------|-------------|-------------|
| `load_platform` | Platform to load checkpoint from | `huggingface`, `mcore` | `None` |
| `save_platform` | Platform to save checkpoint to | `huggingface`, `mcore` | `None` |
| `load_ckpt_path` | Path to load checkpoint | Any valid path | `None` |
| `save_ckpt_path` | Path to save checkpoint | Any valid path | `None` |
| `common_config_path` | Path to common config | Any valid path | `None` |
| `megatron_path` | Base directory of Megatron repository | Any valid path | `None` |
| `no_load_optim` | Do not convert optimizer | `True`/`False` | `False` |
| `no_save_optim` | Do not save optimizer | `True`/`False` | `False` |
| `model_type_custom` | Custom model type | Any string | `None` |
| `safetensors` | Use safetensors format | `True`/`False` | `False` |
| `convert_to_fp8` | Convert float16 weights to fp8 | `True`/`False` | `False` |
| `quant_method` | Quantization method | `te`, `pt` | `te` |
| `fp8_force_no_requant` | Skip dequantize + re-quantize in FP8 conversion | `True`/`False` | `False` |
| `force_pow_2_scales` | Force destination checkpoint's scale to be power-of-two | `True`/`False` | `False` |
| `amax_epsilon` | Epsilon value for amax calculation in FP8 conversion | Any float | `0.0` |
| `pretrain_as_fp8` | Run pretrain as fp8 | `True`/`False` | `False` |
| `fp8_quant_transfer_type` | Transfer dtype when converting from HF fp8 to mcore fp8 | `float32`, `bfloat16` | `float32` |
| `distributed_convert` | Convert checkpoint in distributed mode | `True`/`False` | `False` |
| `config_file` | Config file for model configuration | Any valid path | `None` |
| `convert_file` | Convert file for checkpoint conversion | Any valid path | `None` |
| `torch_dtype` | Target PyTorch dtype | `float16`, `float32`, `bfloat16` | `None` |
| `vocab_size` | Vocabulary size | Any positive integer | `None` |
| `vpp-scheduler` | Virtual pipeline scheduler type | `dualpipev` | `None` |
| `num_virtual_stages_per_pipeline_rank` | Number of virtual pipeline stages per PP rank | Any positive integer | `None` |
| `decoder-first-pipeline-num-layers` | Number of transformer layers on the first PP stage of decoder | Any positive integer | `None` |
| `decoder-last-pipeline-num-layers` | Number of transformer layers on the last PP stage of decoder | Any positive integer | `None` |
| `use_distributed_optimizer` | Use distributed optimizer | `True`/`False` | `False` |
| `tensor_model_parallel_size` | Target tensor model parallel size | Any positive integer | `1` |
| `pipeline_model_parallel_size` | Target pipeline model parallel size | Any positive integer | `1` |
| `data_parallel_size` | Target data parallel size | Any positive integer | `1` |
| `expert_parallel_size` | Target expert parallel size | Any positive integer | `None` |
| `expert_tensor_parallel_size` | Degree of expert model parallelism | Any positive integer | `None` |
| `pad_vocab_size_to` | Pad the vocab size to this value | Any positive integer | `None` |
| `add_embedding_padding` | Whether to add embedding padding | `True`/`False` | `True` |
| `make_vocab_size_divisible_by` | Pad the vocab size to be divisible by this value | Any positive integer | `128` |
| `custom_pipeline_layers` | Custom pipeline layer distribution | Comma-separated numbers | `None` |
| `num_layers_per_virtual_pipeline_stage` | Number of layers per virtual pipeline stage | Any positive integer | `None` |
| `transformer_impl` | Transformer implementation to use | `local`, `transformer_engine` | `transformer_engine` |
| `num_experts` | Number of experts in MoE | Any positive integer | `None` |
| `checkpoint-format` | HF checkpoint format | Any string | `None` |
| `max_workers` | Number of threads for checkpoint conversion | Any positive integer | `1` |
| `no-te` | Self-attention does not contain input_layernorm in mcore | `True`/`False` | `False` |
| `moe-grouped-gemm` | Use grouped GEMM in MoE | `True`/`False` | `False` |
| `resume-convert` | Resume checkpoint conversion on failure | `True`/`False` | `False` |
| `cache-path` | Cache path used during conversion | Any valid path | `None` |
| `layer-for-test` | Get specific layer from checkpoint for testing | Any string | `None` |
| `num-experts-for-test` | Number of experts in MoE for testing | Any positive integer | `None` |
| `sub-num-layers-for-save` | Number of layers to save each time | Any positive integer | `None` |
| `save-sub-checkpoint-by-pp` | Save sub-checkpoints by pipeline parallelism | `True`/`False` | `False` |

You can add new model weight conversions based on LoongForge’s conversion tools.
