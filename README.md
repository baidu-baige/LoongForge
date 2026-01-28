# Overview
![AIAK-Training-Omni Logo](assets/_images/Image_20260117_204343.jpg)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/)
[![arxive](https://img.shields.io/badge/cs.AI-XXXXX-B31C1C?logo=arxiv&logoColor=B31C1C)](https://github.com/baidu-baige/AIAK-Training-Omni)
[![license](https://img.shields.io/github/license/open-mmlab/mmdeploy.svg)](https://github.com/baidu-baige/AIAK-Training-Omni/blob/master/LICENSE)
[![stars](https://img.shields.io/github/stars/baidu-baige/AIAK-Training-Omni=social)](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master)
[![open issues](https://img.shields.io/github/issues-raw/baidu-baige/AIAK-Training-Omni)](https://github.com/baidu-baige/AIAK-Training-Omni/issues)
## Introduction
The AIAK-Omni framework is an all-scenario training framework built on Megatron, covering LLM, VLM, VLA, Diffusion, supporting large-scale training of various models and scenarios, with training performance reaching SOTA.

* Excellent performance, achieving SOTA training performance across multiple models, suitable for both large-scale training (&gt;1000 GPUs) and fine-tuning of small-scale models.
* Coverage of mainstream models and tasks, supporting over 10 types of LLM/VLM, and tasks including model pretraining and fine-tuning.
* Model combination, allowing flexible replacement of model components in VLM to achieve flexible networking.
* Rich tools provided, including efficient and scalable data components, weight conversion, etc.

## Latest News
[2026/01/31] 🔥We have released the AIAK-Training-Omni framework! A brand-new multimodal large model training framework.

## Features
* Flexible networking, we support flexible combination of different components in VLM, such as LLM/VIT, etc. See [model_combination.md](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/features/model_combination.md) for details.
* Heterogeneous TP, supporting different TP size splitting for different components in VLM to cope with various model sizes, see [heterogeneous_tp_parallel.md](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/features/heterogeneous_tp_parallel.md) for details.
* DP data balance, optimizing the data parallel load imbalance problem introduced by data packing, see [data_parallel_balance.md](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/features/data_parallel_balancing.md) for details.
* Offline data packing, supporting offline data packing to reduce the number of padding tokens during training, see [offline_data_packing.md](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/features/offline_data_packing.md) for details.
* FP8 training, supporting FP8 precision training, see [fp8_training.md](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/features/fp8_training.md) and [fp8_training_for_vlm](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/features/fp8_training_for_vlm.md) for details.
* MOE optimization, the framework optimizes the training performance of MOE models, see [moe_all2all_overlap.md](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/features/moe_all2all_overlap.md) for details.

## Roadmap
* [ ] Support more models, including Ernie4.5VL, GROOT N1.5, etc.
* [ ] Support more tasks, including Lora, DPO, Knowledge Distillation, etc.
* [ ] Support heterogeneous DP for VLM models.
* [ ] Support for DP load balancing across adjacent micro-batches.

## Documentation
🔔🔔🔔Please refer to the [documentation](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/) for detailed usage and features of the framework.

## Support Model

|**Model Type**|**Model Category**|**Model**|**Pretrain**|**SFT**|**Config**|
|-|-|-|-|-|-|
|LLM|DeepSeek-V2|deepseek_v2|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v2/pretrain/pretrain_deepseek_v2_group.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v2/finetuning/sft_deepseek_v2_group.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/deepseek2/deepseek_v2.yaml)|
|||deepseek_v2_lite|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v2/pretrain/pretrain_deepseek_v2_lite_group.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v2/finetuning/sft_deepseek_v2_lite_group.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/deepseek2/deepseek_v2_lite.yaml)|
||DeepSeek-V3|deepseek_v3_bf16|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v3/pretrain/pretrain_deepseek_v3_group_bf16.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v3/finetuning/sft_deepseek_v3_group_bf16.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/deepseek3/deepseek_v3.yaml)|
|||deepseek_v3_fp8|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v3/pretrain/pretrain_deepseek_v3_group_fp8.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/deepseek_v3/finetuning/sft_deepseek_v3_group_fp8.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/deepseek3/deepseek_v3.yaml)|
||Llama2|llama2_7b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama2/pretrain/pretrain_llama2_7b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama2/finetuning/sft_llama2_7b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama2/llama2_7b.yaml)|
|||llama2_13b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama2/pretrain/pretrain_llama2_13b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama2/finetuning/sft_llama2_13b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama2/llama2_13b.yaml)|
|||llama2_70b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama2/pretrain/pretrain_llama2_70b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama2/finetuning/sft_llama2_70b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama2/llama2_70b.yaml)|
||Llama3|llama3_8b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3/pretrain/pretrain_llama3_8b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3/finetuning/sft_llama3_8b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama3/llama3_8b.yaml)|
|||llama3_70b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3/pretrain/pretrain_llama3_70b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3/finetuning/sft_llama3_70b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama3/llama3_70b.yaml)|
||Llama3.1|llama3.1_8b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3.1/pretrain/pretrain_llama3.1_8b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3.1/finetuning/sft_llama3.1_8b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama3/llama3_1_8b.yaml)|
|||llama3.1_70b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3.1/pretrain/pretrain_llama3.1_70b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3.1/finetuning/sft_llama3.1_70b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama3/llama3_1_70b.yaml)|
|||llama3.1_405b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3.1/pretrain/pretrain_llama3.1_405b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llama3.1/finetuning/sft_llama3.1_405b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llama3/llama3_1_405b.yaml)|
||Qwen|qwen_1.8b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/pretrain/pretrain_qwen_1.8b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/finetuning/sft_qwen_1.8b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen_1_8b.yaml)|
|||qwen_7b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/pretrain/pretrain_qwen_7b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/finetuning/sft_qwen_7b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen_7b.yaml)|
|||qwen_14b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/pretrain/pretrain_qwen_14b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/finetuning/sft_qwen_14b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen_14b.yaml)|
|||qwen_72b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/pretrain/pretrain_qwen_72b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen/finetuning/sft_qwen_72b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen_72b.yaml)|
||Qwen1.5|qwen1.5_0.5b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/pretrain/pretrain_qwen1.5_0.5b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/finetuning/sft_qwen1.5_0.5b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen1_5_0_5b.yaml)|
|||qwen1.5_1.8b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/pretrain/pretrain_qwen1.5_1.8b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/finetuning/sft_qwen1.5_1.8b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen1_5_1_8b.yaml)|
|||qwen1.5_4b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/pretrain/pretrain_qwen1.5_4b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/finetuning/sft_qwen1.5_4b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen1_5_4b.yaml)|
|||qwen1.5_7b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/pretrain/pretrain_qwen1.5_7b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/finetuning/sft_qwen1.5_7b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen1_5_7b.yaml)|
|||qwen1.5_14b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/pretrain/pretrain_qwen1.5_14b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/finetuning/sft_qwen1.5_14b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen1_5_14b.yaml)|
|||qwen1.5_32b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/pretrain/pretrain_qwen1.5_32b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/finetuning/sft_qwen1.5_32b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen1_5_32b.yaml)|
|||qwen1.5_72b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/pretrain/pretrain_qwen1.5_72b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen1.5/finetuning/sft_qwen1.5_72b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen/qwen1_5_72b.yaml)|
||Qwen2|qwen2_0.5b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/pretrain/pretrain_qwen2_0.5b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/finetuning/sft_qwen2_0.5b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2/qwen2_0_5b.yaml)|
|||qwen2_1.5b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/pretrain/pretrain_qwen2_1.5b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/finetuning/sft_qwen2_1.5b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2/qwen2_1_5b.yaml)|
|||qwen2_7b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/pretrain/pretrain_qwen2_7b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/finetuning/sft_qwen2_7b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2/qwen2_7b.yaml)|
|||qwen2_72b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/pretrain/pretrain_qwen2_72b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2/finetuning/sft_qwen2_72b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2/qwen2_72b.yaml)|
||Qwen2.5|qwen2.5_0.5b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/pretrain/pretrain_qwen2.5_0.5b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/finetuning/sft_qwen2.5_0.5b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5/qwen2_5_0_5b.yaml)|
|||qwen2.5_1.5b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/pretrain/pretrain_qwen2.5_1.5b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/finetuning/sft_qwen2.5_1.5b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5/qwen2_5_1_5b.yaml)|
|||qwen2.5_3b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/pretrain/pretrain_qwen2.5_3b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/finetuning/sft_qwen2.5_3b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5/qwen2_5_3b.yaml)|
|||qwen2.5_7b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/pretrain/pretrain_qwen2.5_7b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/finetuning/sft_qwen2.5_7b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5/qwen2_5_7b.yaml)|
|||qwen2.5_14b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/pretrain/pretrain_qwen2.5_14b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/finetuning/sft_qwen2.5_14b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5/qwen2_5_14b.yaml)|
|||qwen2.5_32b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/pretrain/pretrain_qwen2.5_32b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/finetuning/sft_qwen2.5_32b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5/qwen2_5_32b.yaml)|
|||qwen2.5_72b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/pretrain/pretrain_qwen2.5_72b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5/finetuning/sft_qwen2.5_72b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5/qwen2_5_72b.yaml)|
||Qwen3|qwen3_0.6b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_0.6b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_0.6b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_0_6b.yaml)|
|||qwen3_1.7b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_1.7b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_1.7b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_1_7b.yaml)|
|||qwen3_4b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_4b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_4b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_4b.yaml)|
|||qwen3_8b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_8b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_8b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_8b.yaml)|
|||qwen3_14b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_14b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_14b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_14b.yaml)|
|||qwen3_32b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_32b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_32b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_32b.yaml)|
|||qwen3_30b_a3b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_30b_a3b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_30b_a3b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_30b_a3b.yaml)|
|||qwen3_235b_a22b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_235b_a22b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_235b_a22b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_235b_a22b.yaml)|
|||qwen3_480b_a35b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_480b_a35b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_480b_a35b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_480b_a35b.yaml)|
|||qwen3_coder_30b_a3b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/pretrain/pretrain_qwen3_coder_30b_a3b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3/finetuning/sft_qwen3_coder_30b_a3b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3/qwen3_coder_30b_a3b.yaml)|
|VLM|Qwen2.5-VL|qwen2.5_vl_3b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/pretrain/pretrain_qwen2_5_vl_3b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/sft/sft_qwen2_5_vl_3b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5vl/qwen2_5_vl_3b.yaml)|
|||qwen2.5_vl_7b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/pretrain/pretrain_qwen2_5_vl_7b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/sft/sft_qwen2_5_vl_7b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5vl/qwen2_5_vl_7b.yaml)|
|||qwen2.5_vl_32b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/pretrain/pretrain_qwen2_5_vl_32b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/sft/sft_qwen2_5_vl_32b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5vl/qwen2_5_vl_32b.yaml)|
|||qwen2.5_vl_72b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/pretrain/pretrain_qwen2_5_vl_72b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen2.5_vl/sft/sft_qwen2_5_vl_72b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen2.5vl/qwen2_5_vl_72b.yaml)|
||Qwen3-VL|qwen3_vl_30b_a3b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3_vl/pretrain/pretrain_qwen3_vl_30b_a3b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3_vl/sft/sft_qwen3_vl_30b_a3b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3_vl/qwen3_vl_30b_a3b.yaml)|
|||qwen3_vl_235b_a22b|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3_vl/pretrain/pretrain_qwen3_vl_235b_a22b.sh  ))|✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/qwen3_vl/sft/sft_qwen3_vl_235b_a22b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/qwen3_vl/qwen3_vl_235b_a22b.yaml)|
||LLava-OV-1.5|llava_ov_4b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llavaov_1.5/pretrain/stage_1_alignment_llava_ov_4b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llavaov1.5/llavaov_1_5_4b.yaml)|
|||llava_ov_30b_a3b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/llavaov_1.5/pretrain/stage_1_alignment_llava_ov_30b_a3b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/llavaov1.5/llavaov_1_5_30b_a3b.yaml)|
||InternVL-2.5|internvl2.5_8b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl2.5/sft_internvl2_5_8b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl2.5/internvl2_5_8b.yaml)|
|||internvl2.5_26b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl2.5/sft_internvl2_5_26b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl2.5/internvl2_5_26b.yaml)|
|||internvl2.5_38b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl2.5/sft_internvl2_5_38b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl2.5/internvl2_5_38b.yaml)|
|||internvl2.5_78b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl2.5/sft_internvl2_5_78b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl2.5/internvl2_5_78b.yaml)|
||InternVL-3.5|internvl3.5_8b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl3.5/finetuning/sft_internvl3_5_8b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl3.5/internvl3_5_8b.yaml)|
|||internvl3.5_14b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl3.5/finetuning/sft_internvl3_5_14b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl3.5/internvl3_5_14b.yaml)|
|||internvl3.5_38b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl3.5/finetuning/sft_internvl3_5_38b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl3.5/internvl3_5_38b.yaml)|
|||internvl3.5_30b_a3b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl3.5/finetuning/sft_internvl3_5_30b_a3b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl3.5/internvl3_5_30b_a3b.yaml)|
|||internvl3.5_241b_a28b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/internvl3.5/finetuning/sft_internvl3_5_241b_a28b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/internvl3.5/internvl3_5_241b_a28b.yaml)|
|Wan|Wan2.1|wan2.1_i2v_14b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/wan/pretrain_wan2.1_i2v_14b_480p.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/wan/wan2_1_i2v.yaml)|
||Wan2.2|wan2.2_i2v_a14b||✅([example](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/examples/wan/pretrain_wan2.2_i2v_a14b.sh  ))|[model config](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/configs/models/wan/wan2_2_i2v.yaml)|

## Quick Start

### Quick Start for Qwen3


#### Installation  
Please refer to the [Installation Guide](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/get_started/installation.md) for detailed instructions.

#### Prepare the dataset  
Convert your corpus into JSON format.  
Each sample should contain three fields: `instruction`, `input`, and `output`.

**Example file: `sft_alpaca_zh_data.json`**
```json
[
  {
    "instruction": "Identify and explain the two scientific theories in the given list: cell theory and heliocentric theory.",
    "input": "",
    "output": "Cell theory is a fundamental biological theory stating that all living organisms are composed of cells, the basic unit of life. Proposed by Schleiden, Schwann, and Virchow in 1839, it asserts that all organisms are made of one or more cells, and cells only arise from pre-existing cells through division.\n\nHeliocentric theory posits that the Sun is the center of the solar system, with planets orbiting around it. This theory overturned the geocentric view. Nicolaus Copernicus introduced it in the early 16th century in his work 'De revolutionibus orbium coelestium', laying the groundwork for modern astronomy."
  },
  {
    "instruction": "Enter the names of three basketball teams and generate an appropriate slogan.",
    "input": "Oklahoma City Thunder, Chicago Bulls, Brooklyn Nets",
    "output": "\"Thunder, Bulls and Nets: each shows its prowess, battling for supremacy!\""
  },
  {
    "instruction": "Compare the two famous figures below using a list of characteristics and provide the output in sentences.",
    "input": "Person 1: Albert Einstein\nPerson 2: Stephen Hawking\nCharacteristics: intelligence, education, impact on the world",
    "output": "Both Albert Einstein and Stephen Hawking possessed extraordinary intelligence. They achieved top educational accomplishments and profoundly advanced science. Einstein is renowned for relativity and the mass-energy equivalence E=mc², while Hawking is celebrated for his discoveries about black holes and cosmology. Their remarkable contributions have influenced the world deeply."
  },
  {
    "instruction": "Please construct a question that needs to be answered based on the following article.",
    "input": "",
    "output": "I'm sorry, you haven't provided any article or information for me to construct a question. Please supply the corresponding article or information so that I can help you formulate a question."
  }
]
```

Point the environment variable to your file (or use the built-in test set *$AIAK_TRAINING_PATH/tests/datasets/llm/sft_alpaca_zh_data.json*):
```bash
export DATA_PATH=/your/dataset/path
# use test set: export DATA_PATH=$AIAK_TRAINING_PATH/tests/datasets/llm/sft_alpaca_zh_data.json
```

#### Checkpoint conversion  
1. Download Hugging Face weights from [Qwen3-8B checkpoint](https://huggingface.co/Qwen/Qwen3-8B).  
2. Convert to Megatron-Core format:

```bash
cd $AIAK_TRAINING_PATH/examples/qwen3/checkpoint_convert

# edit LOAD (HF path) and SAVE (MCore output path)
vim convert_qwen3_8b_hf_to_mcore.sh

bash convert_qwen3_8b_hf_to_mcore.sh

export CHECKPOINT_PATH=/path/to/mcore/ckpt
```

#### Launch Qwen3-8B SFT  
```bash
bash examples/qwen3/finetuning/sft_qwen3_8b.sh
```

### Quick Start for LLM Model Training
Refer to the [Quick Start for LLM Pretrain](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/llm_tutorial/quick_start_llm_pretrain.md) and [Quick Start for LLM SFT](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/llm_tutorial/quick_start_llm_sft.md) document for details.

### Quick Start for VLM Model Training
Refer to the [Quick Start for VLM Pretrain](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/vlm_tutorial/quick_start_vlm_pretrain.md) and [Quick Start for VLM SFT](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/vlm_tutorial/quick_start_vlm_sft.md) document for details.

### Quick Start for VLA Model Training
Refer to the [Quick Start for VLA Training](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/vla_tutorial/quick_start_vla_training.md) document for details.

### Quick Start for Diffusion Model Training
Refer to the [Quick Start for WAN Training](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/docs/source/wan_tutorial/quick_start_wan_training.md) document for details.

## References
- [qianfan-vl](https://github.com/baidubce/Qianfan-VL) – is a general-purpose multimodal model enhanced for enterprise-level multimodal applications.   
- [llava-ov-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5) – is a family of fully open-source large multimodal models (LMMs) that operate on native-resolution images, achieve state-of-the-art performance, and require comparatively lower training costs.

## Contributing
Please read [CONTRIBUTING.md](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the Apache License - see the [LICENSE](https://github.com/baidu-baige/AIAK-Training-Omni/tree/master/LICENSE) file for details.

## Citation
If you use this work, please consider citing:
```bibtex
@misc{AIAK-Training-Omni 2025,
      title={}, 
      author={},
      year={2025},
      eprint={2512.24077},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={}, 
}
```