<div align="center">

![OmniTraining Logo](docs/assets/images/omni.jpg)

<h4> Modular, Scalable & High-Efficiency Training Library for Multi-Modal, Multi-Architecture Models </h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/)[![arxive](https://img.shields.io/badge/cs.AI-XXXXX-B31C1C?logo=arxiv&logoColor=B31C1C)](https://github.com/baidu-baige/OmniTraining)[![license](https://img.shields.io/github/license/open-mmlab/mmdeploy.svg)](https://github.com/baidu-baige/OmniTraining/blob/master/LICENSE)[![stars](https://img.shields.io/github/stars/baidu-baige/OmniTraining=social)](https://github.com/baidu-baige/OmniTraining/tree/master)[![open issues](https://img.shields.io/github/issues-raw/baidu-baige/OmniTraining)](https://github.com/baidu-baige/OmniTraining/issues)

</div>

## About

**OmniTraining** (evolved from [AIAK-Training-LLM](https://cloud.baidu.com/doc/AIHC/s/Alyo476jr)) is a high-efficiency training framework designed for large-scale transformer models across diverse modalities and architectures.Through continuous adaptation to emerging model structures and deep optimization of performance, OmniTraining delivers a comprehensive, production-ready training solution.

* **🚀 Comprehensive Model Coverage**: Natively supports mainstream model architectures including LLMs (Large Language Models), VLMs (Vision-Language Models), VLAs (Vision-Language-Action Models), and Diffusion Models. It covers key training workflows including pre-training, continued pre-training, and SFT (Supervised Fine-Tuning), with core strategies validated in production environments.

* **⚡ Performance-Driven Optimization**: Built on Megatron with additional enhancements while maintaining full compatibility with existing optimization strategies. OmniTraining delivers advanced optimizations across communication, computation, and memory management, further optimizing training performance to significantly reduce training costs and accelerate model development.

* **🧪 Heterogeneous Hardware Support**: OmniTraining provides native, high-performance support for both NVIDIA GPUs and Kunlun XPUs, enabling seamless migration and stable training across different hardware backends.


## Latest News 🔥
- [2026/03] Initial release of OmniTraining framework!

## ✨ Key Features

* **Training Backends**: Supports Megatron and Megatron-FSDP training backends.
* **Model Support**: Extensive open-source model coverage with built-in configs and scripts for quick deployment.
* **Flexible Composition**: Enable custom VLMs by combining ViT, LLM, and other components through configuration. [Details](https://github.com/baidu-baige/OmniTraining/tree/master/docs/source/features/model_combination.md)
* **Training Methods**: Supports Pretrain, MidTrain, SFT, and LoRA.
* **Data Processing**: Tools for building datasets, including data packing to reduce padding tokens during training.
* **Weight Conversion**: Bidirectional Megatron ↔ HuggingFace weight conversion supporting FP8, BF16, and other precisions.
* **FP8 Training**: Production-validated FP8 precision training for [LLM](https://github.com/baidu-baige/OmniTraining/tree/master/docs/source/features/fp8_training.md) and [VLM](https://github.com/baidu-baige/OmniTraining/tree/master/docs/source/features/fp8_training_for_vlm.md).
* **MoE Optimization**: All2All communication + activation tensor offload + computation overlap for memory reduction and communication optimization in LLMs/VLMs. [Details](https://github.com/baidu-baige/OmniTraining/tree/master/docs/source/features/moe_all2all_overlap.md)
* **Heterogeneous Parallelism**: Supports different TP/DP sizes per VLM component to improve training throughput. [Details](https://github.com/baidu-baige/OmniTraining/tree/master/docs/source/features/heterogeneous_tp_parallel.md)
* **DP Load Balancing**: Optimizes data parallel load imbalance from data packing, improving multi-node speedup for VLM training. [Details](https://github.com/baidu-baige/OmniTraining/tree/master/docs/source/features/data_parallel_balancing.md)
* **MTP Training**: Multi-head MTP features supporting different model design needs or MTP head extension, with weight sharing/independence options and cascade/serial computation modes.
* **Custom Operators**: High-performance fused operators like FusedDSA, which integrates flashmla and indexer forward operators with custom backward operators (essential for training) to accelerate DSA model training.
* **Heterogeneous Hardware**: Supports NVIDIA GPUs and Kunlun XPUs, with XPU_Plugin to minimize intrusive changes for XPU adaptation.


🔔🔔🔔 Please refer to the [documentation](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/) for detailed usage and features of the framework.

## Supported Models

For model configurations, refer to `configs/models/`. Training launch script examples are available in `examples/`. Visit the [documentation](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/) for more details.

| **Model Type** | **Model Category** | **Models** |
|----------------|-------------------|------------|
| **LLM** | DeepSeek-V2 | deepseek-v2-lite, deepseek-v2 |
| | DeepSeek-V3 | deepseek-v3, deepseek-v32 |
| | LLaMA2 | llama2-7b, llama2-13b, llama2-70b |
| | LLaMA3 | llama3-8b, llama3-70b |
| | LLaMA3.1 | llama3.1-8b, llama3.1-70b, llama3.1-405b |
| | Qwen | qwen-1.8b, qwen-7b, qwen-14b, qwen-72b |
| | Qwen1.5 | qwen1.5-0.5b, qwen1.5-1.8b, qwen1.5-4b, qwen1.5-7b, qwen1.5-14b, qwen1.5-32b, qwen1.5-72b |
| | Qwen2 | qwen2-0.5b, qwen2-1.5b, qwen2-7b, qwen2-72b |
| | Qwen2.5 | qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b, qwen2.5-14b, qwen2.5-32b, qwen2.5-72b |
| | Qwen3 | qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, qwen3-14b, qwen3-32b, qwen3-30b-a3b, qwen3-235b-a22b, qwen3-480b-a35b, qwen3-coder-30b-a3b |
| | Qwen3-Next | qwen3-next-80b-a3b |
| | MiniMax | minimax-m2.1, minimax-m2.5 |
| | MIMO | mimo-7b |
| **VLM** | Qwen2.5-VL | qwen2.5-vl-3b, qwen2.5-vl-7b, qwen2.5-vl-32b, qwen2.5-vl-72b |
| | Qwen3-VL | qwen3-vl-30b-a3b, qwen3-vl-235b-a22b |
| | Qwen3.5 | qwen3.5-35B-A3B, qwen3.5-397B-A17B |
| | ERNIE4.5-VL | ernie4.5vl-28b-a3b |
| | LLaVA-OneVision-1.5 | llava-onevision-1.5-4B |
| | InternVL2.5 | internvl2.5-8b, internvl2.5-26b, internvl2.5-38b, internvl2.5-78b |
| | InternVL3.5 | internvl3.5-8b, internvl3.5-14b, internvl3.5-38b, internvl3.5-30b-a3b, internvl3.5-241b-a28b |
| | CustomCombinedModel | Flexible ViT + LLM backbone configuration ([example](https://github.com/baidu-baige/OmniTraining/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml)) |
| **Diffusion** | WAN2.1 | wan2.1_i2v_14b |
| | WAN2.2 | wan2.2_i2v_a14b |
| **VLA** | Pi | pi0.5 |


## Getting Started

* Quick Start:
  * [Installation](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/get_started/installation.html)
  * [Quick Start for LLM Pretrain](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/llm_tutorial/quick_start_llm_pretrain.html)
  * [Quick Start for LLM SFT](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/llm_tutorial/quick_start_llm_sft.html)
  * [Quick Start for VLM Pretrain](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/vlm_tutorial/quick_start_vlm_pretrain.html)
  * [Quick Start for VLM SFT](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/vlm_tutorial/quick_start_vlm_sft.html)
  * [Quick Start for VLA Training](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/vla_tutorial/quick_start_vla_training.html)
  * [Quick Start for WAN Training](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/wan_tutorial/quick_start_wan_training.html)

* Kunlun XPU Training:
  * [Installation]()
  * [Quick Start]()

## Architecture

```
AIAK-Training-Omni/
├── omni_training/                # Core training framework
│   ├── train/                    # Training entry modules
│   │   ├── pretrain/             # Pretrain entry
│   │   ├── sft/                  # SFT entry
│   │   └── custom/               # Custom task entry
│   ├── models/                   # Model system
│   │   ├── foundation/           # LLM backbones (LLaMA, Qwen, DeepSeek, etc.)
│   │   ├── encoder/              # Vision encoders (ViT, Qwen-VL, InternVL, etc.)
│   │   ├── omni_models/          # Multi-modal model composition
│   │   ├── common/               # Shared layers (norm, projector, etc.)
│   │   ├── custom/               # Custom models (WAN, Pi0.5, etc.)
│   │   └── peft/                 # PEFT support (LoRA)
│   ├── data/                     # Data pipeline
│   │   ├── dp_balance/           # Data parallel load balancing
│   │   ├── multimodal/           # Multimodal data modules
│   │   ├── video/                # Video data processing
│   │   └── lerobot/              # LeRobot data support
│   ├── tokenizer/                # Tokenizer modules
│   └── utils/                    # Utility functions
├── configs/                      # Hydra YAML configurations
│   ├── models/                   # Model configs
│   └── data/                     # Data configs
├── examples/                     # GPU training scripts
├── examples_xpu/                 # Kunlun XPU training scripts
├── tools/                        # Utility tools
├── ops/                          # Custom kernel operators
├── patches/                      # Patches for Megatron-LM and TransformerEngine
│   ├── Megatron-LM_v0.15.0/      # Megatron patches
│   │   ├── megatron/             # Megatron core and training patches
│   │   └── xpu_plugin/           # XPU plugin patches
│   └── TransformerEngine_v2.9/   # TransformerEngine patches
├── tests/                        # Test suite
└── docs/                         # Documentation
```


## Awesome Projects Built with OmniTraining
- [Qianfan-VL: Domain-Enhanced Universal Vision-Language Models](https://github.com/baidubce/Qianfan-VL)
- [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5)

## Contributing
Please read [CONTRIBUTING.md](https://github.com/baidu-baige/OmniTraining/tree/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the Apache License - see the [LICENSE](https://github.com/baidu-baige/OmniTraining/tree/master/LICENSE) file for details.

## Citation
If you use this work, please consider citing:
```bibtex
@misc{OmniTraining 2025,
      title={}, 
      author={},
      year={2025},
      eprint={2512.24077},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={}, 
}
```