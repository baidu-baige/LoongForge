# README
<div align="center">

<h1 align="center">BaigeOmni</h1>
<h3>A modular, scalable, and highly efficient training framework for language and multimodal models.</h3>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://yq02-inf-sci-k8s-a800-aa2ni52-0034.yq02:8080/)[![License](https://img.shields.io/github/license/open-mmlab/mmdeploy.svg)](https://github.com/baidu-baige/BaigeOmni/blob/master/LICENSE)[![Stars](https://img.shields.io/github/stars/baidu-baige/BaigeOmni=social)](https://github.com/baidu-baige/BaigeOmni/tree/master)[![Issues](https://img.shields.io/github/issues-raw/baidu-baige/BaigeOmni)](https://github.com/baidu-baige/BaigeOmni/issues)

</div>

## 📖 About

**BaigeOmni** (evolved from [AIAK-Training-LLM](https://cloud.baidu.com/doc/AIHC/s/Alyo476jr)) is a training framework for large-scale transformer models across diverse modalities and architectures. It supports key stages of the training pipeline, including pre-training, continued pre-training, and Supervised Fine-Tuning (SFT). Through continuous adaptation and performance optimization, BaigeOmni delivers an efficient, easy-to-use, and highly extensible solution for model training.

* **🚀 Comprehensive Model Coverage**: Natively supports mainstream model architectures including LLMs (Large Language Models), VLMs (Vision-Language), VLAs (Vision-Language-Action), and Diffusion Models. Its flexible composition abstraction makes adding new multi-modal variants effortless.
* **⚡ Performance-Driven Optimization**: Built upon Megatron-LM with significant enhancements. BaigeOmni introduces advanced optimizations in communication, computation overlap, and memory management, further optimizing training performance to significantly reduce training costs and accelerate model development.
* **🧪 Heterogeneous Hardware Support**: Provides native, high-performance support for both NVIDIA GPUs and Kunlun XPUs, ensuring seamless migration and stable training at scale across diverse hardware clusters.

## 🔥 Latest News
- **[2026/03]** 🎉 Initial release of the BaigeOmni framework!

## ✨ Key Features

* **Flexible Composition**: A configuration-driven approach to assemble VLMs using interchangeable ViT and LLM components.
* **Heterogeneous Strategies**: Enables assigning independent configurations—such as Tensor/Data Parallel sizes and recomputation layers—to different model components (e.g., Vision Encoder vs. LLM) for optimal throughput and memory efficiency.
* **MoE Optimization**: Integrates All2All communication, activation offloading, and computation overlap to optimize memory usage and communication in large-scale MoE models.
* **DP Load Balancing**: Optimizes data parallel imbalances caused by data packing, improving multi-node scaling efficiency.
* **FP8 Precision Training**: Validated FP8 training pipelines for both LLMs and VLMs to accelerate computation and reduce memory footprint.
* **MTP Training**: Supports Multi-Task Pretraining (MTP) with extensible heads, offering options for shared/independent weights and cascade/serial computation.
* **Custom Fused Operators**: High-performance fused operators like FusedDSA, which integrates flashmla and indexer forward operators with custom backward operators (essential for training) to accelerate DSA model training.
* **Versatile Pipeline & Tools**: Out-of-the-box support for Pretrain, MidTrain, SFT, and LoRA. Includes tools for dataset processing (e.g., format conversion, packing..) and bidirectional Megatron ↔ HuggingFace weight conversion.
* **Heterogeneous Hardware**: Supports training on both NVIDIA GPUs and Kunlun XPUs via a minimally intrusive plugin design.
* **Native Hugging Face checkpoint load/save support**, eliminating the need for offline Megatron conversion.

*(🔔🔔🔔 Please refer to our [Official Documentation](https://baidu-baige.github.io/BaigeOmni/) for detailed tutorials.)*

## 🚀 Ongoing & Upcoming

* **Expanded support for embodied AI models**, including GR00T N1.6, DreamZero, and LingBot VA.
* **Further performance acceleration for diffusion models** such as WAN.
* **Expanded foundation model support** (e.g., Kimi 2.5, GLM5).
* **Decoupled encoder-decoder training** to accelerate VLM workflows.
* **Further enhanced kernel performance** (e.g., DSA, GatedAttention).
* **Adaptive FP8 precision** and **advanced MoE load-balancing** strategies.
* **Real-world application of MTP scaling** to improve speculative decoding acceptance rates.
* ...

## 🛠️ Getting Started

**Quick Start Guides:**
* [Installation](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/get_started/installation.md)
* [Supported Models](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/get_started/support_model.md)
* [Quick Start for LLM Pretrain](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/llm_tutorial/quick_start_llm_pretrain.md)
* [Quick Start for LLM SFT](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/llm_tutorial/quick_start_llm_sft.md)
* [Quick Start for VLM Pretrain](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/vlm_tutorial/quick_start_vlm_pretrain.md)
* [Quick Start for VLM SFT](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/vlm_tutorial/quick_start_vlm_sft.md)
* [Quick Start for VLA Training](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/vla_tutorial/quick_start_vla_training.md)
* [Quick Start for WAN Training](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/wan_tutorial/quick_start_wan_training.md)

**Kunlun XPU Platform:**
* [README](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/README.md)
* [Installation](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/install_p800.md)
* [Quick Start for LLM Pretrain](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_llm_pretrain_p800.md)
* [Quick Start for LLM SFT](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_llm_sft_p800.md)
* [Quick Start for VLM](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_vlm_p800.md)
* [Quick Start for VLA](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_vla_p800.md)

## 🏛️ Supported Models

BaigeOmni supports a massive array of state-of-the-art models. Check out `configs/models/` for YAML configurations and `examples/` for launch scripts.


| **Modality** | **Architectures** | **Models** |
|---------------|------------------|------------|
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
| | Qwen3.5 | qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, qwen3.5-9b, qwen3.5-27b, qwen3.5-35B-A3B, qwen3.5-122B-A10B, qwen3.5-397B-A17B |
| | Qwen3.6 | qwen3.6-35B-A3B |
| | ERNIE4.5-VL | ernie4.5vl-28b-a3b |
| | LLaVA-OneVision-1.5 | llava-onevision-1.5-4B |
| | InternVL2.5 | internvl2.5-8b, internvl2.5-26b, internvl2.5-38b, internvl2.5-78b |
| | InternVL3.5 | internvl3.5-8b, internvl3.5-14b, internvl3.5-38b, internvl3.5-30b-a3b, internvl3.5-241b-a28b |
| | CustomCombinedModel | Flexible ViT + LLM backbone configuration ([example](https://github.com/baidu-baige/BaigeOmni/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml)) |
| **Diffusion** | WAN2.2 | wan2.2_i2v_a14b |
| **VLA** | Pi | pi0.5 |


## 🏗️ Architecture Overview
```
BaigeOmni/
├── baige_omni/                   # Core training framework
│   ├── train/                    # Training entry points
│   ├── models/                   # Unified model abstractions (LLM, Encoder, VLM)
│   │   ├── common/               # Shared layers and utilities
│   │   ├── encoder/              # Vision encoders (ViT, Qwen-VL, InternVL, Ernie4.5VL, LLaVA-OneVision, etc.)
│   │   ├── foundation/           # LLM backbones (LLaMA, Qwen, DeepSeek, InternLM, etc.)
│   │   ├── omni_models/          # Multi-modal model composition
│   │   ├── diffusion/            # Diffusion model support
│   │   └── embodied/             # Embodied AI model support
│   ├── data/                     # Data pipelines and load balancing
│   ├── tokenizer/                # Tokenizer modules
│   └── utils/                    # Utility functions
├── configs/                      # Hydra-based YAML configurations
│   ├── models/                   # Model configs
│   └── data/                     # Data configs
├── examples/                     # GPU launch scripts
├── examples_xpu/                 # Kunlun XPU launch scripts
├── tools/                        # Utility tools
├── ops/                          # Custom fused CUDA/C++ operators
├── patches/                      # Framework adaptations (Megatron, TransformerEngine)
├── tests/                        # Test suite
└── docs/                         # Documentation
```

## 🌟 Powered by BaigeOmni

**Open-Source Ecosystem:**
* [Qianfan-VL: Domain-Enhanced Universal Vision-Language Models](https://github.com/baidubce/Qianfan-VL)
* [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5)

**Enterprise Scale & Performance:**

Before becoming an open-source project, BaigeOmni had already empowered numerous enterprise use cases with its robust training acceleration and scaling capabilities:
* Powers proprietary large-scale models across diverse industries, including **Education, Code Generation, and Embodied AI**.
* Typically achieves a **30%+ average speedup** over standard customer baselines through systemic optimizations.
* Seamlessly supports ultra-large cluster training scaling up to **5,000 XPUs**.

## 🤝 Contributing

We heartily welcome community contributions! Whether it's reporting bugs, proposing features, or submitting code, please read our [Contributing Guidelines](docs/source/CONTRIBUTING.md) before submitting a Pull Request.

## 📄 License

BaigeOmni is released under the [Apache License 2.0](https://github.com/baidu-baige/BaigeOmni/blob/master/LICENSE). 

Some files in this repository are derived from third-party open-source projects. Please refer to the specific file headers for their respective copyright, license notices, and attribution requirements.

## 📝 Citation

If you find BaigeOmni helpful in your research or production, please consider citing our repository:

```bibtex
@software{BaigeOmni2026,
      title={BaigeOmni: A modular, scalable, and highly efficient training framework for language and multimodal models}, 
      author={{The BaigeOmni Authors}},
      year={2026},
      url={https://github.com/baidu-baige/BaigeOmni},
}
```

## 🙏 Acknowledgments

BaigeOmni is built upon excellent open-source projects including but not limited to Megatron-LM, PyTorch, and Transformers. We thank the open-source community for their invaluable foundational work.