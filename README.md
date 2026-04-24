# README
<div align="center">

<h1 align="center">LoongForge</h1>
<h4>A modular, scalable, and highly efficient training framework for language, multimodal, and embodied models.</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://loongforge.readthedocs.io/en/latest/index.html)[![License](https://img.shields.io/github/license/open-mmlab/mmdeploy.svg)](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE)[![Stars](https://img.shields.io/github/stars/baidu-baige/LoongForge=social)](https://github.com/baidu-baige/LoongForge/tree/master)[![Issues](https://img.shields.io/github/issues-raw/baidu-baige/LoongForge)](https://github.com/baidu-baige/LoongForge/issues)

</div>

## 📖 About

> 🐉 **The Baige "Loong" Family:** LoongForge (alongside [LoongFlow](https://github.com/baidu-baige/LoongFlow)) is part of the Loong open-source series from Baidu's Baige AI infrastructure platform, named after the traditional Chinese loong boat.

**LoongForge** is a training framework for large-scale transformer models across diverse modalities and architectures. It supports key stages of the training pipeline, including pre-training, continued pre-training, and supervised fine-tuning (SFT). Built upon Megatron-LM with significant enhancements, LoongForge delivers an efficient, easy-to-use, and highly extensible solution for model training.

* **🚀 Comprehensive Model Coverage**: Natively supports mainstream model architectures including LLMs (Large Language Models), VLMs (Vision-Language), VLAs (Vision-Language-Action), and Diffusion Models. Its flexible composition abstraction makes adding new multi-modal variants effortless.
* **⚡ Performance-Driven Optimization**: Provides advanced optimizations in parallelism and memory management, significantly reducing training costs and accelerating model development.
* **🧪 Heterogeneous Hardware Support**: Provides native, high-performance support for both NVIDIA GPUs and Kunlun XPUs, ensuring seamless migration and stable training at scale across diverse hardware clusters.

## 🔥 Latest News
- **[2026/04]** 🎉 Initial release of the LoongForge framework!

## ✨ Key Features

* **Flexible Composition**: A configuration-driven approach to assemble VLMs using interchangeable ViT and LLM components.
* **Heterogeneous Parallelism**: Enables assigning independent configurations—such as Tensor/Data Parallel sizes and recomputation layers—to different model components (e.g., Vision Encoder vs. LLM) for optimal throughput and memory efficiency.
* **Decoupled Encoder-Decoder Training**: Separates vision encoder and LLM into independent tasks, eliminating encoder-induced pipeline bubbles and preventing ViT computation from blocking LLM throughput.
* **DP Load Balancing**: Leverages a load-aware data redistribution algorithm to optimize data parallel imbalances caused by data packing, improving multi-node scaling efficiency.
* **MoE A2A Optimization**: Overlaps All2All communication, activation offloading, and computation to optimize memory usage and communication in large-scale MoE models, achieving lower memory footprint than upstream Megatron-LM.
* **Custom Fused Operators**: High-performance fused operators like FusedDSA, which integrates flashmla and indexer forward operators with custom backward operators (essential for training) to accelerate DSA model training. Currently the TileLang-based operators are open-sourced.
* **Adaptive FP8 precision**: End-to-end FP8 training support for both LLMs and VLMs, further enhanced with adaptive FP8 that automatically determines whether to enable FP8 per operator based on GEMM shape and computational efficiency to maximize training performance.
* **Flexible Checkpoint Conversion**: Supports both **offline bidirectional Megatron ↔ HuggingFace weight conversion** and **native online HuggingFace checkpoint load/save**, eliminating format barriers throughout the training workflow.
* **Versatile Pipeline & Tools**: Out-of-the-box support for Pretrain, MidTrain, SFT, and LoRA, with built-in tools for dataset processing such as format conversion and packing.
* **Heterogeneous Hardware**: Supports training on both NVIDIA GPUs and Kunlun XPUs via a minimally intrusive plugin design.

*(🔔🔔🔔 Please refer to our [Official Documentation](https://loongforge.readthedocs.io/en/latest/index.html) for detailed tutorials.)*

## 🚀 Ongoing & Upcoming

* **Expanded foundation model support** (e.g., Kimi 2.6).
* **Expanded support for embodied AI models**, including DreamZero, and LingBot VA.
* **Further performance acceleration for diffusion models** such as WAN.
* **Further enhanced kernel performance**.
* **Optimized Full Heterogeneous DP memory overhead** and improved parallelism strategy compatibility.
* **Advanced MoE load-balancing** strategies.
* **Support for INT4 quantization-aware training**, such as the approach proposed by Kimi 2.5.
* **Enhanced long-sequence training** with optimization techniques such as ChunkPipe scheduling and Context Parallelism (CP) to improve throughput and scalability.
* **Real-world application of MTP scaling** to improve speculative decoding acceptance rates.
* ...

## 🛠️ Getting Started

### Installation (Docker Quick Start)

For detailed installation steps (source install, XPU, development setup), see the [Installation Guide](https://loongforge.readthedocs.io/en/latest/get_started/installation.html). Below is a quick start using Docker:

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git

docker build --build-arg COMPILE_ENV=hopper --build-arg ENABLE_LEROBOT=false -t loongforge:latest -f ./LoongForge/docker/Dockerfile .
```

**Tutorials:**
* [Supported Models](https://loongforge.readthedocs.io/en/latest/get_started/support_model.html)
* [Quick Start for LLM Pretrain](https://loongforge.readthedocs.io/en/latest/llm_tutorial/quick_start_llm_pretrain.html)
* [Quick Start for LLM SFT](https://loongforge.readthedocs.io/en/latest/llm_tutorial/quick_start_llm_sft.html)
* [Quick Start for VLM Pretrain](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html)
* [Quick Start for VLM SFT](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_sft.html)
* [Quick Start for VLA Training](https://loongforge.readthedocs.io/en/latest/vla_tutorial/quick_start_vla_training.html)
* [Quick Start for WAN Training](https://loongforge.readthedocs.io/en/latest/wan_tutorial/quick_start_wan_training.html)

**Kunlun XPU Platform:**
* [README](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/index.html)
* [Installation](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/install_p800.html)
* [Quick Start for LLM Pretrain](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_llm_pretrain_p800.html)
* [Quick Start for LLM SFT](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_llm_sft_p800.html)
* [Quick Start for VLM](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_vlm_p800.html)
* [Quick Start for VLA](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_vla_p800.html)

## 🏛️ Supported Models

LoongForge supports a massive array of state-of-the-art models. Check out `configs/models/` for YAML configurations and `examples/` for launch scripts.


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
| | GLM | glm5 |
| **VLM** | Qwen2.5-VL | qwen2.5-vl-3b, qwen2.5-vl-7b, qwen2.5-vl-32b, qwen2.5-vl-72b |
| | Qwen3-VL | qwen3-vl-30b-a3b, qwen3-vl-235b-a22b |
| | Qwen3.5 | qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, qwen3.5-9b, qwen3.5-27b, qwen3.5-35B-A3B, qwen3.5-122B-A10B, qwen3.5-397B-A17B |
| | Qwen3.6 | qwen3.6-27B, qwen3.6-35B-A3B |
| | ERNIE4.5-VL | ernie4.5vl-28b-a3b |
| | LLaVA-OneVision-1.5 | llava-onevision-1.5-4B |
| | InternVL2.5 | internvl2.5-8b, internvl2.5-26b, internvl2.5-38b, internvl2.5-78b |
| | InternVL3.5 | internvl3.5-8b, internvl3.5-14b, internvl3.5-38b, internvl3.5-30b-a3b, internvl3.5-241b-a28b |
| | CustomCombinedModel | Flexible ViT + LLM backbone configuration ([example](https://github.com/baidu-baige/LoongForge/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml)) |
| **Diffusion** | WAN2.2 | wan2.2_i2v_a14b |
| **VLA** | Pi | pi0.5 |


## 🏗️ Architecture Overview
```
LoongForge/
├── loongforge/                   # Core training framework
│   ├── train/                    # Training entry points & trainers
│   │   ├── pretrain/             # Pretrain implementations (LLM, VLM)
│   │   ├── sft/                  # SFT implementations (LLM, VLM, InternVL, ERNIE)
│   │   ├── diffusion/            # Diffusion model trainers (WAN)
│   │   └── embodied/             # Embodied AI model trainers (Pi0.5, GR00T)
│   ├── models/                   # Unified model abstractions
│   │   ├── foundation/           # LLM backbones (LLaMA, Qwen, DeepSeek, InternLM, MiniMax, MIMO, GLM)
│   │   ├── encoder/              # Vision encoders (ViT, Qwen-VL, InternVL, ERNIE4.5-VL, LLaVA-OV)
│   │   ├── omni_models/          # Multi-modal composition (encoder + projector + decoder)
│   │   ├── diffusion/            # Diffusion models (WAN)
│   │   ├── embodied/             # Embodied AI models (Pi0.5, GR00T N1.6)
│   │   └── common/               # Shared layers and utilities
│   ├── data/                     # Data pipelines
│   │   ├── dp_balance/           # Data-parallel load balancing
│   │   ├── multimodal/           # Multi-modal data processing
│   │   └── video/                # Video data processing
│   ├── tokenizer/                # Tokenizer modules
│   └── utils/                    # Utility functions (config map, constants, etc.)
├── third_party/
│   └── Loong-Megatron/           # Patched Megatron-LM (git submodule)
├── configs/                      # Hydra-based YAML configurations
│   ├── models/                   # Model architecture configs
│   └── data/                     # Data configs
├── examples/                     # GPU launch scripts (pretrain / sft / checkpoint_convert)
├── examples_xpu/                 # Kunlun XPU launch scripts
├── tools/                        # Utility tools
│   ├── convert_checkpoint/       # HuggingFace ↔ Megatron checkpoint conversion
│   ├── data_preprocess/          # Data preprocessing utilities
│   └── dist_checkpoint/          # Distributed checkpoint utilities
├── ops/                          # Custom fused operators (TileLang open-sourced)
├── patches/                      # TransformerEngine patch files
├── docker/                       # Dockerfiles (GPU & XPU)
├── tests/                        # E2E test suite (YAML-driven)
└── docs/                         # Documentation
```

## 🌟 Powered by LoongForge

**Open-Source Ecosystem:**
* [Qianfan-VL: Domain-Enhanced Universal Vision-Language Models](https://github.com/baidubce/Qianfan-VL)
* [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5) -  Built upon an earlier version of LoongForge.

**Enterprise Scale & Performance:**

Before becoming an open-source project, LoongForge had already empowered numerous enterprise use cases with its robust training acceleration and scaling capabilities:
* Powers proprietary large-scale models across diverse industries, including **Education, Code Generation, and Embodied AI**.
* Typically achieves a **30%+ average speedup** over standard customer baselines through systemic optimizations.
* Seamlessly supports ultra-large cluster training scaling up to **5,000 XPUs**.

## 🤝 Contributing

We heartily welcome community contributions! Whether it's reporting bugs, proposing features, or submitting code, please read our [Contributing Guidelines](https://github.com/baidu-baige/LoongForge/blob/master/CONTRIBUTING.md) before submitting a Pull Request.

## 📄 License

LoongForge is released under the [Apache License 2.0](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE). 

Some files in this repository are derived from third-party open-source projects. Please refer to the specific file headers for their respective copyright, license notices, and attribution requirements.

## 📝 Citation

If you find LoongForge helpful in your research or production, please consider citing our repository:

```bibtex
@software{LoongForge2026,
      title={LoongForge: A modular, scalable, and highly efficient training framework for language, multimodal, and embodied models.}, 
      author={{The LoongForge Authors}},
      year={2026},
      url={https://github.com/baidu-baige/LoongForge},
}
```

## 🙏 Acknowledgments

LoongForge is built upon NVIDIA's Megatron-LM. During development, we also referenced and drew inspiration from several excellent open-source projects, including but not limited to Transformers, LLaMA-Factory, and Megatron-Bridge. We sincerely thank these communities for their outstanding contributions.

